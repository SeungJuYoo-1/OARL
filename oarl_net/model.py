"""OARL-NET: organelle-aware segmentation + perturbation classification network.

One backbone, two outputs. A 2.5D stem collapses the 7-slice refractive-index stack
to 32 channels, a stride-1 ResNet-34 and FPN produce the pyramid, a shared decoder
trunk feeds three organelle heads (nucleus / lysosome / mitochondria), and a separate
attention-pooled classification head branches off P5 -- upstream of the decoder, so
classification does not depend on the segmentation heads.

    classification_mode=False -> forward() returns {'segmentation': (B,3,H,W) logits,
                                                    'heatmaps': (B,3,H,W) logits}
    classification_mode=True  -> forward() returns (B,1) perturbation logits

The auxiliary heatmap branch predicts organelle centroids during segmentation training.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# -------------------------------------------------------------------------
# Attention modules
# -------------------------------------------------------------------------

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super().__init__()
        reduced = max(1, in_planes // ratio)
        self.shared = nn.Sequential(
            nn.Conv2d(in_planes, reduced, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(reduced, in_planes, 1, bias=False)
        )

    def forward(self, x):
        avg = self.shared(F.adaptive_avg_pool2d(x, 1))
        max_ = self.shared(F.adaptive_max_pool2d(x, 1))
        return x * torch.sigmoid(avg + max_)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max_, _ = torch.max(x, dim=1, keepdim=True)
        return x * torch.sigmoid(self.conv(torch.cat([avg, max_], dim=1)))


class CBAM(nn.Module):
    def __init__(self, in_planes):
        super().__init__()
        self.ca = ChannelAttention(in_planes)
        self.sa = SpatialAttention()

    def forward(self, x):
        return self.sa(self.ca(x))


# -------------------------------------------------------------------------
# 2.5D preprocessing
# -------------------------------------------------------------------------

class Z25DPreprocessing(nn.Module):
    """
    Multi-branch 2.5D conv: local (Conv3D 3x1x1) + global (Conv3D Zx1x1),
    Z-collapsed and concatenated → 32 channels.
    """
    def __init__(self, in_channels=1, out_channels=32, z_slices=9):
        super().__init__()
        self.z_slices = z_slices
        self.local_conv3d = nn.Conv3d(in_channels, out_channels // 2,
                                      kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False)
        self.local_bn = nn.BatchNorm3d(out_channels // 2)
        self.global_conv3d = nn.Conv3d(in_channels, out_channels // 2,
                                       kernel_size=(z_slices, 1, 1), bias=False)
        self.global_bn = nn.BatchNorm3d(out_channels // 2)
        self.fuse_bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(1)
        local = self.relu(self.local_bn(self.local_conv3d(x))).mean(dim=2, keepdim=True)
        global_ctx = self.relu(self.global_bn(self.global_conv3d(x)))
        features = self.relu(self.fuse_bn(torch.cat([local, global_ctx], dim=1)))
        return features.squeeze(2)


# -------------------------------------------------------------------------
# Backbone
# -------------------------------------------------------------------------

class ResNet34Backbone(nn.Module):
    """
    ResNet34 with stride-1 stem and maxpool replaced by identity.
    Returns [stem, P2, P3, P4, P5] feature maps.
    """
    def __init__(self, in_channels=32):
        super().__init__()
        # Segmentation pretraining starts from random weights.
        resnet = models.resnet34(weights=None)

        # The stem takes the 32 preprocessed channels, not RGB, and keeps stride 1 so
        # the decoder can return full-resolution masks.
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=1, padding=3, bias=False)

        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = nn.Identity()
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.stage_channels = [64, 64, 128, 256, 512]

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        stem = x
        x = self.maxpool(x)
        p2 = self.layer1(stem)
        p3 = self.layer2(p2)
        p4 = self.layer3(p3)
        p5 = self.layer4(p4)
        return [stem, p2, p3, p4, p5]


# -------------------------------------------------------------------------
# Feature pyramid + shared decoder trunk
# -------------------------------------------------------------------------

class FeaturePyramidNetwork(nn.Module):
    def __init__(self, in_channels=[64, 64, 128, 256, 512], out_channels=256):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c, out_channels, 1, bias=False) for c in in_channels
        ])
        self.output_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for _ in in_channels
        ])

    def forward(self, backbone_features):
        laterals = [lc(f) for f, lc in zip(backbone_features, self.lateral_convs)]
        prev = laterals[-1]
        for i in range(len(laterals) - 2, -1, -1):
            upsampled = F.interpolate(prev, size=laterals[i].shape[2:],
                                      mode='bilinear', align_corners=False)
            prev = self.output_convs[i](upsampled + laterals[i])
        return prev


# -------------------------------------------------------------------------
# Shared decoder trunk
# -------------------------------------------------------------------------

class SharedDecoderTrunk(nn.Module):
    """Two residual blocks: 256→128→64."""
    def __init__(self, in_channels=256, out_channels=64):
        super().__init__()
        self.block1_conv1 = nn.Conv2d(in_channels, 128, 3, padding=1, bias=False)
        self.block1_gn1   = nn.GroupNorm(8, 128)
        self.block1_conv2 = nn.Conv2d(128, 128, 3, padding=1, bias=False)
        self.block1_gn2   = nn.GroupNorm(8, 128)
        self.block1_skip  = nn.Sequential(nn.Conv2d(in_channels, 128, 1, bias=False),
                                          nn.GroupNorm(8, 128))
        self.block2_conv1 = nn.Conv2d(128, out_channels, 3, padding=1, bias=False)
        self.block2_gn1   = nn.GroupNorm(8, out_channels)
        self.block2_conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.block2_gn2   = nn.GroupNorm(8, out_channels)
        self.block2_skip  = nn.Sequential(nn.Conv2d(128, out_channels, 1, bias=False),
                                          nn.GroupNorm(8, out_channels))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Apply ReLU after adding the residual, as in the trained decoder.
        m1 = self.block1_gn2(self.block1_conv2(
                self.relu(self.block1_gn1(self.block1_conv1(x)))))
        x = self.relu(m1 + self.block1_skip(x))
        m2 = self.block2_gn2(self.block2_conv2(
                self.relu(self.block2_gn1(self.block2_conv1(x)))))
        x = self.relu(m2 + self.block2_skip(x))
        return x

# -------------------------------------------------------------------------
# Organelle heads
# -------------------------------------------------------------------------

class NucleusHead(nn.Module):
    def __init__(self, in_channels=64):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1, bias=True)
        )

    def forward(self, x):
        return self.head(x)


class MultiScaleContext(nn.Module):
    def __init__(self, in_channels=64, scales=[1, 2, 4]):
        super().__init__()
        self.scales = scales
        self.scale_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=s, dilation=s, bias=False),
                nn.GroupNorm(8, in_channels), nn.ReLU(inplace=True)
            ) for s in scales
        ])
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * len(scales), in_channels, 1, bias=False),
            nn.GroupNorm(8, in_channels), nn.ReLU(inplace=True)
        )
        self.scale_attention = ChannelAttention(in_channels)

    def forward(self, x):
        feats = torch.cat([sc(x) for sc in self.scale_convs], dim=1)
        return x + self.scale_attention(self.fusion(feats))


class LysosomeHead(nn.Module):
    def __init__(self, in_channels=64):
        super().__init__()
        self.feature_enhancer = MultiScaleContext(in_channels)
        self.attention = CBAM(in_channels)
        self.context_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, in_channels), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=2, dilation=2, bias=False),
            nn.GroupNorm(8, in_channels), nn.ReLU(inplace=True)
        )
        self.reduce1 = nn.Sequential(nn.Conv2d(in_channels, 32, 3, padding=1, bias=False),
                                     nn.GroupNorm(4, 32), nn.ReLU(inplace=True))
        self.reduce2 = nn.Sequential(nn.Conv2d(32, 16, 3, padding=1, bias=False),
                                     nn.GroupNorm(2, 16), nn.ReLU(inplace=True))
        self.final   = nn.Conv2d(16, 1, 1, bias=True)
        self.skip1   = nn.Sequential(nn.Conv2d(in_channels, 32, 1, bias=False),
                                     nn.GroupNorm(4, 32))
        self.skip2   = nn.Sequential(nn.Conv2d(32, 16, 1, bias=False),
                                     nn.GroupNorm(2, 16))
        nn.init.constant_(self.final.bias, -2.0)
        nn.init.normal_(self.final.weight, std=0.01)

    def forward(self, x):
        x = self.attention(self.feature_enhancer(x))
        ctx = self.context_conv(x)
        x1 = self.reduce1(ctx) + self.skip1(ctx)
        x2 = self.reduce2(x1) + self.skip2(x1)
        return self.final(x2)

class MitochondriaHead(nn.Module):
    def __init__(self, in_channels=64):
        super().__init__()
        self.feature_enhancer = MultiScaleContext(in_channels, scales=[1, 2, 4, 6])
        self.attention = CBAM(in_channels)
        self.context_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, in_channels), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=3, dilation=3, bias=False),
            nn.GroupNorm(8, in_channels), nn.ReLU(inplace=True)
        )
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 48, 3, padding=1, bias=False),
            nn.GroupNorm(8, 48), nn.ReLU(inplace=True),
            nn.Conv2d(48, 32, 3, padding=1, bias=False),
            nn.GroupNorm(8, 32), nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(32, 1, 1, bias=True)
        )
        nn.init.constant_(self.head[-1].bias, -2.0)
        nn.init.normal_(self.head[-1].weight, std=0.01)

    def forward(self, x):
        x = self.attention(self.feature_enhancer(x))
        return self.head(self.context_conv(x))


class HeatmapHead(nn.Module):
    """Auxiliary organelle-centroid detection branch.

    Predicts one heatmap per organelle whose peaks mark centroids, trained against
    Gaussians placed on the ground-truth connected components. It shares the decoder
    trunk with the mask heads and is an auxiliary objective, not an output of the
    published pipeline -- masks come from `organelle_heads`.

    The final bias starts at -4.0: centroid peaks are far rarer than background, and a
    neutral bias makes the branch spend its early epochs unlearning a dense prior.
    """

    def __init__(self, in_channels=64, num_organelles=3):
        super().__init__()
        self.feature_enhancer = MultiScaleContext(in_channels)
        self.heatmap_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1, bias=False),
            nn.GroupNorm(4, 32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1, bias=False),
            nn.GroupNorm(2, 16), nn.ReLU(inplace=True),
            nn.Conv2d(16, num_organelles, 1, bias=True)
        )
        nn.init.constant_(self.heatmap_conv[-1].bias, -4.0)
        nn.init.normal_(self.heatmap_conv[-1].weight, std=0.01)

    def forward(self, x):
        return self.heatmap_conv(self.feature_enhancer(x))


# -------------------------------------------------------------------------
# Classification head
# -------------------------------------------------------------------------

class AttentionPool(nn.Module):
    def __init__(self, in_ch, num_heads=8, temperature=1.0):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = temperature
        self.head_dim = in_ch // num_heads
        assert in_ch % num_heads == 0
        self.query = nn.Conv2d(in_ch, in_ch, 1)
        self.key   = nn.Conv2d(in_ch, in_ch, 1)
        self.value = nn.Conv2d(in_ch, in_ch, 1)
        self.proj  = nn.Conv2d(in_ch, in_ch, 1)
        self.norm  = nn.LayerNorm(in_ch)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        q = self.query(x).view(B, self.num_heads, self.head_dim, N).transpose(-2, -1)
        k = self.key(x).view(B, self.num_heads, self.head_dim, N).transpose(-2, -1)
        v = self.value(x).view(B, self.num_heads, self.head_dim, N).transpose(-2, -1)
        attn = F.softmax(torch.matmul(q, k.transpose(-2, -1)) /
                         (self.head_dim ** 0.5 * self.temperature), dim=-1)
        out = torch.matmul(attn, v).transpose(-2, -1).contiguous().view(B, C, H, W)
        out = self.proj(out)
        pooled = F.adaptive_avg_pool2d(out, 1).view(B, C) + \
                 0.5 * F.adaptive_max_pool2d(out, 1).view(B, C)
        return self.norm(pooled)


class ClassificationHead(nn.Module):
    def __init__(self, in_channels=512, hidden_dims=(256, 128),
                 dropout=0.5, num_heads=8):
        super().__init__()
        self.pool = AttentionPool(in_channels, num_heads=num_heads)
        self.feature_attention = nn.Sequential(
            nn.Linear(in_channels, in_channels // 4), nn.ReLU(),
            nn.Linear(in_channels // 4, in_channels), nn.Sigmoid()
        )
        layers = []
        prev = in_channels
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h),
                       nn.ReLU(inplace=True), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.classifier = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, p5):
        pooled = self.pool(p5)
        attended = pooled * self.feature_attention(pooled)
        return self.classifier(attended)


# -------------------------------------------------------------------------
# Network
# -------------------------------------------------------------------------

class OARLNet(nn.Module):
    ORGANELLE_HEADS = {
        'nucleus': NucleusHead,
        'lysosome': LysosomeHead,
        'mitochondria': MitochondriaHead,
    }

    def __init__(self, z_slices=7, organelles=('nucleus', 'lysosome', 'mitochondria'),
                 classification_mode=False, enable_heatmap_head=True,
                 fpn_channels=256, decoder_channels=64):
        super().__init__()
        self.z_slices = z_slices
        self.organelles = list(organelles)
        self.classification_mode = classification_mode
        self.enable_heatmap_head = enable_heatmap_head

        self.preprocessing = Z25DPreprocessing(1, 32, z_slices)
        self.backbone = ResNet34Backbone(32)
        self.fpn = FeaturePyramidNetwork([64, 64, 128, 256, 512], fpn_channels)
        self.decoder = SharedDecoderTrunk(fpn_channels, decoder_channels)

        unknown = set(self.organelles) - set(self.ORGANELLE_HEADS)
        if unknown:
            raise ValueError(f"no head defined for {sorted(unknown)}; "
                             f"known organelles are {sorted(self.ORGANELLE_HEADS)}")
        self.organelle_heads = nn.ModuleDict(
            {org: self.ORGANELLE_HEADS[org](decoder_channels) for org in self.organelles})

        if enable_heatmap_head:
            self.heatmap_head = HeatmapHead(decoder_channels, len(self.organelles))

        self.classification_head = ClassificationHead(512, (256, 128), 0.5, 8)
        self._initialize_weights()

    def _initialize_weights(self):
        for name, m in self.named_modules():
            # The backbone brings its own torchvision ResNet init; leave it alone.
            if name.startswith('backbone'):
                continue
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                # The heatmap head sets its own output init in its constructor.
                if 'heatmap' in name and getattr(m, 'out_channels', 0) <= 16:
                    nn.init.normal_(m.weight, std=0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, -2.0)
                # Segmentation logits start at a negative bias so the first epochs
                # predict mostly background (organelles are a few % of pixels).
                elif 'final' in name or (hasattr(m, 'out_channels') and m.out_channels == 1):
                    nn.init.xavier_normal_(m.weight, gain=0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, -1.0)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.preprocessing(x)
        backbone_features = self.backbone(x)

        if self.classification_mode:
            # P5, upstream of the decoder: classification does not depend on the
            # segmentation heads.
            return self.classification_head(backbone_features[-1])

        decoder_features = self.decoder(self.fpn(backbone_features))
        outputs = {'segmentation': torch.cat(
            [self.organelle_heads[org](decoder_features) for org in self.organelles], dim=1)}
        if self.enable_heatmap_head:
            outputs['heatmaps'] = self.heatmap_head(decoder_features)
        return outputs


def create_model(model_config):
    """Build an OARLNet from a config.ModelConfig."""
    return OARLNet(
        z_slices=model_config.z_slices,
        organelles=model_config.organelles,
        classification_mode=model_config.classification_mode,
        enable_heatmap_head=model_config.enable_heatmap_head,
        fpn_channels=model_config.fpn_channels,
        decoder_channels=model_config.decoder_channels)
