"""Training losses.

Per organelle: BCE on the logits plus a Dice term, summed, then weighted across
organelles. The weights compensate for area -- lysosome and mitochondria are punctate
and would otherwise contribute almost nothing to the gradient next to the nucleus.

The loss belongs to the trainer, not to the network: it holds no learned state and is
not needed for inference.
"""
import torch
import torch.nn as nn


class SegmentationLoss(nn.Module):
    def __init__(self, organelles, class_weights=(0.5, 2.5, 2.0)):
        super().__init__()
        if len(class_weights) != len(organelles):
            raise ValueError(f"{len(class_weights)} class weights for "
                             f"{len(organelles)} organelles")
        self.organelles = list(organelles)
        self.class_weights = tuple(float(w) for w in class_weights)
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets):
        """-> (total loss, per-organelle components for logging)"""
        total = 0.0
        components = {}
        for i, organelle in enumerate(self.organelles):
            organelle_logits = logits[:, i:i + 1]
            target = targets[:, i:i + 1]
            bce = self.bce(organelle_logits, target).mean()
            dice = self._dice(torch.sigmoid(organelle_logits), target)
            weighted = self.class_weights[i] * (bce + dice)
            components[f'{organelle}_bce'] = bce.item()
            components[f'{organelle}_dice'] = dice.item()
            components[f'{organelle}_total'] = weighted.item()
            total = total + weighted
        components['total'] = total.item()
        return total, components

    @staticmethod
    def _dice(probs, target, smooth=1e-6):
        p, t = probs.flatten(), target.flatten()
        return 1.0 - (2.0 * (p * t).sum() + smooth) / (p.sum() + t.sum() + smooth)


class HeatmapLoss(nn.Module):
    """Penalty-reduced focal loss (CornerNet) for the centroid heatmaps.

    Targets are Gaussians, so "negative" pixels near a peak are penalised less than
    distant ones -- `(1 - target) ** beta` is that discount.
    """

    def __init__(self, alpha=2.0, beta=4.0, normalize_by_objects=False):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.normalize_by_objects = normalize_by_objects

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        positive = targets.eq(1).float()
        negative = targets.lt(1).float()
        pos_loss = torch.log(probs + 1e-8) * torch.pow(1 - probs, self.alpha) * positive
        neg_loss = (torch.log(1 - probs + 1e-8) * torch.pow(probs, self.alpha)
                    * torch.pow(1 - targets, self.beta) * negative)
        if self.normalize_by_objects:
            return -(pos_loss + neg_loss).sum() / positive.sum().clamp(min=1.0)
        return -(pos_loss + neg_loss).mean()


def gaussian_centroid_targets(masks, sigma):
    """Gaussian peaks on the centroid of every connected component of `masks`.

    masks (B,C,H,W) in {0,1} -> targets (B,C,H,W) in [0,1], peak 1.0 at each centroid.
    """
    from scipy.ndimage import center_of_mass, label

    targets = torch.zeros_like(masks, dtype=torch.float32)
    B, C, H, W = masks.shape
    ys, xs = torch.meshgrid(torch.arange(H, device=masks.device),
                            torch.arange(W, device=masks.device), indexing='ij')
    for b in range(B):
        for c in range(C):
            mask = masks[b, c].detach().cpu().numpy()
            labelled, n = label(mask > 0.5)
            if n == 0:
                continue
            centres = center_of_mass(mask, labelled, range(1, n + 1))
            for cy, cx in centres:
                if cy != cy:            # NaN centroid from an empty component
                    continue
                g = torch.exp(-((ys - cy) ** 2 + (xs - cx) ** 2) / (2 * sigma ** 2))
                targets[b, c] = torch.maximum(targets[b, c], g)
    return targets
