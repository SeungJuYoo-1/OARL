"""Segmentation training with an auxiliary centroid loss and hard-negative mining.

Default training loss: 0.7 * (segmentation + 0.5 * heatmap) + 0.3 * mined BCE.
Validation uses segmentation + 0.5 * heatmap for checkpoint selection.
"""
import logging
import random
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from .losses import HeatmapLoss, SegmentationLoss, gaussian_centroid_targets

logger = logging.getLogger(__name__)


class HardNegativeMiner:
    """Flags the highest-loss background pixels so they can be up-weighted."""

    def __init__(self, mining_ratio=0.15, min_hard_negatives=16):
        self.mining_ratio = mining_ratio
        self.min_hard_negatives = min_hard_negatives

    def mine(self, targets, loss_per_pixel):
        hard = torch.zeros_like(targets, dtype=torch.bool)
        B, C = targets.shape[:2]
        for b in range(B):
            for c in range(C):
                negative = targets[b, c] == 0
                if negative.sum() == 0:
                    continue
                losses = loss_per_pixel[b, c][negative]
                order = torch.argsort(losses, descending=True)
                n_hard = max(int(len(losses) * self.mining_ratio),
                             min(self.min_hard_negatives, len(losses)))
                ys, xs = torch.where(negative)
                idx = order[:n_hard]
                hard[b, c, ys[idx], xs[idx]] = True
        return hard

    @staticmethod
    def weights(targets, hard_mask):
        w = torch.ones_like(targets, dtype=torch.float32)
        w[hard_mask] = 3.0
        w[(targets == 0) & (~hard_mask)] = 0.5
        return w


class PreservingAugmentation:
    """On-GPU augmentation that keeps image and mask in register.

    Rotation is capped at a few degrees and resampled nearest for the mask, so thin
    or punctate structures are not smeared into new pixels.
    """

    def __init__(self, rotation_range=10.0, brightness_range=0.08,
                 contrast_range=0.08, noise_std=0.015):
        self.rotation_range = rotation_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.noise_std = noise_std

    def __call__(self, image, mask):
        if random.random() < 0.5:
            image, mask = self._rotate(
                image, mask, random.uniform(-self.rotation_range, self.rotation_range))
        if random.random() < 0.5:
            brightness = 1.0 + random.uniform(-self.brightness_range, self.brightness_range)
            contrast = 1.0 + random.uniform(-self.contrast_range, self.contrast_range)
            image = torch.clamp(image * contrast + (brightness - 1.0), 0, 1)
        if random.random() < 0.3:
            image = torch.clamp(image + torch.randn_like(image) * self.noise_std, 0, 1)
        if random.random() < 0.5:
            image = torch.flip(image, dims=[-1])
            mask = torch.flip(mask, dims=[-1])
        return image, mask

    @staticmethod
    def _rotate(image, mask, angle):
        if abs(angle) >= 5.0:
            return image, mask
        rad = angle * np.pi / 180
        m = torch.tensor([[np.cos(rad), -np.sin(rad), 0],
                          [np.sin(rad), np.cos(rad), 0]],
                         dtype=image.dtype, device=image.device).unsqueeze(0)
        squeeze = image.dim() == 3
        if squeeze:
            image, mask = image.unsqueeze(0), mask.unsqueeze(0)
        grid = F.affine_grid(m, image.size(), align_corners=False)
        image = F.grid_sample(image, grid, mode='bilinear', align_corners=False)
        mask = F.grid_sample(mask, grid, mode='nearest', align_corners=False)
        return (image.squeeze(0), mask.squeeze(0)) if squeeze else (image, mask)


class SegTrainer:
    def __init__(self, model, config, device=None, output_dir='training_output_segmentation'):
        self.config = config
        self.tcfg = config.seg_training
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.miner = HardNegativeMiner(self.tcfg.mining_ratio,
                                       self.tcfg.min_hard_negatives)
        self.augment = PreservingAugmentation(self.tcfg.rotation_range,
                                              self.tcfg.brightness_range,
                                              self.tcfg.contrast_range,
                                              self.tcfg.noise_std)
        self.criterion = SegmentationLoss(model.organelles).to(self.device)
        self.heatmap_criterion = (
            HeatmapLoss(normalize_by_objects=self.tcfg.heatmap_normalize_by_objects)
            if model.enable_heatmap_head else None)
        self.epoch = 0
        self.history = defaultdict(list)
        self._setup_optimizer()

    def _setup_optimizer(self):
        lr = self.tcfg.learning_rate
        backbone = [p for n, p in self.model.named_parameters()
                    if 'backbone' in n and p.requires_grad]
        heatmap = [p for n, p in self.model.named_parameters()
                   if 'heatmap' in n and p.requires_grad]
        rest = [p for n, p in self.model.named_parameters()
                if 'backbone' not in n and 'heatmap' not in n and p.requires_grad]
        self.optimizer = optim.AdamW(
            [{'params': backbone, 'lr': lr * self.tcfg.backbone_lr_multiplier},
             {'params': heatmap, 'lr': lr * self.tcfg.heatmap_lr_multiplier},
             {'params': rest, 'lr': lr}],
            weight_decay=self.tcfg.weight_decay, eps=1e-8)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=min(self.tcfg.scheduler_T_0, max(1, self.tcfg.epochs // 3)),
            T_mult=self.tcfg.scheduler_T_mult,
            eta_min=self.tcfg.scheduler_eta_min)
        use_amp = self.tcfg.use_amp and self.device.type == 'cuda'
        self.scaler = torch.amp.GradScaler('cuda') if use_amp else None

    # -- one step -----------------------------------------------------------

    def _step(self, images, masks):
        self.optimizer.zero_grad(set_to_none=True)
        outputs = self.model(images)
        seg_logits = outputs['segmentation']
        masks = masks.float()

        total_loss, components = self.criterion(seg_logits, masks)

        if self.heatmap_criterion is not None:
            targets = gaussian_centroid_targets(masks, self.tcfg.heatmap_sigma)
            heatmap_loss = self.heatmap_criterion(outputs['heatmaps'], targets)
            total_loss = total_loss + self.tcfg.heatmap_loss_weight * heatmap_loss
            components['heatmap'] = heatmap_loss.item()

        with torch.no_grad():
            per_pixel = F.binary_cross_entropy_with_logits(
                seg_logits, masks, reduction='none')
            hard = self.miner.mine(masks, per_pixel)
            w = self.miner.weights(masks, hard)
            if torch.isnan(w).any() or torch.isinf(w).any():
                w = torch.ones_like(w)

        mining_loss = F.binary_cross_entropy_with_logits(
            seg_logits, masks, weight=w, reduction='mean')
        loss = (1.0 - self.tcfg.mining_loss_weight) * total_loss \
            + self.tcfg.mining_loss_weight * mining_loss

        if not torch.isfinite(loss):
            logger.warning(f"non-finite loss at epoch {self.epoch}, batch skipped")
            return {'total_loss': 0.0, 'nan_skip': 1.0}

        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.tcfg.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.tcfg.max_grad_norm)
            self.optimizer.step()

        metrics = {'total_loss': loss.item(), 'seg_loss': total_loss.item(),
                   'mining_loss': mining_loss.item(),
                   'lr': self.optimizer.param_groups[-1]['lr']}
        metrics.update(components)
        return metrics

    def train_epoch(self, loader):
        self.model.train()
        totals = defaultdict(float)
        n = 0
        pbar = tqdm(loader, desc=f"epoch {self.epoch + 1}")
        for images, masks, _, _ in pbar:
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)
            if self.config.data.augment_training:
                for i in range(images.shape[0]):
                    images[i], masks[i] = self.augment(images[i], masks[i])
            if self.scaler is not None:
                with torch.amp.autocast('cuda'):
                    m = self._step(images, masks)
            else:
                m = self._step(images, masks)
            for k, v in m.items():
                totals[k] += v
            n += 1
            pbar.set_postfix({'loss': f"{m.get('total_loss', 0.0):.4f}"})
        self.scheduler.step()
        return {k: v / max(n, 1) for k, v in totals.items()}

    @torch.no_grad()
    def validate(self, loader):
        self.model.eval()
        totals = defaultdict(float)
        n = 0
        inter = np.zeros(len(self.model.organelles))
        pred_sum = np.zeros(len(self.model.organelles))
        true_sum = np.zeros(len(self.model.organelles))

        for images, masks, _, _ in tqdm(loader, desc="validation"):
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True).float()
            outputs = self.model(images)
            loss, components = self.criterion(outputs['segmentation'], masks)
            if self.heatmap_criterion is not None:
                targets = gaussian_centroid_targets(masks, self.tcfg.heatmap_sigma)
                heatmap_loss = self.heatmap_criterion(outputs['heatmaps'], targets)
                loss = loss + self.tcfg.heatmap_loss_weight * heatmap_loss
                components['heatmap'] = heatmap_loss.item()
            totals['val_loss'] += loss.item()
            for k, v in components.items():
                totals[f'val_{k}'] += v
            n += 1

            binary = (torch.sigmoid(outputs['segmentation']) > 0.5).float()
            for i in range(len(self.model.organelles)):
                inter[i] += (binary[:, i] * masks[:, i]).sum().item()
                pred_sum[i] += binary[:, i].sum().item()
                true_sum[i] += masks[:, i].sum().item()

        metrics = {k: v / max(n, 1) for k, v in totals.items()}
        # Micro-averaged over the whole validation split, not averaged per crop:
        # per-crop Dice is undefined on empty crops and biased by them.
        for i, org in enumerate(self.model.organelles):
            metrics[f'{org}_dice'] = float(
                2 * inter[i] / (pred_sum[i] + true_sum[i] + 1e-8))
            metrics[f'{org}_precision'] = float(inter[i] / pred_sum[i]) if pred_sum[i] else 0.0
            metrics[f'{org}_recall'] = float(inter[i] / true_sum[i]) if true_sum[i] else 1.0
        return metrics

    # -- loop ---------------------------------------------------------------

    def train(self, train_loader, val_loader, epochs=None):
        epochs = epochs or self.tcfg.epochs
        # drop_last on a small corpus can silently yield zero batches; fail loudly instead.
        if len(train_loader) == 0:
            raise ValueError(f"batch_size {self.tcfg.batch_size} exceeds the training set; "
                             f"lower --batch-size")
        best_val = float('inf')
        patience = 0

        for epoch in range(epochs):
            self.epoch = epoch
            t0 = time.time()
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)
            for k, v in {**train_metrics, **val_metrics}.items():
                self.history[k].append(v)

            print(f"epoch {epoch + 1}/{epochs} ({time.time() - t0:.0f}s) "
                  f"train_loss={train_metrics['total_loss']:.4f} "
                  f"val_loss={val_metrics['val_loss']:.4f}")
            for org in self.model.organelles:
                print(f"  {org:13s} dice={val_metrics[f'{org}_dice']:.3f} "
                      f"P={val_metrics[f'{org}_precision']:.3f} "
                      f"R={val_metrics[f'{org}_recall']:.3f}")

            if val_metrics['val_loss'] < best_val:
                best_val = val_metrics['val_loss']
                patience = 0
                self.save(epoch, val_metrics, self.output_dir / 'best_model.pth')
                print(f"  -> new best (val_loss {best_val:.4f})")
            else:
                patience += 1
                if patience >= self.tcfg.patience:
                    print(f"early stop: {patience} epochs without improvement")
                    break

            if (epoch + 1) % self.tcfg.save_every_n_epochs == 0:
                self.save(epoch, val_metrics, self.output_dir / f'checkpoint_epoch_{epoch + 1}.pth')

        return dict(self.history)

    def save(self, epoch, val_metrics, path):
        torch.save({'model_state_dict': self.model.state_dict(),
                    'epoch': epoch,
                    'val_metrics': val_metrics,
                    'config': asdict(self.config),
                    'organelles': list(self.model.organelles)}, path)
