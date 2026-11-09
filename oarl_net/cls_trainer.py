"""Classification training with weighted BCE and validation-AUROC selection.

AUROC uses the mean crop probability for each tomogram. Class labels at inference
use a majority vote over crops.
"""
import logging
import time
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@torch.no_grad()
def predict(model, loader, device):
    """-> {tomogram id: [subimage probabilities]}, {tomogram id: label}"""
    model.eval()
    probs_by_id, label_by_id = defaultdict(list), {}
    for images, _, labels, ids in loader:
        probs = torch.sigmoid(model(images.to(device)).squeeze(-1)).cpu().numpy()
        for prob, label, tomogram_id in zip(probs, labels.numpy(), ids):
            probs_by_id[tomogram_id].append(float(prob))
            label_by_id[tomogram_id] = int(label)
    return probs_by_id, label_by_id


def tomogram_auc(probs_by_id, label_by_id):
    """AUROC from mean crop probabilities; NaN for a single-class split."""
    y = np.array([label_by_id[k] for k in probs_by_id])
    p = np.array([float(np.mean(probs_by_id[k])) for k in probs_by_id])
    return float(roc_auc_score(y, p)) if len(set(y)) > 1 else float('nan')


def train_classifier(model, config, train_ds, val_ds, device, epochs=None):
    """Train the classification head. Returns (model, best_val_auc)."""
    tcfg = config.cls_training
    epochs = epochs or tcfg.epochs

    if tcfg.freeze_encoder:
        for name, p in model.named_parameters():
            p.requires_grad = name.startswith('classification_head')
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable parameters: {trainable:,} of {total:,}"
          f"{' (encoder frozen)' if tcfg.freeze_encoder else ''}")

    train_loader = DataLoader(train_ds, batch_size=tcfg.batch_size, shuffle=True,
                              num_workers=tcfg.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config.data.num_subimages ** 2,
                            shuffle=False, num_workers=tcfg.num_workers, pin_memory=True)

    # drop_last on a small corpus can silently yield zero batches; fail loudly instead.
    if len(train_loader) == 0:
        raise ValueError(
            f"batch_size {tcfg.batch_size} exceeds the {len(train_ds)} training crops; "
            f"lower --batch-size")

    counts = Counter(train_ds.labels)
    pos_weight = torch.tensor([counts.get(0, 1) / max(counts.get(1, 1), 1)], device=device)
    print(f"train tomograms: {counts.get(0, 0)} healthy / {counts.get(1, 0)} perturbed "
          f"(pos_weight {pos_weight.item():.3f})")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=tcfg.learning_rate, weight_decay=tcfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_auc, best_state = -1.0, None
    for epoch in range(1, epochs + 1):
        # Frozen parameters still allow BatchNorm running statistics to update.
        model.train()
        t0, epoch_loss, n = time.time(), 0.0, 0
        for images, _, labels, _ in train_loader:
            x = images.to(device, non_blocking=True)
            y = labels.float().to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x).squeeze(-1), y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n += 1
        scheduler.step()

        if epoch % tcfg.eval_every == 0 or epoch == epochs:
            auc = tomogram_auc(*predict(model, val_loader, device))
            if not np.isnan(auc) and auc > best_auc:
                best_auc = auc
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
            print(f"epoch {epoch}/{epochs} ({time.time() - t0:.0f}s) "
                  f"loss={epoch_loss / max(n, 1):.4f} val_tomogram_auc={auc:.3f} "
                  f"best={best_auc:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        # Only happens when validation AUC is undefined every time it is measured,
        # i.e. the validation split is single-class. The last epoch is returned.
        logger.warning("validation AUC was never defined (single-class validation "
                       "split?); returning the final epoch, not a selected one")
        best_auc = float('nan')
    return model, best_auc
