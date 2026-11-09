#!/usr/bin/env python3
"""Train a control-vs-ETC classifier from a segmentation checkpoint.

Uses the 1,018 labelled tomograms, with splits read from the manifest.
Only the classification head is trained by default. Fluorescence is not read.
"""
import argparse
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from oarl_net import Config, ODTDataset, create_model, load_tomograms
from oarl_net.checkpoint import load_encoder, save_checkpoint
from oarl_net.cls_trainer import train_classifier

HERE = Path(__file__).resolve().parent
METADATA = HERE / 'metadata'


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-dir', required=True, help='root of the classification archive')
    p.add_argument('--encoder', required=True,
                   help='segmentation checkpoint to warm-start from')
    p.add_argument('--manifest', default=str(METADATA / 'tomograms.csv'))
    p.add_argument('--output', default='training_output_classification/best_model.pth')
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--batch-size', type=int, default=None)
    p.add_argument('--learning-rate', type=float, default=None)
    p.add_argument('--no-freeze', action='store_true', help='fine-tune the whole network')
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--device', default='cuda')
    p.add_argument('--seed', type=int, default=42,
                   help='random seed; split assignments are read from the manifest')
    args = p.parse_args()
    for name in ('epochs', 'batch_size', 'learning_rate'):
        value = getattr(args, name)
        if value is not None and value <= 0:
            p.error(f"--{name.replace('_', '-')} must be positive")
    if args.num_workers < 0:
        p.error('--num-workers must be non-negative')

    cfg = Config()
    cfg.model.classification_mode = True
    cfg.model.enable_heatmap_head = False
    cfg.cls_training.freeze_encoder = not args.no_freeze
    if args.epochs:        cfg.cls_training.epochs = args.epochs
    if args.batch_size:    cfg.cls_training.batch_size = args.batch_size
    if args.learning_rate: cfg.cls_training.learning_rate = args.learning_rate
    cfg.cls_training.num_workers = args.num_workers

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    train = load_tomograms(args.manifest, args.data_dir, task='classification', split='train')
    val = load_tomograms(args.manifest, args.data_dir, task='classification', split='val')
    test = load_tomograms(args.manifest, args.data_dir, task='classification', split='test')
    print(f"train {len(train)} / val {len(val)} / test {len(test)} tomograms")

    common = dict(classification_only=True)
    train_ds = ODTDataset(train, cfg.data, augment=True, use_subimages=False, **common)
    train_ds.transform.set_random_seed(args.seed)
    val_ds = ODTDataset(val, cfg.data, augment=False, use_subimages=True, **common)

    model = create_model(cfg.model)
    load_encoder(model, args.encoder)
    model = model.to(device)

    model, best_auc = train_classifier(model, cfg, train_ds, val_ds, device,
                                       epochs=args.epochs)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(out, model, val_auc=best_auc, encoder=str(args.encoder),
                    freeze_encoder=cfg.cls_training.freeze_encoder,
                    config=asdict(cfg), seed=args.seed,
                    classification_mode=True)
    print(f"done -> {out}  (best validation tomogram AUC {best_auc:.3f})"
          if best_auc == best_auc else f"done -> {out}  (no validation AUC available)")
    print(f"{len(test)} test tomograms; use predict_classification.py "
          f"--split test --checkpoint {out} with the same data directory and manifest")


if __name__ == '__main__':
    main()
