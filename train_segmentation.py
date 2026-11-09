#!/usr/bin/env python3
"""Train segmentation using ODT volumes and fluorescence-derived masks.

Both data directories are required. Training and validation assignments are
read from the manifest, which covers the 1,622-tomogram corpus.
"""
import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from oarl_net import Config, ODTDataset, create_model, load_tomograms
from oarl_net.seg_trainer import SegTrainer

HERE = Path(__file__).resolve().parent
METADATA = HERE / 'metadata'


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-dir', required=True, help='root of the classification archive')
    p.add_argument('--supplement-dir', required=True,
                   help='root of the segmentation supplement archive')
    p.add_argument('--manifest', default=str(METADATA / 'tomograms.csv'))
    p.add_argument('--output-dir', default='training_output_segmentation')
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--batch-size', type=int, default=None)
    p.add_argument('--learning-rate', type=float, default=None)
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
    cfg.model.classification_mode = False
    if args.epochs:        cfg.seg_training.epochs = args.epochs
    if args.batch_size:    cfg.seg_training.batch_size = args.batch_size
    if args.learning_rate: cfg.seg_training.learning_rate = args.learning_rate
    cfg.seg_training.num_workers = args.num_workers

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    roots = dict(data_dir=args.data_dir, supplement_dir=args.supplement_dir)
    train = load_tomograms(args.manifest, task='segmentation', split='train', **roots)
    val = load_tomograms(args.manifest, task='segmentation', split='val', **roots)
    print(f"train {len(train)} / val {len(val)} tomograms across "
          f"{len({t.plate for t in train + val})} plates, "
          f"{len({t.well for t in train + val})} wells")

    train_ds = ODTDataset(train, cfg.data, augment=True, use_subimages=False)
    train_ds.transform.set_random_seed(args.seed)
    val_ds = ODTDataset(val, cfg.data, augment=False, use_subimages=True)
    train_loader = DataLoader(train_ds, batch_size=cfg.seg_training.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.data.num_subimages ** 2, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = create_model(cfg.model)
    trainer = SegTrainer(model, cfg, device=device, output_dir=args.output_dir)
    trainer.train(train_loader, val_loader)
    print(f"done -> {Path(args.output_dir) / 'best_model.pth'}")


if __name__ == '__main__':
    main()
