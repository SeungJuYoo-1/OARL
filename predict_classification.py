#!/usr/bin/env python3
"""Classify tomograms by majority vote over nine 224 x 224 crops.

The output CSV includes the mean crop probability for ROC analysis, the voted
class label, and the individual crop probabilities. Fluorescence is not read.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from oarl_net import Config, ODTDataset, create_model, load_tomograms
from oarl_net.checkpoint import load_weights
from oarl_net.cls_trainer import predict

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-dir', required=True, help='root of the classification archive')
    p.add_argument('--manifest', default=str(HERE / 'metadata' / 'tomograms.csv'))
    p.add_argument('--checkpoint', default=str(HERE / 'checkpoints' / 'classification.pth'))
    p.add_argument('--output', default='predictions_classification.csv')
    p.add_argument('--split', choices=['all', 'train', 'val', 'test'], default='all',
                   help="'test' selects the held-out tomograms recorded in the manifest")
    p.add_argument('--limit', type=int, default=None, help='stop after N tomograms')
    p.add_argument('--threshold', type=float, default=0.5,
                   help='crop probability threshold used in the majority vote')
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    if not 0 <= args.threshold <= 1:
        p.error('--threshold must be between 0 and 1')
    if args.limit is not None and args.limit < 1:
        p.error('--limit must be positive')

    cfg = Config()
    cfg.model.classification_mode = True
    cfg.model.enable_heatmap_head = False

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = create_model(cfg.model)
    load_weights(model, args.checkpoint)
    model = model.to(device).eval()

    tomograms = load_tomograms(args.manifest, args.data_dir,
                               task='classification', split=args.split)
    if args.limit:
        tomograms = tomograms[:args.limit]
    meta = {t.id: t for t in tomograms}
    print(f"{len(tomograms)} tomograms ({args.split} split) x "
          f"{cfg.data.num_subimages ** 2} crops -> {args.output}")

    ds = ODTDataset(tomograms, cfg.data, augment=False, use_subimages=True,
                    classification_only=True, in_memory=False)
    loader = DataLoader(ds, batch_size=cfg.data.num_subimages ** 2, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    probs_by_id, label_by_id = predict(model, loader, device)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['tomogram', 'chemical', 'plate', 'well', 'session', 'label',
                    'n_subimages', 'tomogram_probability', 'prediction',
                    'subimage_probabilities'])
        for tomogram_id in sorted(probs_by_id):
            t = meta[tomogram_id]
            sub = probs_by_id[tomogram_id]
            tomogram_prob = float(np.mean(sub))
            prediction = int(np.count_nonzero(np.asarray(sub) > args.threshold)
                             > len(sub) / 2)
            w.writerow([tomogram_id, t.chemical, t.plate, t.well, t.session,
                        label_by_id[tomogram_id], len(sub), f"{tomogram_prob:.6f}",
                        prediction,
                        ';'.join(f"{s:.6f}" for s in sub)])
    print(f"wrote {len(probs_by_id)} rows to {args.output}")


if __name__ == '__main__':
    main()
