#!/usr/bin/env python3
"""Classify ODT tomograms using predicted organelle masks and logistic regression.

Thirty descriptors are extracted from each of nine fixed crops. The mean crop
probability determines the tomogram label. Fluorescence is not read.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from oarl_net import Config, ODTDataset, create_model, load_tomograms
from oarl_net.checkpoint import load_weights
from oarl_net.morphometrics import MorphometricClassifier, extract_features

HERE = Path(__file__).resolve().parent


@torch.inference_mode()
def predict_features(model, loader, device):
    """Return {tomogram id: [30-feature vectors]} in crop order."""
    features_by_id = {}
    model.eval()
    for images, _, _, ids in loader:
        maps = torch.sigmoid(model(images.to(device))['segmentation']).cpu().numpy()
        projections = images.numpy().mean(axis=1)
        for tomogram_id, probabilities, projection in zip(ids, maps, projections):
            features_by_id.setdefault(tomogram_id, []).append(
                extract_features(probabilities, projection))
    return features_by_id


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-dir', required=True, help='root of the classification archive')
    p.add_argument('--manifest', default=str(HERE / 'metadata' / 'tomograms.csv'))
    p.add_argument('--checkpoint', default=str(HERE / 'checkpoints' / 'logistic_regression.json'))
    p.add_argument('--segmentation-checkpoint', default=str(HERE / 'checkpoints' / 'segmentation.pth'))
    p.add_argument('--output', default='predictions_logistic_regression.csv')
    p.add_argument('--split', choices=['all', 'train', 'val', 'test'], default='all')
    p.add_argument('--limit', type=int, default=None, help='stop after N tomograms')
    p.add_argument('--batch-size', type=int, default=3, help='crops per forward pass')
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        p.error('--batch-size must be positive and --num-workers nonnegative')
    if args.limit is not None and args.limit < 1:
        p.error('--limit must be positive')

    cfg = Config()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = create_model(cfg.model)
    load_weights(model, args.segmentation_checkpoint)
    model = model.to(device).eval()
    classifier = MorphometricClassifier(args.checkpoint)
    tomograms = load_tomograms(args.manifest, args.data_dir,
                               task='classification', split=args.split)
    if args.limit:
        tomograms = tomograms[:args.limit]
    dataset = ODTDataset(tomograms, cfg.data, augment=False, use_subimages=True,
                         classification_only=True, in_memory=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)
    features = predict_features(model, loader, device)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['tomogram', 'chemical', 'plate', 'well', 'session', 'label',
                         'n_subimages', 'tomogram_probability', 'prediction',
                         'subimage_probabilities'])
        for tomogram in tomograms:
            probabilities = classifier.predict_probability(features[tomogram.id])
            score = float(probabilities.mean())
            writer.writerow([tomogram.id, tomogram.chemical, tomogram.plate,
                             tomogram.well, tomogram.session, tomogram.label,
                             len(probabilities), f'{score:.6f}', int(score > 0.5),
                             ';'.join(f'{x:.6f}' for x in probabilities)])
    print(f'wrote {len(tomograms)} rows to {output}')


if __name__ == '__main__':
    main()
