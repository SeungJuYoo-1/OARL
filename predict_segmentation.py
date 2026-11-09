#!/usr/bin/env python3
"""Predict nucleus, lysosome and mitochondria probability maps from ODT volumes.

Writes one .npz per tomogram with probabilities shaped (3, 741, 741).
Overlapping 224 x 224 tile predictions are averaged. Fluorescence is not read.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from oarl_net import Config, ODTDataset, create_model, load_tomograms
from oarl_net.checkpoint import load_weights

HERE = Path(__file__).resolve().parent


def tile_positions(extent, crop):
    """Tile starts covering [0, extent) with stride `crop`, last one flush to the edge."""
    if extent <= crop:
        return [0]
    starts = list(range(0, extent - crop + 1, crop))
    if starts[-1] != extent - crop:
        starts.append(extent - crop)
    return starts


@torch.no_grad()
def predict_tomogram(model, image, crop, device, batch_size=9):
    """image (H,W,Z) -> probabilities (C,H,W), averaged over overlapping tiles."""
    H, W, _ = image.shape
    ys, xs = tile_positions(H, crop), tile_positions(W, crop)
    tiles = [(y, x) for y in ys for x in xs]

    total, count, out_channels = None, np.zeros((H, W), dtype=np.float32), None
    for i in range(0, len(tiles), batch_size):
        chunk = tiles[i:i + batch_size]
        batch = torch.from_numpy(np.stack(
            [image[y:y + crop, x:x + crop].transpose(2, 0, 1) for y, x in chunk]))
        probs = torch.sigmoid(model(batch.to(device))['segmentation']).cpu().numpy()
        if total is None:
            out_channels = probs.shape[1]
            total = np.zeros((out_channels, H, W), dtype=np.float32)
        for (y, x), p in zip(chunk, probs):
            total[:, y:y + crop, x:x + crop] += p
            count[y:y + crop, x:x + crop] += 1
    return total / np.maximum(count, 1)[None]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-dir', required=True, help='root of the classification archive')
    p.add_argument('--supplement-dir', default=None,
                   help='root of the segmentation supplement archive; required unless '
                        'the selected split contains no supplement tomograms')
    p.add_argument('--manifest', default=str(HERE / 'metadata' / 'tomograms.csv'))
    p.add_argument('--checkpoint', default=str(HERE / 'checkpoints' / 'segmentation.pth'))
    p.add_argument('--output', default='predictions_segmentation')
    p.add_argument('--split', choices=['all', 'train', 'val', 'test'], default='all',
                   help="'test' selects the held-out tomograms recorded in the manifest")
    p.add_argument('--limit', type=int, default=None, help='stop after N tomograms')
    p.add_argument('--batch-size', type=int, default=9,
                   help='number of tiles per forward pass')
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    if args.batch_size < 1:
        p.error('--batch-size must be positive')
    if args.limit is not None and args.limit < 1:
        p.error('--limit must be positive')

    cfg = Config()
    cfg.model.classification_mode = False

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = create_model(cfg.model)
    load_weights(model, args.checkpoint)
    model = model.to(device).eval()

    tomograms = load_tomograms(args.manifest, args.data_dir, args.supplement_dir,
                               task='segmentation', split=args.split)
    if args.limit:
        tomograms = tomograms[:args.limit]
    print(f"{len(tomograms)} tomograms ({args.split} split) -> {args.output}")

    # Skip fluorescence and load one tomogram at a time.
    loader = ODTDataset(tomograms, cfg.data, augment=False, use_subimages=False,
                        classification_only=True, in_memory=False)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    for tomogram in tqdm(tomograms, desc="predicting"):
        image, _, _ = loader.load_tomogram(tomogram)
        probs = predict_tomogram(model, image, cfg.data.crop_size, device,
                                batch_size=args.batch_size)
        np.savez_compressed(out_dir / f"{tomogram.id}.npz",
                            probabilities=probs.astype(np.float16),
                            organelles=np.array(cfg.data.organelles),
                            well=tomogram.well)
    print(f"wrote {len(tomograms)} files to {out_dir}")


if __name__ == '__main__':
    main()
