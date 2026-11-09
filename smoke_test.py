#!/usr/bin/env python3
"""Load both checkpoints and check output shapes and finite values on CPU."""
import sys
from pathlib import Path

import torch

from oarl_net import Config, create_model
from oarl_net.checkpoint import load_weights

HERE = Path(__file__).resolve().parent
EXPECTED_TENSORS = {'segmentation': 417, 'classification': 395}


def main():
    torch.manual_seed(0)
    x = torch.randn(2, 7, 224, 224)
    ok = True

    for task, path in [('segmentation', 'checkpoints/segmentation.pth'),
                       ('classification', 'checkpoints/classification.pth')]:
        print(f"--- {task} ---")
        cfg = Config()
        cfg.model.classification_mode = (task == 'classification')
        cfg.model.enable_heatmap_head = (task == 'segmentation')
        model = create_model(cfg.model).eval()

        meta = load_weights(model, HERE / path)
        n = len(meta['model_state_dict'])
        if n != EXPECTED_TENSORS[task]:
            print(f"  UNEXPECTED tensor count: {n} (want {EXPECTED_TENSORS[task]})")
            ok = False

        with torch.no_grad():
            out = model(x)
        expected = ({'segmentation': (2, 3, 224, 224), 'heatmaps': (2, 3, 224, 224)}
                    if task == 'segmentation' else {'classification': (2, 1)})
        outputs = out if isinstance(out, dict) else {'classification': out}
        if outputs.keys() != expected.keys():
            print(f"  UNEXPECTED output keys: {list(outputs)}")
            ok = False
        for key, value in outputs.items():
            if tuple(value.shape) != expected.get(key) or not torch.isfinite(value).all():
                print(f"  INVALID output: {key} {tuple(value.shape)}")
                ok = False
        if isinstance(out, dict):
            print(f"  outputs: {', '.join(f'{k} {tuple(v.shape)}' for k, v in out.items())}")
            out = out['segmentation']
        params = sum(p.numel() for p in model.parameters())
        print(f"  epoch {meta['epoch']}, {params:,} parameters")
        print(f"  output {tuple(out.shape)}  mean {out.mean():+.6f}  std {out.std():.6f}")

    print("\nOK" if ok else "\nFAILED")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
