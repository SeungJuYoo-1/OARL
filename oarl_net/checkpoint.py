"""Read and write tensor checkpoints with weights_only=True."""
from pathlib import Path

import torch


def load_checkpoint(path, map_location='cpu'):
    if not Path(path).is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}. See checkpoints/README.md for filenames "
            "and placement.")
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    if 'model_state_dict' not in ckpt:      # a bare state_dict
        ckpt = {'model_state_dict': ckpt}
    return ckpt


def load_weights(model, path):
    """Load a full checkpoint into `model`. Every tensor must match."""
    ckpt = load_checkpoint(path)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    print(f"loaded {len(ckpt['model_state_dict'])} tensors from {path}"
          + (f" (epoch {ckpt['epoch']})" if 'epoch' in ckpt else ""))
    return ckpt


def load_encoder(model, path):
    """Transfer segmentation weights, leaving the classification head uninitialized."""
    state = load_checkpoint(path)['model_state_dict']
    transfer = {k: v for k, v in state.items()
                if not k.startswith(('classification_head.', 'heatmap_head.'))}
    missing, unexpected = model.load_state_dict(transfer, strict=False)
    uninitialised = [k for k in missing if not k.startswith('classification_head.')]
    if uninitialised or unexpected:
        raise ValueError(f"Incompatible encoder checkpoint: missing {uninitialised[:3]}, "
                         f"unexpected {unexpected[:3]}")
    print(f"warm-started {len(transfer)} tensors from {path}")
    return model


def save_checkpoint(path, model, **metadata):
    """Write weights + metadata. Keep metadata JSON-able so weights_only=True works."""
    torch.save({'model_state_dict': model.state_dict(), **metadata}, path)
