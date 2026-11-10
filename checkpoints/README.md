# Model weights

Download [oarl-net-weights.zip](https://github.com/SeungJuYoo-1/OARL/releases/download/v1.0.0/oarl-net-weights.zip)
and extract it from the directory containing the cloned `OARL/` folder.
The archive places these files in `OARL/checkpoints/`.

| File | Size | Saved epoch |
|---|---:|---:|
| `segmentation.pth` | 109,043,712 bytes | 244 |
| `classification.pth` | 108,446,624 bytes | 109 |

The files contain a `model_state_dict` and metadata and load with
`torch.load(..., weights_only=True)`. The segmentation checkpoint includes the
auxiliary heatmap head. The classification checkpoint uses the frozen-encoder
configuration.

From this directory, check downloaded files with:

```bash
sha256sum -c SHA256SUMS
```

Weights are distributed separately from the Git repository.
