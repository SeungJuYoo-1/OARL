# Tomogram manifest

`tomograms.csv` contains one row per tomogram. Paths are relative to the data
directory supplied to the command.

| Column | Meaning |
|---|---|
| `id` | Tomogram ID, also used in prediction filenames |
| `source` | `classification` resolves against `--data-dir`; `supplement` against `--supplement-dir` |
| `path` | `<Chemical>/<id>` for classification, `<id>` for the supplement |
| `chemical` | Ctrl, Antimycin, Oligomycin or Rotenone; blank for the supplement |
| `label` | 0 for control, 1 for ETC-perturbed; blank for the supplement |
| `plate` | Opaque plate identifier |
| `well` | Opaque physical-well identifier, nested within a plate |
| `session` | Opaque acquisition-session identifier |
| `seg_split` | Segmentation assignment: train, val or test |
| `cls_split` | Classification assignment; blank for the supplement |

| Task | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| Segmentation | 1,297 | 162 | 163 | 1,622 |
| Classification | 814 | 102 | 102 | 1,018 |

These are tomogram-level assignments made before cropping. The two tasks were
split independently; a classification-test tomogram can belong to segmentation
training or validation. Wells can also span multiple partitions.

The fluorescence channel mapping is in `DataConfig.fl_channel_files`:
`0_0.tif` nucleus, `1_0.tif` lysosome and `2_0.tif` mitochondria. Confirm this
mapping before using data from other acquisitions.
