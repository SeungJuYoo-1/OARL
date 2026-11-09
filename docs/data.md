# Data format

One tomogram represents one field of view. The classification dataset contains
1,018 tomograms, grouped by treatment:

| Directory | Tomograms | Label |
|---|---:|---|
| `Ctrl` | 670 | 0 — control |
| `Antimycin` | 139 | 1 — ETC-perturbed |
| `Oligomycin` | 131 | 1 — ETC-perturbed |
| `Rotenone` | 78 | 1 — ETC-perturbed |

The segmentation supplement contains 604 additional tomograms without
classification labels. Both datasets are used for organelle segmentation,
giving a total of 1,622 tomograms. The uncompressed image files occupy
approximately 37.69 GB and 22.36 GB, respectively.

## Files in each tomogram

```text
3D/000000/0_0.tif ... 0_20.tif   21 refractive-index slices
2DMIP/000000/0_0.tif             Refractive-index maximum-intensity projection
2DFLMIP/000000/0_0.tif           Nucleus fluorescence projection
2DFLMIP/000000/1_0.tif           Lysosome fluorescence projection
2DFLMIP/000000/2_0.tif           Mitochondria fluorescence projection
```

ODT slices and their projection are 741 × 741, uint16 TIFF images. Values are
stored as refractive index multiplied by 10,000: a stored value of 13,370
corresponds to an RI of 1.3370. Fluorescence projections are 1464 × 1464, uint16.

## Loading and preprocessing

The loader clips the RI stack to 1.335–1.400, selects seven consecutive focused
slices, and scales that range to [0, 1]. Focus selection uses image gradients
and Laplacian variance, with the RI projection used to resolve ties. Model
inputs have shape `(7, 224, 224)`.

Classification inference uses nine fixed crops in a 3 × 3 grid. Full-field
segmentation inference averages predictions from overlapping tiles. Both use
only the `3D/` and `2DMIP/` images.

For segmentation training, fluorescence projections are resized to 741 × 741
and converted into binary targets using per-organelle gamma adjustment and Otsu
thresholding. Separate mask files are not needed. The fluorescence channel
assignment above applies to the supplied data; check the assignment before
adding images from other acquisitions.

Preprocessing settings are in [oarl_net/config.py](../oarl_net/config.py), and
the loader is in [oarl_net/data.py](../oarl_net/data.py).

## IDs and split assignments

Tomograms use opaque IDs, `odt_0001` through `odt_1622`.
The [manifest](../metadata/tomograms.csv) records each ID, relative path,
classification label, plate/well/session group, and task-specific split.
Use these assignments when running the supplied models.

See [metadata/README.md](../metadata/README.md) for column definitions and
training, validation and test counts.
