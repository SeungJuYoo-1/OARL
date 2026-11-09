# OARL-NET

Segmentation and neural classification code for *Organelle-Aware Representation
Learning Enables Label-Free Detection of Mitochondrial Perturbation in Live Human
Neurons*.

OARL-NET predicts nucleus, lysosome and mitochondria masks from optical diffraction
tomograms. A classifier trained on the segmentation encoder distinguishes control
neurons from neurons treated with ETC inhibitors. Fluorescence provides the
segmentation targets during training; inference uses ODT alone.

## Installation

Use Python 3.12. From the repository directory:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The requirements use PyTorch 2.11 and torchvision 0.26. For a CPU build or a
different CUDA build, install those versions using the
[PyTorch instructions](https://pytorch.org/get-started/locally/) before installing
the remaining requirements. The commands below use CUDA when available;
`--device cpu` runs inference on the CPU.

## Data and weights

The [Zenodo record](https://doi.org/10.5281/zenodo.17115497) provides the dataset
preview. Download links for the full archives matching this manifest and the
model weights are pending.

Place `segmentation.pth` and `classification.pth` in `checkpoints/`. File details
and checksums are in [checkpoints/README.md](checkpoints/README.md). Then check
that both models load and run:

```bash
python smoke_test.py
```

The examples use this directory layout:

```text
OARL/
    oarl_net/
    checkpoints/
    metadata/tomograms.csv
    train_segmentation.py
    train_classification.py
    predict_segmentation.py
    predict_classification.py
classification_dataset/
    Ctrl/odt_.../
    Antimycin/odt_.../
    Oligomycin/odt_.../
    Rotenone/odt_.../
segmentation_supplement/
    odt_.../
```

The classification dataset contains 1,018 tomograms. The segmentation supplement
contains another 604, giving a segmentation corpus of 1,622. Each tomogram has:

```text
3D/000000/0_0.tif ... 0_20.tif   21 RI slices, 741 x 741, uint16
2DMIP/000000/0_0.tif             RI maximum-intensity projection
2DFLMIP/000000/0_0.tif           nucleus fluorescence
2DFLMIP/000000/1_0.tif           lysosome fluorescence
2DFLMIP/000000/2_0.tif           mitochondria fluorescence
```

RI values are stored as refractive index multiplied by 10,000. The loader scales
the 1.335–1.400 range to [0, 1] and selects seven focused slices. Segmentation
training derives masks from the fluorescence images; separate mask files are
not required. Inference reads only `3D/` and `2DMIP/`.

[`metadata/tomograms.csv`](metadata/tomograms.csv) records paths, labels, grouping
and the original splits. Use these assignments rather than splitting the renamed
directories again. Segmentation and classification were split separately, so
these assignments do not enforce separation across the two training stages or
between wells. Column definitions are in [metadata/README.md](metadata/README.md).

## Inference

For an interactive example, open [examples/inference.ipynb](examples/inference.ipynb).
It displays an ODT tomogram, organelle probability maps and the nine crop scores
used for classification, then saves the predictions. After preparing the data
and weights above, run:

```bash
python -m pip install -r requirements-notebooks.txt
python -m jupyter lab examples/inference.ipynb
```

Run on one held-out classification tomogram:

```bash
python predict_classification.py \
    --data-dir ../classification_dataset \
    --split test --limit 1 \
    --output predictions/classification.csv
```

The CSV contains one row per tomogram. `prediction` is 1 (ETC-perturbed) when at
least five of the nine crops have probability above 0.5; otherwise it is 0
(control). `tomogram_probability` is the mean of the nine crop probabilities,
used as a continuous score for ROC analysis. `--threshold` changes the crop
probability threshold used in the vote.

Predict organelle masks:

```bash
python predict_segmentation.py \
    --data-dir ../classification_dataset \
    --supplement-dir ../segmentation_supplement \
    --split test --limit 1 \
    --output predictions/segmentation
```

Each output `.npz` contains `probabilities`, a `(3, 741, 741)` float16 array in
nucleus, lysosome, mitochondria order. Predictions from overlapping 224 x 224
tiles are averaged. Threshold the maps at 0.5 to obtain binary masks.

Remove `--limit 1` to process the selected split. Both scripts also accept
`--split all` and `--checkpoint PATH`. For segmentation, `--batch-size` controls
the number of tiles processed together; reduce it if memory is limited.

## Training

Train the segmentation model:

```bash
python train_segmentation.py \
    --data-dir ../classification_dataset \
    --supplement-dir ../segmentation_supplement
```

The default is up to 500 epochs with batch size 8, AdamW learning rate 2e-4,
and early stopping after 30 epochs without validation-loss improvement.
The backbone and auxiliary heatmap head use learning-rate multipliers of 0.1
and 1.5. The best model is saved to
`training_output_segmentation/best_model.pth`.

Train the classifier using that encoder:

```bash
python train_classification.py \
    --data-dir ../classification_dataset \
    --encoder training_output_segmentation/best_model.pth
```

The classifier trains for 200 epochs with batch size 32 and learning rate 1e-3.
Encoder parameters are frozen by default; BatchNorm running statistics continue
to update during training. `--no-freeze` enables end-to-end fine-tuning. The model
with the best validation tomogram AUROC is saved to
`training_output_classification/best_model.pth`. Pass that path with
`--checkpoint` when running inference on the trained classifier.

Both training scripts accept `--epochs`, `--batch-size`, `--learning-rate`,
`--num-workers` and `--seed`. They load the selected tomograms into RAM before
training. Configuration defaults are in [oarl_net/config.py](oarl_net/config.py).
