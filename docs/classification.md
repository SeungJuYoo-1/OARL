# Classification from ODT

Both classifiers distinguish control neurons (label 0) from ETC-perturbed neurons
(label 1). Each tomogram supplies nine fixed 224 × 224 crops in a 3 × 3 grid.
The input to each crop is a stack of seven normalized RI slices; see
[data preprocessing](data.md). Inference does not read fluorescence images.

## Neural network

The neural classifier uses the OARL encoder initialized from organelle
segmentation. It pools the encoder's C5 features and applies the saved
classification head to obtain one probability per crop.

The tomogram label is a majority vote: at least five crop probabilities must
exceed 0.5 for label 1. The mean crop probability is retained as a continuous
tomogram score. The [notebook](../examples/classification_neural_network.ipynb)
shows the input crop positions, individual probabilities and final predictions.

## Logistic regression

The segmentation model predicts nucleus, lysosome and mitochondria probability
maps for each crop. Maps are thresholded at 0.5, and 30 descriptors are extracted
from the resulting masks and the mean projection of the seven normalized RI
slices. Features are computed on individual crops before aggregation.

The logistic model uses `StandardScaler` followed by
`LogisticRegression(C=1, solver="liblinear", class_weight="balanced",
max_iter=2000, random_state=42)` with L2 regularization. It was fitted to 7,326
predicted-mask feature vectors from the 814 classification training tomograms.
The scaler and logistic parameters are stored in
[`checkpoints/logistic_regression.json`](../checkpoints/logistic_regression.json).
The JSON representation supports inference without loading a pickled estimator.

For each crop, the classifier applies the saved training mean and scale, then
computes the sigmoid of the weighted feature sum plus intercept. The mean of the
nine crop probabilities is the tomogram score; a mean above 0.5 gives label 1.
The [notebook](../examples/classification_logistic_regression.ipynb) runs this
workflow and plots the fitted coefficients.

## The 30 descriptors

Eight features are computed for each of the three organelles (24 total):

| Feature suffix | Definition |
|---|---|
| `area` | Mean connected-component area, in pixels squared |
| `perimeter` | Mean connected-component perimeter, in pixels |
| `aspect_ratio` | Mean major-axis length divided by mean minor-axis length; 1 if the latter is zero |
| `eccentricity` | Mean component eccentricity |
| `solidity` | Mean component area divided by its convex-hull area |
| `mean_intensity` | Mean normalized RI across all foreground pixels |
| `std_intensity` | Population standard deviation of normalized RI across all foreground pixels |
| `num_regions` | Number of connected components |

Components use 8-connectivity. An empty organelle mask gives zero for all eight
features. Intensity values use the normalized [0, 1] RI window, not physical RI
units. Component shape properties are computed with `skimage.measure.regionprops`.

Two features are computed for each organelle pair (six total):

| Feature suffix | Definition |
|---|---|
| `jaccard` | Intersection area divided by union area; zero if both masks are empty |
| `distance` | Euclidean distance in pixels between the whole-mask centroids; zero if either mask is empty |

The feature order is nucleus, lysosome, mitochondria, followed by the
nucleus–lysosome, nucleus–mitochondria and lysosome–mitochondria pairs.
[`FEATURE_NAMES`](../oarl_net/morphometrics.py) defines the complete order used by
the JSON model. Extraction uses the thresholded masks without size filtering.
