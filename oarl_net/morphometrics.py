"""Thirty organelle descriptors and the morphometric logistic classifier."""
import json
from pathlib import Path

import numpy as np
from scipy.special import expit
from skimage.measure import label, regionprops

ORGANELLES = ('nucleus', 'lysosome', 'mitochondria')
PAIRS = ((0, 1), (0, 2), (1, 2))
PER_ORGANELLE = ('area', 'perimeter', 'aspect_ratio', 'eccentricity', 'solidity',
                 'mean_intensity', 'std_intensity', 'num_regions')
FEATURE_NAMES = tuple(
    [f'{org}_{name}' for org in ORGANELLES for name in PER_ORGANELLE]
    + [f'{ORGANELLES[a]}_{ORGANELLES[b]}_{name}'
       for a, b in PAIRS for name in ('jaccard', 'distance')])


def extract_features(probabilities, image, threshold=0.5):
    """Return 30 features from (3,H,W) organelle maps and an (H,W) RI image.

    `image` is the mean of the seven normalized RI slices within one crop.
    Shape descriptors are component means; intensity statistics use all pixels
    in each mask. See docs/classification.md for units and empty-mask handling.
    """
    probabilities, image = np.asarray(probabilities), np.asarray(image)
    if image.ndim != 2 or probabilities.shape != (3, *image.shape):
        raise ValueError('expected probabilities (3,H,W) and image (H,W)')
    masks = probabilities > threshold
    features, centroids = [], []
    for mask in masks:
        components = label(mask, connectivity=2)
        if not mask.any():
            features.extend([0.0] * 8)
            centroids.append(None)
            continue
        props = regionprops(components)
        major = np.mean([p.axis_major_length for p in props])
        minor = np.mean([p.axis_minor_length for p in props])
        values = image[mask]
        features.extend([
            np.mean([p.area for p in props]),
            np.mean([p.perimeter for p in props]),
            major / minor if minor > 0 else 1.0,
            np.mean([p.eccentricity for p in props]),
            np.mean([p.solidity for p in props]),
            values.mean(), values.std(), float(components.max()),
        ])
        centroids.append(np.array(np.nonzero(mask)).mean(axis=1))
    for a, b in PAIRS:
        union = (masks[a] | masks[b]).sum()
        jaccard = (masks[a] & masks[b]).sum() / union if union else 0.0
        distance = (np.linalg.norm(centroids[a] - centroids[b])
                    if centroids[a] is not None and centroids[b] is not None else 0.0)
        features.extend([jaccard, distance])
    return np.asarray(features, dtype=np.float64)


class MorphometricClassifier:
    """StandardScaler and binary logistic-regression parameters stored as JSON."""

    def __init__(self, path):
        params = json.loads(Path(path).read_text())
        if params['feature_names'] != list(FEATURE_NAMES) or params['classes'] != [0, 1]:
            raise ValueError('checkpoint feature order or class labels do not match')
        self.mean = np.asarray(params['scaler_mean'], dtype=np.float64)
        self.scale = np.asarray(params['scaler_scale'], dtype=np.float64)
        self.coef = np.asarray(params['coefficients'], dtype=np.float64)
        self.intercept = float(params['intercept'])
        for values in (self.mean, self.scale, self.coef):
            if values.shape != (len(FEATURE_NAMES),) or not np.isfinite(values).all():
                raise ValueError('checkpoint must contain 30 finite values per vector')
        if (self.scale <= 0).any() or not np.isfinite(self.intercept):
            raise ValueError('invalid checkpoint scale or intercept')

    def standardize(self, features):
        """Apply the training-set mean and scale to an (N,30) matrix."""
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES):
            raise ValueError('expected an (N,30) feature matrix')
        if not np.isfinite(features).all():
            raise ValueError('features must be finite')
        return (features - self.mean) / self.scale

    def predict_probability(self, features):
        """Return P(ETC-perturbed) for each row of an (N,30) feature matrix."""
        return expit(self.standardize(features) @ self.coef + self.intercept)
