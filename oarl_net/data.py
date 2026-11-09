"""Manifest loading, RI preprocessing and fluorescence-derived segmentation targets.

The manifest resolves `classification` paths against data_dir and `supplement`
paths against supplement_dir. See metadata/README.md for the column definitions.
"""
import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import albumentations as A
import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from skimage.filters import sobel_h, sobel_v, laplace
from torch.utils.data import Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)

MAX_CROP_ATTEMPTS = 15
GAMMA_JITTER_RANGE = (0.9, 1.1)
BRIGHTNESS_JITTER_RANGE = (-0.05, 0.05)
FOCUS_SMOOTHING_SIGMA = 1.0
MORPHOLOGY_KERNEL_SIZE = (3, 3)
MORPHOLOGY_ITERATIONS = 2

TASKS = ('segmentation', 'classification')
SPLITS = ('all', 'train', 'val', 'test')


@dataclass(frozen=True)
class Tomogram:
    id: str
    directory: Path          # resolved; contains 3D/, 2DMIP/, 2DFLMIP/
    label: Optional[int]     # 0 healthy, 1 perturbed, None outside the labelled set
    chemical: str            # treatment; '' for the segmentation supplement
    # Wells are nested within plates; acquisition sessions can span wells.
    plate: str
    well: str
    session: str


def load_tomograms(manifest, data_dir, supplement_dir=None,
                   task='segmentation', split='all'):
    """Resolve the manifest into `Tomogram`s for one task and split.

    Classification selects only the tomograms that carry a label; segmentation selects
    everything, and therefore also needs `supplement_dir`.
    """
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}, got {task!r}")
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")

    split_column = 'seg_split' if task == 'segmentation' else 'cls_split'
    roots = {'classification': Path(data_dir)}
    if supplement_dir:
        roots['supplement'] = Path(supplement_dir)

    selected, need_supplement = [], False
    for r in csv.DictReader(open(manifest)):
        row_split = r[split_column]
        if not row_split:                      # not part of this task's corpus
            continue
        if split != 'all' and row_split != split:
            continue
        if r['source'] not in roots:
            need_supplement = True
            continue
        selected.append(Tomogram(
            id=r['id'],
            directory=roots[r['source']] / r['path'],
            label=int(r['label']) if r['label'] else None,
            chemical=r['chemical'],
            plate=r['plate'], well=r['well'], session=r['session']))

    if need_supplement:
        raise ValueError("the segmentation corpus includes tomograms from the "
                         "supplement archive; pass --supplement-dir")
    if not selected:
        raise ValueError(f"no tomograms selected for task={task} split={split}")
    missing = [t.id for t in selected if not t.directory.is_dir()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} tomogram directories not found under "
                                f"the given roots, e.g. {missing[:3]}")
    return selected


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ODTDataset(Dataset):
    """Crops of a 7-slice refractive-index stack, with per-organelle masks.

    use_subimages=False : one random crop per tomogram (training)
    use_subimages=True  : a fixed num_subimages x num_subimages grid of crops per
                          tomogram, in a deterministic order (validation / inference).
                          __len__ is then n_tomograms * num_subimages**2.

    classification_only=True skips fluorescence entirely and returns zero masks, which
    is what lets the classifier run without a channel map.

    Each item is (image, masks, label, id) with image (Z,H,W) float32 in [0,1] and
    masks (n_organelles,H,W) float32 in {0,1}. `label` is -1 where the manifest gives
    none, which only happens for tomograms outside the labelled set.
    """

    def __init__(self, tomograms, data_config, augment=True, use_subimages=True,
                 classification_only=False, in_memory=True):
        self.config = data_config
        self.tomograms = list(tomograms)
        self.organelles = list(data_config.organelles)
        self.classification_only = classification_only
        self.in_memory = in_memory
        self.augment = augment
        self.use_subimages = use_subimages

        self.crop_size = data_config.crop_size
        self.num_subimages = data_config.num_subimages
        self.image_shape = tuple(data_config.image_shape)
        self.num_slices = data_config.num_slices
        self.image_min = data_config.image_min
        self.image_max = data_config.image_max
        self.min_content_threshold = data_config.min_content_threshold

        self.fl_channel_files = data_config.fl_channel_files

        self.subimage_positions = self._get_subimage_positions(
            self.image_shape, self.crop_size, self.num_subimages)

        self.images, self.masks, self.labels, self.ids = [], [], [], []
        if self.in_memory:
            self._load_all()

        self.transform = self._augmentation_pipeline() if augment else A.Compose([])

    # -- loading ------------------------------------------------------------

    def _load_all(self):
        failed = []
        for tomogram in tqdm(self.tomograms, desc="Loading tomograms"):
            image, masks, label = self._load_one(tomogram)
            if image is None:
                failed.append(tomogram.id)
                continue
            self.images.append(image)
            self.masks.append(masks)
            self.labels.append(label)
            self.ids.append(tomogram.id)
        # Do not silently change the training corpus when a file is missing.
        if failed:
            raise RuntimeError(
                f"{len(failed)} of {len(self.tomograms)} tomograms failed to load, "
                f"e.g. {failed[:3]} -- see the warnings above")

    def load_tomogram(self, tomogram):
        """Load one whole tomogram: (image (H,W,Z) float32 in [0,1], masks, label).

        Used for full-field-of-view inference, where the caller tiles the image itself
        rather than taking the fixed subimage grid.
        """
        image, masks, label = self._load_one(tomogram)
        if image is None:
            raise RuntimeError(f"could not load tomogram: {tomogram.id}")
        return image, masks, label

    def _load_one(self, tomogram):
        try:
            label = -1 if tomogram.label is None else tomogram.label

            slices = self._load_slices(tomogram.directory / "3D" / "000000")
            if slices is None:
                return None, None, None
            z, h, w = slices.shape
            if (h, w) != self.image_shape or z < self.num_slices:
                logger.warning(f"unexpected stack shape {slices.shape} in {tomogram.id}")
                return None, None, None

            mip = cv2.imread(str(tomogram.directory / "2DMIP" / "000000" / "0_0.tif"),
                             cv2.IMREAD_UNCHANGED)
            image = self._select_focal_slices(slices, mip)

            if self.classification_only:
                masks = np.zeros((*image.shape[:2], len(self.organelles)), dtype=np.float32)
            else:
                masks = self._load_masks(tomogram)
                if masks is None:
                    return None, None, None
            return image, masks, label
        except Exception as e:
            logger.warning(f"failed to load {tomogram.id}: {e}")
            return None, None, None

    def _load_slices(self, data_path):
        files = sorted(data_path.glob("*.tif"),
                       key=lambda p: int(p.name.split("_")[1].replace(".tif", "")))
        if not files:
            return None
        slices = []
        for p in files:
            s = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if s is None:
                return None
            slices.append(s)
        return np.clip(np.array(slices), self.image_min, self.image_max)

    def _select_focal_slices(self, slices, mip):
        """Pick the `num_slices`-slice window with the sharpest content, then scale
        the refractive-index window to [0,1]. Returns (H,W,Z)."""
        idx = self._best_focus_window(slices, mip, self.num_slices)
        sel = slices[idx].astype(np.float32)
        sel = np.clip((sel - self.image_min) / (self.image_max - self.image_min), 0, 1)
        return sel.transpose(1, 2, 0)

    def _best_focus_window(self, slices, mip, n):
        def focus_score(s):
            gx, gy = sobel_h(s), sobel_v(s)
            return 0.7 * np.mean(gx * gx + gy * gy) + 0.3 * np.var(laplace(s))

        def robust_norm(x, eps=1e-9):
            med = np.median(x)
            mad = np.median(np.abs(x - med)) + eps
            return (x - med) / (1.4826 * mad)

        scores = robust_norm(gaussian_filter1d(
            np.array([focus_score(s) for s in slices]), sigma=FOCUS_SMOOTHING_SIGMA))
        cumsum = np.concatenate([[0], np.cumsum(scores)])
        win_sums = cumsum[n:] - cumsum[:-n]
        best_starts = np.where(win_sums == win_sums.max())[0]

        # Ties are broken by agreement with the max-intensity projection, high-pass
        # filtered so the comparison is on texture rather than on absolute intensity.
        if mip is not None and len(best_starts) > 1:
            def highpass(img, sigma=1.0):
                return img - gaussian_filter1d(
                    gaussian_filter1d(img, sigma, axis=0), sigma, axis=1)

            def znorm(a):
                return (a - np.mean(a)) / (np.std(a) + 1e-8)

            mip_hp = znorm(highpass(mip.astype(np.float32)))
            cors = np.array([
                float(np.mean(znorm(highpass(slices[z0:z0 + n].astype(np.float32)).mean(axis=0))
                              * mip_hp))
                for z0 in best_starts])
            z_start = int(best_starts[np.argmax(0.85 * win_sums[best_starts] + 0.15 * cors)])
        else:
            z_start = int(best_starts[0])
        return np.arange(z_start, z_start + n)

    # -- masks --------------------------------------------------------------

    def _load_masks(self, tomogram):
        fl_dir = tomogram.directory / "2DFLMIP" / "000000"
        masks = []
        for org in self.organelles:
            channel = self.fl_channel_files[org]
            fl = cv2.imread(str(fl_dir / channel), cv2.IMREAD_UNCHANGED)
            if fl is None:
                logger.warning(f"missing fluorescence file {channel} in {fl_dir}")
                return None
            masks.append(self._otsu_mask(
                cv2.resize(fl, self.image_shape),
                gamma=self.config.mask_gamma.get(org, 1.0),
                floor=self.config.mask_min.get(org, 0),
                fill_holes=(org == "nucleus"),
                blur=(org == "nucleus")))
        return np.stack(masks, axis=-1).astype(np.float32) / 255.0

    def _otsu_mask(self, fl, gamma, floor, fill_holes, blur):
        fl = fl.astype(np.float32)
        if abs(fl.max() - fl.min()) < 1e-6:
            return np.zeros_like(fl, dtype=np.uint8)
        # Gamma < 1 lifts dim puncta above the Otsu split; the per-organelle floor
        # discards the camera pedestal before normalising.
        fl = np.clip((fl - floor) / (fl.max() - floor), 0, 1)
        fl = ((fl ** gamma) * 255).astype(np.uint8)
        if blur:
            fl = cv2.GaussianBlur(fl, (5, 5), 0)
        mask = cv2.threshold(fl, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        return self._fill_holes(mask) if fill_holes else mask

    @staticmethod
    def _fill_holes(mask):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPHOLOGY_KERNEL_SIZE)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel,
                                iterations=MORPHOLOGY_ITERATIONS)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(
            cv2.bitwise_not(mask), connectivity=8)
        filled = np.zeros_like(mask)
        for lab in range(1, n):
            x, y, w, h, _ = stats[lab]
            # A background component that never touches the border is an interior hole.
            if x > 0 and y > 0 and x + w < mask.shape[1] and y + h < mask.shape[0]:
                filled[labels == lab] = 255
        return cv2.bitwise_or(mask, filled)

    # -- cropping and augmentation -----------------------------------------

    @staticmethod
    def _get_subimage_positions(image_shape, crop_size, num_subimages):
        H, W = image_shape
        if num_subimages == 1:
            return [((H - crop_size) // 2, (W - crop_size) // 2)]
        sh = (H - crop_size) / (num_subimages - 1)
        sw = (W - crop_size) / (num_subimages - 1)
        ys = [min(int(i * sh), H - crop_size) for i in range(num_subimages)]
        xs = [min(int(j * sw), W - crop_size) for j in range(num_subimages)]
        return [(y, x) for y in ys for x in xs]

    @staticmethod
    def _augmentation_pipeline():
        return A.Compose([
            A.Rotate(limit=5, p=0.5, border_mode=cv2.BORDER_REFLECT),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ElasticTransform(p=0.3, alpha=10, sigma=3),
            A.OneOf([
                A.RandomGamma(gamma_limit=(1.0, 1.1), p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
            ], p=0.7),
        ])

    def _organelle_rich_crop(self, masks, organelle):
        """Random crop biased towards the sparsest organelle, so lysosome/mito
        training crops are not mostly empty."""
        H, W, _ = masks.shape
        if organelle not in self.organelles:
            return None
        org = masks[:, :, self.organelles.index(organelle)]
        best_score, best_pos = 0, None
        for _ in range(MAX_CROP_ATTEMPTS):
            y = np.random.randint(0, H - self.crop_size + 1)
            x = np.random.randint(0, W - self.crop_size + 1)
            score = org[y:y + self.crop_size, x:x + self.crop_size].mean()
            if score >= self.min_content_threshold:
                return (y, x)
            if score > best_score:
                best_score, best_pos = score, (y, x)
        return best_pos

    def _random_valid_crop(self, image, masks):
        """Random crop containing enough signal; falls back to a plain random crop."""
        H, W = image.shape[:2]
        for _ in range(MAX_CROP_ATTEMPTS):
            y = np.random.randint(0, H - self.crop_size + 1)
            x = np.random.randint(0, W - self.crop_size + 1)
            proj = image[y:y + self.crop_size, x:x + self.crop_size].max(axis=2)
            if proj.mean() < self.min_content_threshold:
                continue
            if self.classification_only or masks is None:
                return (y, x)
            mp = masks[y:y + self.crop_size, x:x + self.crop_size, :].max(axis=2)
            if mp.mean() >= self.min_content_threshold:
                return (y, x)
        return (np.random.randint(0, H - self.crop_size + 1),
                np.random.randint(0, W - self.crop_size + 1))

    @staticmethod
    def _per_slice_jitter(image):
        """Independent gamma/brightness per z-slice: mimics slice-to-slice
        illumination drift without moving structure."""
        out = image.copy()
        for z in range(out.shape[2]):
            g = np.random.uniform(*GAMMA_JITTER_RANGE)
            b = np.random.uniform(*BRIGHTNESS_JITTER_RANGE)
            out[:, :, z] = np.clip(np.clip(out[:, :, z] ** g, 0, 1) + b, 0, 1)
        return out

    # -- Dataset protocol ---------------------------------------------------

    def __len__(self):
        n = len(self.images) if self.in_memory else len(self.tomograms)
        return n * self.num_subimages ** 2 if self.use_subimages else n

    def __getitem__(self, idx):
        sub_idx = 0
        if self.use_subimages:
            sub_idx = idx % self.num_subimages ** 2
            idx = idx // self.num_subimages ** 2

        if self.in_memory:
            image, masks = self.images[idx], self.masks[idx]
            label, tomogram_id = self.labels[idx], self.ids[idx]
        else:
            tomogram = self.tomograms[idx]
            tomogram_id = tomogram.id
            image, masks, label = self._load_one(tomogram)
            if image is None:
                raise RuntimeError(f"failed to load {tomogram_id} at eval time")

        if self.use_subimages:
            y, x = self.subimage_positions[sub_idx]
        elif self.classification_only:
            y, x = self._random_valid_crop(image, None)
        elif np.random.random() < 0.5:
            y, x = self._random_valid_crop(image, masks)
        else:
            y, x = self._organelle_rich_crop(masks, "lysosome") or \
                   self._random_valid_crop(image, masks)

        cs = self.crop_size
        image_crop = image[y:y + cs, x:x + cs]
        masks_crop = (np.zeros((cs, cs, len(self.organelles)), dtype=np.float32)
                      if self.classification_only else masks[y:y + cs, x:x + cs])

        if self.augment:
            if np.random.random() < 0.3:
                image_crop = self._per_slice_jitter(image_crop)
            if self.classification_only:
                image_crop = self.transform(image=image_crop)["image"]
            else:
                t = self.transform(image=image_crop, mask=masks_crop)
                image_crop, masks_crop = t["image"], t["mask"]

        return (torch.from_numpy(np.ascontiguousarray(image_crop)).permute(2, 0, 1),
                torch.from_numpy(np.ascontiguousarray(masks_crop)).permute(2, 0, 1),
                label, tomogram_id)
