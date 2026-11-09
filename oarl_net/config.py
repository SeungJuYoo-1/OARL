"""Model, preprocessing and training settings for OARL-NET."""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class ModelConfig:
    z_slices: int = 7
    organelles: List[str] = field(
        default_factory=lambda: ['nucleus', 'lysosome', 'mitochondria'])
    classification_mode: bool = False     # True -> forward() returns class logits
    # The auxiliary centroid head is used only for segmentation training.
    enable_heatmap_head: bool = True
    fpn_channels: int = 256
    decoder_channels: int = 64


@dataclass
class DataConfig:
    organelles: List[str] = field(
        default_factory=lambda: ['nucleus', 'lysosome', 'mitochondria'])

    # Tomogram geometry and intensity window (refractive-index counts).
    image_min: int = 13350
    image_max: int = 14000
    image_shape: Tuple[int, int] = (741, 741)
    num_slices: int = 7
    crop_size: int = 224
    num_subimages: int = 3                # -> 3x3 = 9 fixed crops per tomogram at eval
    min_content_threshold: float = 0.08

    # This mapping applies to the supplied corpus; other acquisitions may differ.
    fl_channel_files: Dict[str, str] = field(default_factory=lambda: {
        'nucleus': '0_0.tif', 'lysosome': '1_0.tif', 'mitochondria': '2_0.tif'})

    # Fluorescence -> binary mask (Otsu on a per-organelle gamma-stretched image).
    mask_gamma: Dict[str, float] = field(default_factory=lambda: {
        'nucleus': 0.8, 'lysosome': 0.5, 'mitochondria': 0.6})
    mask_min: Dict[str, int] = field(default_factory=lambda: {
        'nucleus': 10, 'lysosome': 100, 'mitochondria': 50})

    augment_training: bool = True


@dataclass
class SegTrainingConfig:
    epochs: int = 500
    batch_size: int = 8
    learning_rate: float = 2e-4
    backbone_lr_multiplier: float = 0.1
    heatmap_lr_multiplier: float = 1.5
    weight_decay: float = 1e-4
    # Hard-negative mining on background pixels, mixed into the loss at 0.3.
    mining_ratio: float = 0.15
    min_hard_negatives: int = 16
    mining_loss_weight: float = 0.3
    # The heatmap term is included before mixing in hard-negative mining loss.
    heatmap_loss_weight: float = 0.5
    heatmap_sigma: float = 2.0
    # Use pixel averaging, as in the original training configuration.
    heatmap_normalize_by_objects: bool = False
    # Mask-preserving geometric/photometric jitter applied on-GPU per sample.
    rotation_range: float = 10.0
    brightness_range: float = 0.08
    contrast_range: float = 0.08
    noise_std: float = 0.015
    scheduler_T_0: int = 50
    scheduler_T_mult: int = 1
    scheduler_eta_min: float = 1e-7
    patience: int = 30
    use_amp: bool = True
    max_grad_norm: float = 1.0
    num_workers: int = 4
    save_every_n_epochs: int = 25


@dataclass
class ClsTrainingConfig:
    """Recorded protocol of the released classification checkpoint
    (freeze_level='classification_only', 1,348,481 of 27,059,416 parameters trainable)."""
    epochs: int = 200
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    freeze_encoder: bool = True           # train the classification head only
    eval_every: int = 5                   # val AUC is the model-selection metric
    num_workers: int = 4


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    seg_training: SegTrainingConfig = field(default_factory=SegTrainingConfig)
    cls_training: ClsTrainingConfig = field(default_factory=ClsTrainingConfig)

    def __post_init__(self):
        self.data.num_slices = self.model.z_slices
        self.data.organelles = self.model.organelles
