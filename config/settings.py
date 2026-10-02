"""
Central configuration for the Handwritten OCR research project.
All hyperparameters, paths, and model settings live here.
Import this module everywhere instead of hardcoding values.
"""

from dataclasses import dataclass, field
from pathlib import Path

# ── Project Root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent


@dataclass
class ModelConfig:
    name: str = "microsoft/trocr-base-handwritten"
    beam_size: int = 1
    max_new_tokens: int = 64
    batch_size: int = 4
    device: str = "auto"
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    repetition_penalty: float = 1.1
    illegible_token_threshold: float = 0.10
    illegible_line_threshold: float = 0.05
    hallucination_conf_threshold: float = 0.45
    hallucination_length_ratio: float = 2.5


@dataclass
class TrainingConfig:
    learning_rate: float = 5e-5
    num_epochs: int = 10
    train_batch_size: int = 8
    eval_batch_size: int = 8
    warmup_steps: int = 500
    weight_decay: float = 0.01
    early_stopping_patience: int = 3
    save_steps: int = 500
    eval_steps: int = 500
    max_target_length: int = 128
    fp16: bool = False
    hard_sample_oversample_ratio: int = 3
    focal_loss_gamma: float = 2.0
    hard_sample_tag: str = "_hard"


@dataclass
class PreprocessingConfig:
    max_image_side: int = 2048
    min_resize_side: int = 900
    min_image_side: int = 32
    clahe_clip_limit: float = 2.0
    clahe_tile_size: tuple = (8, 8)
    gaussian_kernel: tuple = (3, 3)
    deskew_max_angle: float = 15.0
    deskew_min_angle: float = 0.7
    save_intermediate_images: bool = True
    ocr_debug: bool = True


@dataclass
class DetectionConfig:
    # ── Stage flags ───────────────────────────────────────────────────────────
    # Set use_unet=True after running: python ocr/detection/train_unet.py
    # Set use_craft=True after running: python ocr/detection/train_craft.py
    # use_projection is always True — it is the final refinement stage
    use_unet: bool = True
    use_craft: bool = True
    use_projection: bool = True

    # ── U-Net settings ────────────────────────────────────────────────────────
    unet_checkpoint: str = "checkpoints/unet_lineseg.pth"
    unet_input_height: int = 512
    unet_input_width: int = 512
    unet_threshold: float = 0.5          # probability threshold → binary mask

    # ── CRAFT settings ────────────────────────────────────────────────────────
    craft_checkpoint: str = "checkpoints/craft_textdetector.pth"
    craft_text_threshold: float = 0.4    # character region score threshold
    craft_link_threshold: float = 0.3    # affinity score threshold

    # ── Projection profiling settings ─────────────────────────────────────────
    projection_min_pixel_ratio: float = 0.005
    overlap_valley_smooth: int = 11
    overlap_min_valley_depth: float = 0.35
    overlap_seam_iterations: int = 3

    # ── Shared line settings ──────────────────────────────────────────────────
    line_padding: int = 4
    min_line_height: int = 8
    min_line_width: int = 8
    word_overlap_threshold: float = 0.4
    overlap_variance_threshold: float = 0.18


@dataclass
class PostprocessingConfig:
    spell_correction: bool = True
    confidence_threshold: float = 0.75
    max_edit_distance: int = 2


@dataclass
class AugmentationConfig:
    elastic_alpha: float = 34.0
    elastic_sigma: float = 4.0
    stroke_jitter_sigma: float = 1.5
    line_bleed_prob: float = 0.4
    line_bleed_max_shift: int = 6
    slant_range: tuple = (-15, 15)
    stroke_width_range: tuple = (0.8, 1.4)
    enabled: bool = True


@dataclass
class CRNNConfig:
    cnn_channels: list = None
    rnn_hidden: int = 256
    rnn_layers: int = 2
    rnn_bidirectional: bool = True
    input_height: int = 32
    dropout: float = 0.1
    use_attention: bool = True
    ctc_blank_idx: int = 0

    def __post_init__(self):
        if self.cnn_channels is None:
            self.cnn_channels = [1, 64, 128, 256, 256, 512, 512]


@dataclass
class EnsembleConfig:
    trocr_confidence_threshold: float = 0.70
    blend_mode: str = "confidence"
    use_textract_crosscheck: bool = False
    textract_region: str = "us-east-1"
    crosscheck_cer_flag_threshold: float = 0.30


@dataclass
class SageMakerConfig:
    role_arn: str = "arn:aws:iam::<ACCOUNT_ID>:role/SageMakerExecutionRole"
    region: str = "us-east-1"
    instance_type_train: str = "ml.g4dn.xlarge"
    instance_type_infer: str = "ml.g4dn.xlarge"
    instance_count: int = 1
    volume_size_gb: int = 50
    max_runtime_sec: int = 86400
    s3_bucket: str = "s3://<YOUR-BUCKET>/handwritten-ocr"
    ecr_image_uri: str = ""
    endpoint_name: str = "handwritten-ocr-endpoint"


@dataclass
class PathConfig:
    data_dir: Path = field(default_factory=lambda: ROOT / "data")
    raw_dir: Path = field(default_factory=lambda: ROOT / "data" / "raw")
    processed_dir: Path = field(default_factory=lambda: ROOT / "data" / "processed")
    samples_dir: Path = field(default_factory=lambda: ROOT / "data" / "samples")
    outputs_dir: Path = field(default_factory=lambda: ROOT / "outputs")
    results_dir: Path = field(default_factory=lambda: ROOT / "outputs" / "results")
    pdfs_dir: Path = field(default_factory=lambda: ROOT / "outputs" / "pdfs")
    checkpoints_dir: Path = field(default_factory=lambda: ROOT / "checkpoints")
    logs_dir: Path = field(default_factory=lambda: ROOT / "logs")
    models_dir: Path = field(default_factory=lambda: ROOT / "models")


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    postprocessing: PostprocessingConfig = field(default_factory=PostprocessingConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    crnn: CRNNConfig = field(default_factory=CRNNConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    sagemaker: SageMakerConfig = field(default_factory=SageMakerConfig)
    paths: PathConfig = field(default_factory=PathConfig)


# ── Singleton ─────────────────────────────────────────────────────────────────
cfg = Config()
