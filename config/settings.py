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
    # Hallucination suppression
    # length_penalty < 1.0 discourages the decoder from generating long fluent
    # sequences when visual evidence is weak (penalises LM-driven completions)
    length_penalty: float = 0.5
    # no_repeat_ngram_size breaks repetitive LM-driven loops
    no_repeat_ngram_size: int = 3
    # repetition_penalty > 1.0 further reduces decoder prior dominance
    repetition_penalty: float = 1.3
    # Per-token confidence below this → replaced with [illegible] placeholder
    illegible_token_threshold: float = 0.40
    # Mean line confidence below this → entire line marked [illegible]
    illegible_line_threshold: float = 0.35


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
    # Hard-sample weighting
    # Oversample ratio: hard (overlapping) samples appear this many times per epoch
    hard_sample_oversample_ratio: int = 3
    # Focal-loss gamma: 0 = standard CE, 2 = strong focus on hard examples
    focal_loss_gamma: float = 2.0
    # Tag used in label filenames to mark hard samples: e.g. img_001_hard.txt
    hard_sample_tag: str = "_hard"


@dataclass
class PreprocessingConfig:
    max_image_side: int = 2048
    min_image_side: int = 32
    clahe_clip_limit: float = 2.0
    clahe_tile_size: tuple = (8, 8)
    gaussian_kernel: tuple = (3, 3)
    deskew_max_angle: float = 15.0


@dataclass
class DetectionConfig:
    use_craft: bool = False
    line_padding: int = 4
    min_line_height: int = 8
    min_line_width: int = 8
    projection_min_pixel_ratio: float = 0.005
    word_overlap_threshold: float = 0.4
    overlap_seam_iterations: int = 3
    overlap_valley_smooth: int = 5
    overlap_min_valley_depth: float = 0.15
    # U-Net segmentation
    use_unet: bool = False               # set True once unet_checkpoint exists
    unet_checkpoint: str = "checkpoints/unet_lineseg.pth"
    unet_input_height: int = 512         # resize page height before U-Net
    # Difficulty classification: lines whose ink-density variance exceeds this
    # threshold are tagged "hard" (overlapping strokes raise local variance)
    overlap_variance_threshold: float = 0.18


@dataclass
class PostprocessingConfig:
    spell_correction: bool = True
    confidence_threshold: float = 0.75
    max_edit_distance: int = 2


@dataclass
class AugmentationConfig:
    elastic_alpha: float = 34.0          # elastic distortion magnitude
    elastic_sigma: float = 4.0           # elastic distortion smoothness
    stroke_jitter_sigma: float = 1.5     # per-pixel noise on strokes
    line_bleed_prob: float = 0.4         # probability of simulating ascender/descender bleed
    line_bleed_max_shift: int = 6        # max pixel shift for bleed simulation
    slant_range: tuple = (-15, 15)       # degrees for random slant augmentation
    stroke_width_range: tuple = (0.8, 1.4)  # scale factor for dilation/erosion
    enabled: bool = True


@dataclass
class CRNNConfig:
    """CRNN+CTC model — used as fallback/ensemble for low-confidence TrOCR lines."""
    cnn_channels: list = None            # set in __post_init__
    rnn_hidden: int = 256
    rnn_layers: int = 2
    rnn_bidirectional: bool = True
    input_height: int = 32               # fixed height after resize
    dropout: float = 0.1
    use_attention: bool = True           # additive attention over RNN states
    ctc_blank_idx: int = 0

    def __post_init__(self):
        if self.cnn_channels is None:
            self.cnn_channels = [1, 64, 128, 256, 256, 512, 512]


@dataclass
class EnsembleConfig:
    """Controls TrOCR <-> CRNN routing and Textract cross-check."""
    trocr_confidence_threshold: float = 0.70
    blend_mode: str = "confidence"       # "trocr_only" | "crnn_only" | "confidence"
    # Textract cross-check: flag lines where TrOCR and Textract disagree
    use_textract_crosscheck: bool = False  # requires boto3 + AWS credentials
    textract_region: str = "us-east-1"
    # CER threshold above which a cross-check disagreement is flagged for review
    crosscheck_cer_flag_threshold: float = 0.30


@dataclass
class SageMakerConfig:
    role_arn: str = "arn:aws:iam::<ACCOUNT_ID>:role/SageMakerExecutionRole"
    region: str = "us-east-1"
    instance_type_train: str = "ml.g4dn.xlarge"   # 1× T4 GPU, cost-effective
    instance_type_infer: str = "ml.g4dn.xlarge"
    instance_count: int = 1
    volume_size_gb: int = 50
    max_runtime_sec: int = 86400
    s3_bucket: str = "s3://<YOUR-BUCKET>/handwritten-ocr"
    ecr_image_uri: str = ""                        # filled by build script
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
