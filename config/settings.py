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
    max_new_tokens: int = 128
    batch_size: int = 8                  # lines processed per GPU batch
    device: str = "auto"                 # "auto" | "cpu" | "cuda"


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
    fp16: bool = False                   # set True if GPU supports it


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
    use_craft: bool = False              # craft-text-detector incompatible with Python 3.14+
    line_padding: int = 4
    min_line_height: int = 8
    min_line_width: int = 8
    projection_min_pixel_ratio: float = 0.005
    word_overlap_threshold: float = 0.4


@dataclass
class PostprocessingConfig:
    spell_correction: bool = True
    confidence_threshold: float = 0.75
    max_edit_distance: int = 2


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
    paths: PathConfig = field(default_factory=PathConfig)


# ── Singleton ─────────────────────────────────────────────────────────────────
cfg = Config()
