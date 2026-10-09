"""
Configuration for Custom TrOCR — built entirely from scratch.
"""
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent


@dataclass
class ModelConfig:
    image_height: int = 128
    image_width:  int = 1024
    patch_size:   int = 16
    embed_dim:    int = 256
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    num_heads:    int = 8
    ff_dim:       int = 1024
    dropout:      float = 0.1
    max_text_length: int = 128

    @property
    def num_patches(self):
        return (self.image_height // self.patch_size) * (self.image_width // self.patch_size)


@dataclass
class TrainingConfig:
    batch_size:    int   = 8
    learning_rate: float = 1e-4
    weight_decay:  float = 1e-4
    epochs:        int   = 20
    warmup_steps:  int   = 1000
    grad_clip:     float = 1.0
    val_split:     float = 0.1
    seed:          int   = 42
    num_workers:   int   = 0
    log_interval:  int   = 50


@dataclass
class PathConfig:
    checkpoint_dir: Path = field(default_factory=lambda: ROOT / "checkpoints")
    data_processed: Path = field(default_factory=lambda: ROOT / "data" / "processed")


@dataclass
class Config:
    model:    ModelConfig    = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    paths:    PathConfig     = field(default_factory=PathConfig)


cfg = Config()
