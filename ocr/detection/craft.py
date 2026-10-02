"""
CRAFT — Character Region Awareness For Text Detection
======================================================
Implemented from scratch using basic PyTorch layers.

What CRAFT does
---------------
CRAFT detects text by predicting two heatmaps:
  1. Character region score  — probability that a pixel is inside a character
  2. Affinity score          — probability that two adjacent characters belong
                               to the same word/line

For handwritten line detection we use:
  - Character score → find where ink/text exists
  - Affinity score  → link nearby characters into line regions

Architecture
------------
Input: (B, 3, H, W) RGB image (normalized)

Feature Extraction Backbone (VGG-style, from scratch):
  Block1: Conv(3→64)×2   + MaxPool  → (B, 64,  H/2,  W/2)
  Block2: Conv(64→128)×2 + MaxPool  → (B, 128, H/4,  W/4)
  Block3: Conv(128→256)×3+ MaxPool  → (B, 256, H/8,  W/8)
  Block4: Conv(256→512)×3+ MaxPool  → (B, 512, H/16, W/16)
  Block5: Conv(512→512)×3           → (B, 512, H/16, W/16)  ← no pool

Feature Fusion (U-Net style upsampling):
  Fuse block5 + block4 → upsample → fuse block3 → upsample → fuse block2

Prediction Head:
  Conv layers → two output channels → Sigmoid
  Output: (B, 2, H/2, W/2)
    channel 0 = character region score
    channel 1 = affinity score

IMPORTANT NOTE ON TRAINING
---------------------------
CRAFT requires character-level bounding box annotations to generate
ground-truth heatmaps. Standard handwriting datasets (IAM, RIMES) provide
line-level boxes, NOT character-level boxes.

For this project we use CRAFT in a weakly-supervised mode:
  - Ground truth is generated from line-level boxes using Gaussian heatmaps
  - Each line box → Gaussian blob centered on the line
  - This gives approximate character/region scores without character annotations

See train_craft.py for the full training pipeline.

STATUS: IMPLEMENTED, NOT YET TRAINED
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── VGG-style conv block ──────────────────────────────────────────────────────

def _conv_block(in_ch: int, out_ch: int, num_convs: int = 2) -> nn.Sequential:
    """
    Stack of num_convs × (Conv2d → BatchNorm → ReLU).
    Used to build the VGG-style feature extraction backbone.
    """
    layers = []
    for i in range(num_convs):
        layers += [
            nn.Conv2d(in_ch if i == 0 else out_ch, out_ch,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
    return nn.Sequential(*layers)


# ── Feature fusion block ──────────────────────────────────────────────────────

class FusionBlock(nn.Module):
    """
    Fuse two feature maps of different scales.

    Upsamples the deeper (smaller) feature map to match the shallower
    (larger) feature map, concatenates them, then applies a conv to
    reduce channels.

    Args:
        deep_ch:    Channels in the deeper (upsampled) feature map.
        skip_ch:    Channels in the skip (shallower) feature map.
        out_ch:     Output channels after fusion.
    """
    def __init__(self, deep_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(deep_ch + skip_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, deep: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # Upsample deep feature map to match skip spatial size
        deep_up = F.interpolate(deep, size=skip.shape[2:],
                                mode="bilinear", align_corners=False)
        fused = torch.cat([deep_up, skip], dim=1)
        return self.conv(fused)


# ── CRAFT model ───────────────────────────────────────────────────────────────

class CRAFTDetector(nn.Module):
    """
    CRAFT-style text detector implemented from scratch.

    Input:  (B, 3, H, W) — RGB image, normalized to [0, 1]
    Output: (B, 2, H/2, W/2)
              channel 0 = character region score in [0, 1]
              channel 1 = affinity score in [0, 1]

    The output is at half the input resolution because the backbone
    downsamples by 2× in the first block. For line detection we only
    need the region score (channel 0).

    Architecture decisions
    ----------------------
    - VGG-style backbone: proven feature extractor for text detection
    - No pretrained weights: trained from scratch on handwriting data
    - BatchNorm instead of original CRAFT's no-BN: more stable training
      from scratch without ImageNet pretraining
    - Smaller than original CRAFT (512 max channels vs 1024): appropriate
      for our dataset size and CPU inference requirement
    """

    def __init__(self):
        super().__init__()

        # ── Backbone: VGG-style feature extraction ────────────────────────────
        # Block 1: 3 → 64, stride 2 via MaxPool
        self.block1 = _conv_block(3,   64,  num_convs=2)
        self.pool1  = nn.MaxPool2d(2, stride=2)

        # Block 2: 64 → 128, stride 2 via MaxPool
        self.block2 = _conv_block(64,  128, num_convs=2)
        self.pool2  = nn.MaxPool2d(2, stride=2)

        # Block 3: 128 → 256, stride 2 via MaxPool
        self.block3 = _conv_block(128, 256, num_convs=3)
        self.pool3  = nn.MaxPool2d(2, stride=2)

        # Block 4: 256 → 512, stride 2 via MaxPool
        self.block4 = _conv_block(256, 512, num_convs=3)
        self.pool4  = nn.MaxPool2d(2, stride=2)

        # Block 5: 512 → 512, NO pooling (keeps spatial resolution)
        self.block5 = _conv_block(512, 512, num_convs=3)

        # ── Feature fusion: bottom-up path ────────────────────────────────────
        # Fuse block5 (512) + block4 (512) → 256
        self.fuse54 = FusionBlock(deep_ch=512, skip_ch=512, out_ch=256)
        # Fuse fuse54 (256) + block3 (256) → 128
        self.fuse43 = FusionBlock(deep_ch=256, skip_ch=256, out_ch=128)
        # Fuse fuse43 (128) + block2 (128) → 64
        self.fuse32 = FusionBlock(deep_ch=128, skip_ch=128, out_ch=64)

        # ── Prediction head ───────────────────────────────────────────────────
        # Takes fused features (64 channels) → 2 output maps
        self.pred_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 2, kernel_size=1),   # 2 output channels
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) normalized RGB image

        Returns:
            (B, 2, H/2, W/2)
              [:, 0, :, :] = character region score
              [:, 1, :, :] = affinity score
        """
        # Backbone forward pass — save intermediate feature maps for fusion
        f1 = self.block1(x)          # (B, 64,  H/2,  W/2)
        f2 = self.block2(self.pool1(f1))  # (B, 128, H/4,  W/4)
        f3 = self.block3(self.pool2(f2))  # (B, 256, H/8,  W/8)
        f4 = self.block4(self.pool3(f3))  # (B, 512, H/16, W/16)
        f5 = self.block5(self.pool4(f4))  # (B, 512, H/16, W/16)

        # Feature fusion (bottom-up)
        y = self.fuse54(f5, f4)   # (B, 256, H/8,  W/8)
        y = self.fuse43(y,  f3)   # (B, 128, H/4,  W/4)
        y = self.fuse32(y,  f2)   # (B, 64,  H/2,  W/2)

        # Prediction
        out = self.pred_head(y)   # (B, 2, H/2, W/2)
        return torch.sigmoid(out)
