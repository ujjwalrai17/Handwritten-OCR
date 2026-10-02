"""
U-Net for Handwritten Text-Line Segmentation
=============================================
Built from scratch using only basic PyTorch layers.

Architecture
------------
Input: (1, H, W) grayscale page image
Output: (1, H, W) probability mask — pixel=1 means text-line region

Encoder path (downsampling):
  DoubleConv → MaxPool → DoubleConv → MaxPool → DoubleConv → MaxPool

Bottleneck:
  DoubleConv

Decoder path (upsampling with skip connections):
  ConvTranspose2d + skip → DoubleConv → ConvTranspose2d + skip → DoubleConv → ...

Final layer:
  Conv2d(base_ch, 1, 1) → Sigmoid → probability map

STATUS: IMPLEMENTED, NOT YET TRAINED
"""

import torch
import torch.nn as nn


# ── Building block ────────────────────────────────────────────────────────────

class DoubleConv(nn.Module):
    """
    Two consecutive Conv2d → BatchNorm → ReLU blocks.
    This is the core repeating unit in U-Net.

    Why two convolutions?
    Each conv expands the receptive field. Two convs give the network
    enough capacity to learn local texture patterns (ink strokes, gaps).

    Args:
        in_ch:  Number of input channels.
        out_ch: Number of output channels.
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ── Encoder block ─────────────────────────────────────────────────────────────

class EncoderBlock(nn.Module):
    """
    One encoder step: DoubleConv followed by MaxPool2d.

    Returns both:
      - skip: feature map BEFORE pooling (used in skip connection)
      - pooled: downsampled feature map passed to next encoder level

    Args:
        in_ch:  Input channels.
        out_ch: Output channels.
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor):
        skip   = self.conv(x)
        pooled = self.pool(skip)
        return skip, pooled


# ── Decoder block ─────────────────────────────────────────────────────────────

class DecoderBlock(nn.Module):
    """
    One decoder step: ConvTranspose2d (upsample) + skip connection + DoubleConv.

    The skip connection concatenates the encoder feature map at the same
    resolution — this is what makes U-Net work. The encoder saw the full
    spatial detail; the decoder uses it to reconstruct precise boundaries.

    Args:
        in_ch:  Input channels (from previous decoder level).
        out_ch: Output channels.
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        # ConvTranspose2d doubles spatial resolution
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        # After concat with skip: channels = out_ch (from up) + out_ch (from skip)
        self.conv = DoubleConv(out_ch * 2, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Handle odd-sized inputs: pad if spatial dims don't match exactly
        if x.shape != skip.shape:
            x = nn.functional.interpolate(
                x, size=skip.shape[2:], mode="bilinear", align_corners=False
            )
        x = torch.cat([skip, x], dim=1)   # concatenate along channel axis
        return self.conv(x)


# ── Full U-Net ────────────────────────────────────────────────────────────────

class LineSegUNet(nn.Module):
    """
    U-Net for pixel-level handwritten text-line segmentation.

    Input:  (B, 1, H, W) — grayscale page image, H should be unet_input_height
    Output: (B, 1, H, W) — soft probability mask in [0, 1]
                           threshold at cfg.detection.unet_threshold to get binary

    Training target:
        Binary mask where text-line pixels = 1, background/gap pixels = 0.
        The model learns to separate overlapping ascenders/descenders by
        using surrounding context (skip connections carry spatial detail).

    Architecture (base_ch=32):
        Encoder:
          enc1: DoubleConv(1  → 32)   + MaxPool  → (B, 32,  H/2,  W/2)
          enc2: DoubleConv(32 → 64)   + MaxPool  → (B, 64,  H/4,  W/4)
          enc3: DoubleConv(64 → 128)  + MaxPool  → (B, 128, H/8,  W/8)
        Bottleneck:
          DoubleConv(128 → 256)                  → (B, 256, H/8,  W/8)
        Decoder:
          dec3: Up(256→128) + skip(128) → DoubleConv(256→128) → (B, 128, H/4, W/4)
          dec2: Up(128→64)  + skip(64)  → DoubleConv(128→64)  → (B, 64,  H/2, W/2)
          dec1: Up(64→32)   + skip(32)  → DoubleConv(64→32)   → (B, 32,  H,   W)
        Output:
          Conv2d(32→1, 1×1) → Sigmoid              → (B, 1,  H,   W)

    Args:
        base_ch: Base channel count. Doubles at each encoder level.
                 32 is a good balance of capacity vs memory for CPU inference.
    """

    def __init__(self, base_ch: int = 32):
        super().__init__()

        # Encoder
        self.enc1 = EncoderBlock(1,          base_ch)       # 1   → 32
        self.enc2 = EncoderBlock(base_ch,    base_ch * 2)   # 32  → 64
        self.enc3 = EncoderBlock(base_ch*2,  base_ch * 4)   # 64  → 128

        # Bottleneck — deepest representation, no pooling
        self.bottleneck = DoubleConv(base_ch * 4, base_ch * 8)  # 128 → 256

        # Decoder
        self.dec3 = DecoderBlock(base_ch * 8, base_ch * 4)  # 256 → 128
        self.dec2 = DecoderBlock(base_ch * 4, base_ch * 2)  # 128 → 64
        self.dec1 = DecoderBlock(base_ch * 2, base_ch)      # 64  → 32

        # Final 1×1 conv: map to single-channel probability mask
        self.out_conv = nn.Conv2d(base_ch, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder — save skip connections
        skip1, x = self.enc1(x)   # skip1: (B, 32,  H,   W)
        skip2, x = self.enc2(x)   # skip2: (B, 64,  H/2, W/2)
        skip3, x = self.enc3(x)   # skip3: (B, 128, H/4, W/4)

        # Bottleneck
        x = self.bottleneck(x)    # (B, 256, H/8, W/8)

        # Decoder — use skip connections
        x = self.dec3(x, skip3)   # (B, 128, H/4, W/4)
        x = self.dec2(x, skip2)   # (B, 64,  H/2, W/2)
        x = self.dec1(x, skip1)   # (B, 32,  H,   W)

        return torch.sigmoid(self.out_conv(x))  # (B, 1, H, W) in [0,1]
