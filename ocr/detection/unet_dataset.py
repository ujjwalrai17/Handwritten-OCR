"""
U-Net Dataset Loader
====================
Loads handwritten page images and their corresponding binary line-mask
ground-truth files for training the LineSegUNet.

Expected folder structure
-------------------------
data/unet/
    images/
        page_001.jpg
        page_002.jpg
        ...
    masks/
        page_001.png   ← binary mask: 255=text-line, 0=background
        page_002.png
        ...

How to create masks
-------------------
Option A — Manual annotation:
    Use LabelMe or GIMP to paint white rectangles over each text line.
    Export as PNG. Each line region = white (255), gaps = black (0).

Option B — Synthetic generation (recommended for getting started):
    Run: python ocr/detection/generate_unet_masks.py
    This uses the existing projection profiling on your sample images
    to auto-generate approximate masks. Not perfect but enough to
    bootstrap training.

Option C — IAM dataset:
    The IAM Handwriting Database provides line-level bounding boxes.
    Convert those boxes to binary masks using the provided script.

STATUS: IMPLEMENTED, NOT YET TESTED WITH REAL DATA
"""

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, random_split


class UNetLineDataset(Dataset):
    """
    Dataset for U-Net text-line segmentation training.

    Each sample returns:
        image_tensor: (1, H, W) float32 in [0, 1]  — grayscale page
        mask_tensor:  (1, H, W) float32 in {0, 1}  — binary line mask

    Args:
        images_dir:  Path to folder containing page images (.jpg/.png).
        masks_dir:   Path to folder containing binary mask images (.png).
        input_height: Resize height for U-Net input (width scaled proportionally).
        input_width:  Fixed width. If None, width is scaled from height.
        augment:      Apply random augmentation during training.
    """

    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        input_height: int = 512,
        input_width: int = 512,
        augment: bool = False,
    ):
        self.images_dir   = Path(images_dir)
        self.masks_dir    = Path(masks_dir)
        self.input_height = input_height
        self.input_width  = input_width
        self.augment      = augment

        # Collect matched image/mask pairs
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        self.pairs = []
        for img_path in sorted(self.images_dir.iterdir()):
            if img_path.suffix.lower() not in exts:
                continue
            # Look for mask with same stem (any extension)
            mask_path = None
            for ext in [".png", ".jpg", ".bmp"]:
                candidate = self.masks_dir / (img_path.stem + ext)
                if candidate.exists():
                    mask_path = candidate
                    break
            if mask_path is not None:
                self.pairs.append((img_path, mask_path))

        if not self.pairs:
            raise FileNotFoundError(
                f"No matched image/mask pairs found.\n"
                f"  images_dir: {images_dir}\n"
                f"  masks_dir:  {masks_dir}\n"
                f"Create masks using: python ocr/detection/generate_unet_masks.py"
            )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.pairs[idx]

        # Load image as grayscale
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError(f"Cannot read image: {img_path}")

        # Load mask as grayscale
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise IOError(f"Cannot read mask: {mask_path}")

        # Resize both to fixed input size
        img  = cv2.resize(img,  (self.input_width, self.input_height),
                          interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.input_width, self.input_height),
                          interpolation=cv2.INTER_NEAREST)  # nearest for binary mask

        # Binarize mask: any pixel > 127 = text line
        mask = (mask > 127).astype(np.float32)

        # Optional augmentation
        if self.augment:
            img, mask = self._augment(img, mask)

        # Normalize image to [0, 1]
        img_tensor  = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)

        return img_tensor, mask_tensor

    def _augment(self, img: np.ndarray, mask: np.ndarray):
        """
        Simple augmentation: horizontal flip, brightness jitter.
        Vertical flip is NOT used — text lines have a top-to-bottom order.
        """
        # Random horizontal flip
        if random.random() > 0.5:
            img  = cv2.flip(img,  1)
            mask = cv2.flip(mask, 1)

        # Random brightness jitter ±20%
        if random.random() > 0.5:
            factor = random.uniform(0.8, 1.2)
            img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        # Random Gaussian noise
        if random.random() > 0.5:
            noise = np.random.normal(0, 5, img.shape).astype(np.float32)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        return img, mask


def build_unet_dataloaders(
    images_dir: str,
    masks_dir: str,
    input_height: int = 512,
    input_width: int = 512,
    val_split: float = 0.15,
    batch_size: int = 4,
    num_workers: int = 0,
):
    """
    Build train and validation DataLoaders for U-Net training.

    Args:
        images_dir:   Path to page images.
        masks_dir:    Path to binary masks.
        input_height: Resize height.
        input_width:  Resize width.
        val_split:    Fraction of data for validation (default 15%).
        batch_size:   Batch size.
        num_workers:  DataLoader workers (0 = main process, safe on Windows).

    Returns:
        (train_loader, val_loader)
    """
    from torch.utils.data import DataLoader

    full_ds = UNetLineDataset(
        images_dir, masks_dir,
        input_height=input_height,
        input_width=input_width,
        augment=True,
    )

    n_val   = max(1, int(len(full_ds) * val_split))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    # Disable augmentation for validation
    val_ds.dataset.augment = False

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=False,
    )
    return train_loader, val_loader
