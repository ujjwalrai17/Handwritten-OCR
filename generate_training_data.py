"""
Generate synthetic training data for U-Net and CRAFT from sample images.

Uses projection profiling (already working) to auto-detect line bounding boxes,
then generates:
  - data/unet/images/  + data/unet/masks/   for U-Net training
  - data/craft/images/ + data/craft/labels/ for CRAFT training

Each source image is augmented into multiple variants so training has
enough samples despite having only a few source images.

Usage:
    python generate_training_data.py
    python generate_training_data.py --samples data/samples --augments 20
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

# ── Augmentation helpers ──────────────────────────────────────────────────────

def _augment(img_bgr: np.ndarray, bboxes: list, idx: int):
    """Apply deterministic-seeded augmentation to image + bboxes."""
    rng = random.Random(idx * 137 + 7)
    img = img_bgr.copy()
    h, w = img.shape[:2]

    # 1. Brightness / contrast jitter
    alpha = rng.uniform(0.75, 1.25)   # contrast
    beta  = rng.randint(-30, 30)       # brightness
    img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    # 2. Gaussian blur (simulate out-of-focus)
    if rng.random() > 0.5:
        k = rng.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)

    # 3. Salt-and-pepper noise
    if rng.random() > 0.5:
        noise_mask = np.random.RandomState(idx).randint(0, 100, img.shape[:2])
        img[noise_mask < 2]  = 0
        img[noise_mask > 97] = 255

    # 4. Small rotation (±3°) — keep bboxes valid
    angle = rng.uniform(-3, 3)
    cx, cy = w / 2, h / 2
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), borderValue=(255, 255, 255))

    new_boxes = []
    for (x1, y1, x2, y2) in bboxes:
        corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        ones = np.ones((4, 1), dtype=np.float32)
        corners_h = np.hstack([corners, ones])
        rotated = (M @ corners_h.T).T
        nx1 = int(np.clip(rotated[:, 0].min(), 0, w - 1))
        ny1 = int(np.clip(rotated[:, 1].min(), 0, h - 1))
        nx2 = int(np.clip(rotated[:, 0].max(), 0, w - 1))
        ny2 = int(np.clip(rotated[:, 1].max(), 0, h - 1))
        if nx2 > nx1 and ny2 > ny1:
            new_boxes.append((nx1, ny1, nx2, ny2))

    # 5. Horizontal flip (50%)
    if rng.random() > 0.5:
        img = cv2.flip(img, 1)
        flipped = []
        for (x1, y1, x2, y2) in new_boxes:
            flipped.append((w - x2, y1, w - x1, y2))
        new_boxes = flipped

    return img, new_boxes


# ── Projection-based line detection ──────────────────────────────────────────

def _detect_lines_projection(gray: np.ndarray) -> list:
    """
    Detect text-line bounding boxes using horizontal projection profiling.
    Returns list of (x1, y1, x2, y2).
    """
    # Binarize
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Dilate horizontally to merge words into lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 3))
    dilated = cv2.dilate(binary, kernel, iterations=2)

    # Horizontal projection
    proj = np.sum(dilated > 0, axis=1).astype(float)

    # Smooth
    window = 11
    kernel_1d = np.ones(window) / window
    smoothed = np.convolve(proj, kernel_1d, mode="same")

    threshold = smoothed.max() * 0.08
    in_line = False
    start = 0
    runs = []
    for y, val in enumerate(smoothed):
        if val > threshold and not in_line:
            in_line = True
            start = y
        elif val <= threshold and in_line:
            in_line = False
            runs.append((start, y))
    if in_line:
        runs.append((start, len(smoothed) - 1))

    h, w = gray.shape
    boxes = []
    for (y1, y2) in runs:
        if y2 - y1 < 8:
            continue
        # Find horizontal extent of ink in this row band
        row_band = binary[y1:y2, :]
        cols = np.where(row_band.sum(axis=0) > 0)[0]
        if len(cols) == 0:
            continue
        x1 = max(0, int(cols.min()) - 4)
        x2 = min(w, int(cols.max()) + 4)
        boxes.append((x1, max(0, y1 - 4), x2, min(h, y2 + 4)))

    return boxes


# ── Mask generation ───────────────────────────────────────────────────────────

def _make_unet_mask(h: int, w: int, bboxes: list) -> np.ndarray:
    """Binary mask: white (255) inside text-line bboxes, black elsewhere."""
    mask = np.zeros((h, w), dtype=np.uint8)
    for (x1, y1, x2, y2) in bboxes:
        mask[y1:y2, x1:x2] = 255
    return mask


def _make_craft_label(bboxes: list) -> str:
    """CRAFT label file: one line per bbox, format x1,y1,x2,y2."""
    return "\n".join(f"{x1},{y1},{x2},{y2}" for (x1, y1, x2, y2) in bboxes)


# ── Main ──────────────────────────────────────────────────────────────────────

def generate(samples_dir: str, n_augments: int):
    samples_path = Path(samples_dir)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    source_images = [p for p in samples_path.iterdir()
                     if p.suffix.lower() in exts]

    if not source_images:
        print(f"No images found in {samples_dir}")
        return

    # Create output dirs
    unet_img_dir  = Path("data/unet/images");  unet_img_dir.mkdir(parents=True, exist_ok=True)
    unet_msk_dir  = Path("data/unet/masks");   unet_msk_dir.mkdir(parents=True, exist_ok=True)
    craft_img_dir = Path("data/craft/images"); craft_img_dir.mkdir(parents=True, exist_ok=True)
    craft_lbl_dir = Path("data/craft/labels"); craft_lbl_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for src in source_images:
        bgr  = cv2.imread(str(src))
        if bgr is None:
            print(f"  Cannot read {src.name}, skipping")
            continue
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        bboxes = _detect_lines_projection(gray)

        if not bboxes:
            print(f"  No lines detected in {src.name}, skipping")
            continue

        print(f"  {src.name}: {len(bboxes)} lines detected")

        for i in range(n_augments):
            aug_img, aug_boxes = _augment(bgr, bboxes, idx=total)
            if not aug_boxes:
                aug_boxes = bboxes  # fallback if rotation ate all boxes

            stem = f"{src.stem}_aug{i:03d}"
            h, w = aug_img.shape[:2]

            # Save U-Net pair
            cv2.imwrite(str(unet_img_dir / f"{stem}.jpg"), aug_img)
            mask = _make_unet_mask(h, w, aug_boxes)
            cv2.imwrite(str(unet_msk_dir / f"{stem}.png"), mask)

            # Save CRAFT pair
            cv2.imwrite(str(craft_img_dir / f"{stem}.jpg"), aug_img)
            (craft_lbl_dir / f"{stem}.txt").write_text(_make_craft_label(aug_boxes))

            total += 1

    print(f"\nDone. Generated {total} samples.")
    print(f"  U-Net  → data/unet/images/  + data/unet/masks/")
    print(f"  CRAFT  → data/craft/images/ + data/craft/labels/")
    print(f"\nNext steps:")
    print(f"  python -m ocr.detection.train_unet")
    print(f"  python -m ocr.detection.train_craft")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--samples",   default="data/samples")
    p.add_argument("--augments",  type=int, default=20,
                   help="Augmented copies per source image (default 20)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate(args.samples, args.augments)
