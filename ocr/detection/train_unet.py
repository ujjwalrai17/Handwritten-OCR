"""
U-Net Training Script
=====================
Trains the LineSegUNet on handwritten page images + binary line masks.

Loss function: BCE + Dice Loss
  - BCE (Binary Cross Entropy): penalizes wrong pixel predictions
  - Dice Loss: maximizes overlap between predicted and ground-truth mask
  - Combined: better than either alone for imbalanced segmentation
    (background pixels >> text-line pixels in a typical page)

Usage
-----
    # Prepare data first:
    python ocr/detection/generate_unet_masks.py

    # Then train:
    python ocr/detection/train_unet.py

    # Custom settings:
    python ocr/detection/train_unet.py --epochs 30 --batch 2 --lr 1e-4

Checkpoint saved to: checkpoints/unet_lineseg.pth

STATUS: IMPLEMENTED, NOT YET TRAINED
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ocr.detection.unet import LineSegUNet
from ocr.detection.unet_dataset import build_unet_dataloaders
from ocr.utils.logger import get_logger

log = get_logger("train_unet")


# ── Loss functions ────────────────────────────────────────────────────────────

def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """
    Dice Loss = 1 - (2 * |pred ∩ target|) / (|pred| + |target|)

    Measures overlap between predicted and ground-truth mask.
    Ranges from 0 (perfect) to 1 (no overlap).
    Better than BCE alone for segmentation because it handles
    class imbalance (most pixels are background).

    Args:
        pred:   (B, 1, H, W) predicted probabilities in [0, 1]
        target: (B, 1, H, W) binary ground truth in {0, 1}
        smooth: Laplace smoothing to avoid division by zero
    """
    pred   = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return 1.0 - (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def bce_dice_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Combined BCE + Dice loss.
    BCE handles per-pixel accuracy; Dice handles region overlap.
    Equal weighting (0.5 each) works well for text-line segmentation.
    """
    bce  = F.binary_cross_entropy(pred, target)
    dice = dice_loss(pred, target)
    return 0.5 * bce + 0.5 * dice


# ── Metrics ───────────────────────────────────────────────────────────────────

def iou_score(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Intersection over Union (IoU) — standard segmentation metric.
    IoU = |pred ∩ target| / |pred ∪ target|
    Range: 0 (no overlap) to 1 (perfect).
    """
    pred_bin = (pred > threshold).float()
    intersection = (pred_bin * target).sum().item()
    union = (pred_bin + target).clamp(0, 1).sum().item()
    return intersection / (union + 1e-6)


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device)

    # Build dataloaders
    try:
        train_loader, val_loader = build_unet_dataloaders(
            images_dir=args.images,
            masks_dir=args.masks,
            input_height=args.height,
            input_width=args.width,
            val_split=0.15,
            batch_size=args.batch,
            num_workers=0,
        )
    except FileNotFoundError as e:
        log.error(str(e))
        return

    log.info("Train batches: %d | Val batches: %d", len(train_loader), len(val_loader))

    # Model
    model = LineSegUNet(base_ch=32).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    log.info("LineSegUNet parameters: %d (%.1fM)", total_params, total_params / 1e6)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # LR scheduler: reduce on plateau if val loss stops improving
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    # Checkpoint path
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    best_ckpt = ckpt_dir / "unet_lineseg.pth"

    best_val_loss = float("inf")
    patience_counter = 0
    patience = args.patience

    for epoch in range(1, args.epochs + 1):
        # ── Training ──────────────────────────────────────────────────────────
        model.train()
        train_losses = []
        t0 = time.perf_counter()

        for imgs, masks in train_loader:
            imgs  = imgs.to(device)
            masks = masks.to(device)

            preds = model(imgs)
            loss  = bce_dice_loss(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = float(np.mean(train_losses))

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        val_losses, val_ious = [], []

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs  = imgs.to(device)
                masks = masks.to(device)
                preds = model(imgs)
                val_losses.append(bce_dice_loss(preds, masks).item())
                val_ious.append(iou_score(preds.cpu(), masks.cpu()))

        avg_val_loss = float(np.mean(val_losses))
        avg_val_iou  = float(np.mean(val_ious))
        elapsed = time.perf_counter() - t0

        log.info(
            "Epoch %02d/%02d | train_loss=%.4f | val_loss=%.4f | val_IoU=%.4f | %.1fs",
            epoch, args.epochs, avg_train_loss, avg_val_loss, avg_val_iou, elapsed,
        )

        scheduler.step(avg_val_loss)

        # ── Save best checkpoint ───────────────────────────────────────────────
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_ckpt)
            log.info("  ✓ Best model saved → %s  (val_loss=%.4f)", best_ckpt, best_val_loss)
        else:
            patience_counter += 1
            log.info("  No improvement. Patience %d/%d", patience_counter, patience)
            if patience_counter >= patience:
                log.info("Early stopping at epoch %d", epoch)
                break

    log.info("Training complete. Best val_loss=%.4f | Checkpoint: %s", best_val_loss, best_ckpt)


# ── Inference / visualization ─────────────────────────────────────────────────

def predict_and_visualize(image_path: str, checkpoint: str, output_dir: str = "outputs/debug"):
    """
    Run trained U-Net on a single image and save visualizations.

    Saves:
        outputs/debug/unet_mask.png          — raw probability map
        outputs/debug/unet_binary_mask.png   — thresholded binary mask
        outputs/debug/unet_overlay.png       — original + mask overlay

    Args:
        image_path: Path to input page image.
        checkpoint: Path to trained .pth checkpoint.
        output_dir: Where to save debug images.
    """
    import cv2
    from config.settings import cfg

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = LineSegUNet(base_ch=32).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    log.info("Model loaded from %s", checkpoint)

    # Load and preprocess image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")

    h_orig, w_orig = img.shape
    H = cfg.detection.unet_input_height
    W = cfg.detection.unet_input_width
    img_resized = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)

    tensor = torch.from_numpy(img_resized.astype(np.float32) / 255.0)
    tensor = tensor.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, H, W)

    with torch.no_grad():
        prob_map = model(tensor)[0, 0].cpu().numpy()  # (H, W) in [0,1]

    # Scale back to original resolution
    prob_map_orig = cv2.resize(prob_map, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)

    # Threshold
    threshold = cfg.detection.unet_threshold
    binary_mask = (prob_map_orig > threshold).astype(np.uint8) * 255

    # Save probability map (scaled to 0-255)
    cv2.imwrite(str(out_dir / "unet_mask.png"),
                (prob_map_orig * 255).astype(np.uint8))

    # Save binary mask
    cv2.imwrite(str(out_dir / "unet_binary_mask.png"), binary_mask)

    # Save overlay: original image with mask in green
    original_bgr = cv2.imread(image_path)
    if original_bgr is not None:
        overlay = original_bgr.copy()
        overlay[binary_mask > 0] = (overlay[binary_mask > 0] * 0.6 +
                                    np.array([0, 200, 0]) * 0.4).astype(np.uint8)
        cv2.imwrite(str(out_dir / "unet_overlay.png"), overlay)

    log.info("U-Net visualizations saved to %s", out_dir)
    return prob_map_orig, binary_mask


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train U-Net for text-line segmentation")
    parser.add_argument("--images",   default="data/unet/images",  help="Page images folder")
    parser.add_argument("--masks",    default="data/unet/masks",   help="Binary masks folder")
    parser.add_argument("--epochs",   type=int,   default=30)
    parser.add_argument("--batch",    type=int,   default=2)
    parser.add_argument("--lr",       type=float, default=1e-4)
    parser.add_argument("--height",   type=int,   default=512)
    parser.add_argument("--width",    type=int,   default=512)
    parser.add_argument("--patience", type=int,   default=7)
    parser.add_argument("--predict",  type=str,   default=None,
                        help="Run inference on this image (requires trained checkpoint)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.predict:
        predict_and_visualize(args.predict, "checkpoints/unet_lineseg.pth")
    else:
        train(args)
