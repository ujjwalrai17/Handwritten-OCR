"""
Detection Pipeline Visualizer
==============================
Saves debug images for every stage of the detection pipeline.
Used to demonstrate each stage to your project mentor.

Output structure
----------------
outputs/debug/
    original.png              — input page image
    unet_mask.png             — U-Net probability map (grayscale)
    unet_binary_mask.png      — U-Net thresholded binary mask
    unet_overlay.png          — original + U-Net mask overlay (green)
    craft_char_map.png        — CRAFT character region score map
    craft_affinity_map.png    — CRAFT affinity score map
    craft_boxes.png           — CRAFT candidate region boxes on image
    projection_profile.png    — projection profile + cut lines
    final_lines.png           — final line boxes on original image
    line_000.png              — individual cropped line 0
    line_001.png              — individual cropped line 1
    ...
"""

from pathlib import Path

import cv2
import numpy as np

from ocr.utils.logger import get_logger

log = get_logger(__name__)


def save_original(bgr: np.ndarray, out_dir: str):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(Path(out_dir) / "original.png"), bgr)


def save_unet_outputs(
    prob_mask: np.ndarray,
    binary_mask: np.ndarray,
    bgr: np.ndarray,
    out_dir: str,
):
    """
    Save U-Net probability map, binary mask, and overlay.

    Args:
        prob_mask:   (H, W) float32 in [0, 1] — raw U-Net output.
        binary_mask: (H, W) uint8 {0, 255} — thresholded mask.
        bgr:         Original BGR image for overlay.
        out_dir:     Output directory.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Probability map as grayscale
    cv2.imwrite(str(out / "unet_mask.png"),
                (prob_mask * 255).astype(np.uint8))

    # Binary mask
    cv2.imwrite(str(out / "unet_binary_mask.png"), binary_mask)

    # Overlay: green tint on text regions
    overlay = bgr.copy()
    if binary_mask.shape[:2] != bgr.shape[:2]:
        binary_mask = cv2.resize(binary_mask, (bgr.shape[1], bgr.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)
    mask_bool = binary_mask > 0
    overlay[mask_bool] = (
        overlay[mask_bool].astype(np.float32) * 0.6 +
        np.array([0, 200, 0], dtype=np.float32) * 0.4
    ).astype(np.uint8)
    cv2.imwrite(str(out / "unet_overlay.png"), overlay)
    log.debug("U-Net visualizations saved to %s", out_dir)


def save_craft_outputs(
    char_map: np.ndarray,
    aff_map: np.ndarray,
    craft_boxes: list,
    bgr: np.ndarray,
    out_dir: str,
):
    """
    Save CRAFT character map, affinity map, and detected region boxes.

    Args:
        char_map:    (H, W) float32 character region score.
        aff_map:     (H, W) float32 affinity score.
        craft_boxes: List of (x1, y1, x2, y2) candidate regions.
        bgr:         Original BGR image.
        out_dir:     Output directory.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Character score map — use colormap for visibility
    char_vis = cv2.applyColorMap(
        (char_map * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    cv2.imwrite(str(out / "craft_char_map.png"), char_vis)

    # Affinity score map
    aff_vis = cv2.applyColorMap(
        (aff_map * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    cv2.imwrite(str(out / "craft_affinity_map.png"), aff_vis)

    # CRAFT boxes on original image
    boxes_vis = bgr.copy()
    for (x1, y1, x2, y2) in craft_boxes:
        cv2.rectangle(boxes_vis, (x1, y1), (x2, y2), (255, 100, 0), 2)
    cv2.imwrite(str(out / "craft_boxes.png"), boxes_vis)
    log.debug("CRAFT visualizations saved to %s", out_dir)


def save_projection_visualization(
    binary: np.ndarray,
    boxes: list,
    out_dir: str,
):
    """
    Save projection profile visualization with cut lines.

    Args:
        binary: Binary page image (0=ink, 255=background).
        boxes:  Final line bounding boxes.
        out_dir: Output directory.
    """
    from ocr.detection.projection_detector import save_projection_visualization as _save
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _save(binary, boxes, str(out / "projection_profile.png"))


def save_final_lines(bgr: np.ndarray, boxes: list, out_dir: str):
    """
    Save final line bounding boxes drawn on the original image,
    plus individual cropped line images.

    Args:
        bgr:     Original BGR image.
        boxes:   Final (x1, y1, x2, y2) line boxes.
        out_dir: Output directory.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # All boxes on one image
    vis = bgr.copy()
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, str(i), (x1 + 4, y1 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.imwrite(str(out / "final_lines.png"), vis)

    # Individual line crops
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        crop = bgr[y1:y2, x1:x2]
        if crop.size > 0:
            cv2.imwrite(str(out / f"line_{i:03d}.png"), crop)

    log.debug("Final line visualizations saved to %s (%d lines)", out_dir, len(boxes))
