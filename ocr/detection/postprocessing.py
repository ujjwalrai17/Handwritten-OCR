"""
Detection Post-Processing
=========================
Converts raw model outputs (masks, heatmaps) into clean line bounding boxes.

Stage 1 output (U-Net):
    (H, W) probability mask → binary mask → connected components → boxes

Stage 2 output (CRAFT):
    (H/2, W/2) character score map → threshold → connected components → boxes
    These boxes are candidate text regions (may contain multiple lines each)

Stage 3 (Projection):
    Each CRAFT region → projection profiling → individual line boxes

Final output:
    List of (x1, y1, x2, y2) sorted top-to-bottom
"""

import cv2
import numpy as np
from config.settings import cfg
from ocr.utils.logger import get_logger

log = get_logger(__name__)
C = cfg.detection


# ── U-Net mask → boxes ────────────────────────────────────────────────────────

def unet_mask_to_boxes(
    prob_mask: np.ndarray,
    orig_h: int,
    orig_w: int,
) -> list:
    """
    Convert U-Net probability mask to line bounding boxes.

    Steps:
      1. Threshold at cfg.detection.unet_threshold → binary mask
      2. Morphological closing to fill small gaps within a line
      3. Connected components → one component per text line
      4. Filter by minimum height/width
      5. Scale back to original image resolution

    Args:
        prob_mask: (H, W) float32 probability map from U-Net in [0, 1].
                   May be at unet_input_height resolution.
        orig_h:    Original page image height.
        orig_w:    Original page image width.

    Returns:
        List of (x1, y1, x2, y2) in original image coordinates,
        sorted top-to-bottom.
    """
    threshold = C.unet_threshold

    # Threshold
    binary = (prob_mask > threshold).astype(np.uint8) * 255

    # Morphological closing: connect broken strokes within a line
    # Horizontal kernel merges gaps within a line without merging adjacent lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Scale to original resolution
    if binary.shape != (orig_h, orig_w):
        binary = cv2.resize(binary, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    # Connected components
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    boxes = []
    for lbl in range(1, num_labels):
        x = stats[lbl, cv2.CC_STAT_LEFT]
        y = stats[lbl, cv2.CC_STAT_TOP]
        w = stats[lbl, cv2.CC_STAT_WIDTH]
        h = stats[lbl, cv2.CC_STAT_HEIGHT]
        if h >= C.min_line_height and w >= C.min_line_width:
            boxes.append((x, y, x + w, y + h))

    boxes.sort(key=lambda b: b[1])
    log.debug("U-Net post-processing: %d line boxes", len(boxes))
    return boxes


# ── CRAFT heatmap → candidate regions ────────────────────────────────────────

def craft_scores_to_boxes(
    char_score: np.ndarray,
    orig_h: int,
    orig_w: int,
) -> list:
    """
    Convert CRAFT character score map to candidate text region boxes.

    The CRAFT output is at half resolution (H/2, W/2). We scale it back
    to original resolution, threshold, and find connected components.
    Each component is a candidate text region (may span multiple lines).

    Steps:
      1. Scale char_score to original resolution
      2. Threshold at cfg.detection.craft_text_threshold
      3. Dilate to merge nearby characters into word/line blobs
      4. Connected components → candidate region boxes

    Args:
        char_score: (H/2, W/2) float32 character region score in [0, 1].
        orig_h:     Original image height.
        orig_w:     Original image width.

    Returns:
        List of (x1, y1, x2, y2) candidate region boxes,
        sorted top-to-bottom.
    """
    threshold = C.craft_text_threshold

    # Scale to original resolution
    score_full = cv2.resize(char_score, (orig_w, orig_h),
                            interpolation=cv2.INTER_LINEAR)

    # Threshold
    binary = (score_full > threshold).astype(np.uint8) * 255

    # Dilate to merge characters into line blobs
    # Horizontal dilation merges characters in the same line
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 3))
    binary = cv2.dilate(binary, h_kernel, iterations=2)

    # Connected components
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    boxes = []
    for lbl in range(1, num_labels):
        x = stats[lbl, cv2.CC_STAT_LEFT]
        y = stats[lbl, cv2.CC_STAT_TOP]
        w = stats[lbl, cv2.CC_STAT_WIDTH]
        h = stats[lbl, cv2.CC_STAT_HEIGHT]
        if h >= C.min_line_height and w >= C.min_line_width:
            boxes.append((x, y, x + w, y + h))

    boxes.sort(key=lambda b: b[1])
    log.debug("CRAFT post-processing: %d candidate regions", len(boxes))
    return boxes


# ── Merge overlapping boxes ───────────────────────────────────────────────────

def merge_overlapping_boxes(boxes: list, iou_threshold: float = 0.3) -> list:
    """
    Merge bounding boxes that overlap significantly.

    Used after projection profiling to merge boxes that were split
    at ascender/descender crossings but belong to the same line.

    Args:
        boxes:         List of (x1, y1, x2, y2).
        iou_threshold: Boxes with IoU > this are merged.

    Returns:
        Merged list of boxes, sorted top-to-bottom.
    """
    if not boxes:
        return []

    boxes = sorted(boxes, key=lambda b: b[1])
    merged = [list(boxes[0])]

    for box in boxes[1:]:
        x1, y1, x2, y2 = box
        last = merged[-1]

        # Check vertical overlap with last merged box
        overlap_y = min(y2, last[3]) - max(y1, last[1])
        height_min = min(y2 - y1, last[3] - last[1])

        if height_min > 0 and overlap_y / height_min > iou_threshold:
            # Merge: expand last box
            last[0] = min(last[0], x1)
            last[1] = min(last[1], y1)
            last[2] = max(last[2], x2)
            last[3] = max(last[3], y2)
        else:
            merged.append(list(box))

    return [tuple(b) for b in merged]


# ── Tighten box to ink extents ────────────────────────────────────────────────

def tighten_to_ink(binary: np.ndarray, box: tuple) -> tuple:
    """
    Shrink a bounding box to the actual ink pixel extents inside it.

    Projection boxes span the full page width. Passing wide empty boxes
    to TrOCR wastes computation and can distort the image during resize.

    Args:
        binary: Binary page image (0=ink, 255=background).
        box:    (x1, y1, x2, y2) loose bounding box.

    Returns:
        Tightened (x1, y1, x2, y2).
    """
    x1, y1, x2, y2 = [int(v) for v in box]
    band = binary[y1:y2, x1:x2]
    if band.size == 0:
        return box

    ys, xs = np.where(band == 0)
    if len(xs) == 0:
        return box

    return (
        x1 + int(xs.min()),
        y1 + int(ys.min()),
        x1 + int(xs.max()) + 1,
        y1 + int(ys.max()) + 1,
    )
