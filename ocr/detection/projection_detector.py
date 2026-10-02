"""
Projection Profiling — Horizontal Ink Density Line Detector
============================================================
Classical computer vision technique for finding text lines.

How it works
------------
1. For each row y in the binary image, count the number of ink pixels:
       projection[y] = number of pixels where binary[y, :] == 0 (ink)

2. This gives a 1D profile. Text lines appear as peaks (many ink pixels).
   Gaps between lines appear as valleys (few or zero ink pixels).

3. Find the valleys → these are the cut points between lines.

4. Refine each cut using seam analysis: look in a small band around
   the valley and find the row with the absolute minimum ink crossing.

5. Return bounding boxes for each line band.

This module is used as Stage 3 in the full pipeline:
    U-Net mask → CRAFT regions → Projection refinement → Final boxes

It can also run standalone as a fallback when U-Net/CRAFT are not trained.
"""

import numpy as np
import cv2
from config.settings import cfg
from ocr.utils.logger import get_logger

log = get_logger(__name__)
C = cfg.detection


# ── Smoothing ─────────────────────────────────────────────────────────────────

def smooth_projection(arr: np.ndarray, window: int) -> np.ndarray:
    """
    1-D Gaussian smoothing of the projection profile.

    Why smooth?
    Handwriting has ascenders (tall letters like 'h', 'l') and descenders
    ('g', 'p', 'y') that create small dips in the projection even within
    a single line. Smoothing merges these dips so we only cut at true
    inter-line gaps.

    Args:
        arr:    1-D projection array.
        window: Smoothing window size in pixels.

    Returns:
        Smoothed 1-D array, same length as input.
    """
    if window < 3:
        return arr
    kernel = np.exp(
        -0.5 * (np.arange(window) - window // 2) ** 2 / (window / 4) ** 2
    )
    kernel /= kernel.sum()
    return np.convolve(arr, kernel, mode="same")


# ── Valley detection ──────────────────────────────────────────────────────────

def find_valley_cuts(projection: np.ndarray, min_px: int) -> list:
    """
    Find row indices where the projection drops below a threshold.
    These are the gaps between text lines.

    Algorithm:
      1. Smooth the projection to remove ascender/descender noise.
      2. Find the global peak (maximum ink density).
      3. Threshold = peak × overlap_min_valley_depth (default 0.35).
         Only valleys that drop below 35% of the peak are real gaps.
      4. For each contiguous below-threshold region, take the midpoint
         as the cut row.

    Args:
        projection: 1-D array of ink pixel counts per row.
        min_px:     Minimum ink pixels to consider a row non-empty.

    Returns:
        Sorted list of cut row indices.
    """
    smoothed = smooth_projection(projection.astype(float), C.overlap_valley_smooth)
    peak = smoothed.max()
    if peak == 0:
        return []

    threshold = peak * C.overlap_min_valley_depth
    cuts = []
    in_valley = False
    valley_start = 0

    for i, v in enumerate(smoothed):
        if not in_valley and v < threshold:
            in_valley = True
            valley_start = i
        elif in_valley and v >= threshold:
            in_valley = False
            cuts.append((valley_start + i) // 2)

    return cuts


# ── Seam refinement ───────────────────────────────────────────────────────────

def refine_cut_with_seam(binary: np.ndarray, y_cut: int) -> int:
    """
    Refine a cut row by finding the minimum-ink row in a small band.

    When two lines have overlapping ascenders/descenders, the valley
    in the projection is not at the true gap — it's slightly off.
    This function looks in a ±seam_iterations pixel band and finds
    the row with the fewest ink pixels (the true gap).

    Args:
        binary: Binary page image (0=ink, 255=background).
        y_cut:  Initial cut row from valley detection.

    Returns:
        Refined cut row index.
    """
    half = C.overlap_seam_iterations
    y1 = max(0, y_cut - half)
    y2 = min(binary.shape[0], y_cut + half + 1)
    band = binary[y1:y2, :]
    ink_per_row = np.sum(band == 0, axis=1)
    best_offset = int(np.argmin(ink_per_row))
    return y1 + best_offset


# ── Main projection detector ──────────────────────────────────────────────────

def projection_detect(binary: np.ndarray) -> list:
    """
    Detect text lines using horizontal projection profiling.

    Args:
        binary: Binary page image (0=ink, 255=background), shape (H, W).

    Returns:
        List of (x1, y1, x2, y2) bounding boxes, sorted top-to-bottom.
        x1=0, x2=W (full page width) — tightened later by detector.py.
    """
    h, w = binary.shape
    projection = np.sum(binary == 0, axis=1).astype(float)
    min_px = max(3, int(w * C.projection_min_pixel_ratio))

    cuts = find_valley_cuts(projection, min_px)
    refined_cuts = sorted(set(refine_cut_with_seam(binary, c) for c in cuts))

    boundaries = [0] + refined_cuts + [h]
    boxes = []
    for i in range(len(boundaries) - 1):
        y1, y2 = boundaries[i], boundaries[i + 1]
        if y2 - y1 < C.min_line_height:
            continue
        if np.sum(binary[y1:y2, :] == 0) < min_px:
            continue
        boxes.append((0, y1, w, y2))

    log.debug("Projection detected %d lines", len(boxes))
    return boxes


# ── Projection within a CRAFT region ─────────────────────────────────────────

def projection_refine_region(
    binary: np.ndarray,
    region_box: tuple,
) -> list:
    """
    Apply projection profiling WITHIN a single CRAFT-detected region.

    This is Stage 3 of the pipeline: CRAFT gives us a candidate text
    region (possibly containing multiple lines), and projection profiling
    splits it into individual lines.

    Args:
        binary:     Full binary page image.
        region_box: (x1, y1, x2, y2) CRAFT region bounding box.

    Returns:
        List of (x1, y1, x2, y2) line boxes within the region,
        in original image coordinates.
    """
    x1, y1, x2, y2 = [int(v) for v in region_box]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(binary.shape[1], x2)
    y2 = min(binary.shape[0], y2)

    region = binary[y1:y2, x1:x2]
    if region.size == 0:
        return [region_box]

    # Run projection on the cropped region
    local_boxes = projection_detect(region)

    if not local_boxes:
        return [region_box]

    # Convert local coordinates back to full image coordinates
    global_boxes = []
    for lx1, ly1, lx2, ly2 in local_boxes:
        global_boxes.append((
            x1 + lx1,
            y1 + ly1,
            x1 + lx2,
            y1 + ly2,
        ))

    return global_boxes


# ── Projection profile visualization ─────────────────────────────────────────

def save_projection_visualization(
    binary: np.ndarray,
    boxes: list,
    output_path: str,
):
    """
    Save a visualization of the projection profile and detected line cuts.

    Left panel:  original binary image with line boxes drawn in green.
    Right panel: horizontal projection profile as a bar chart.

    Args:
        binary:      Binary page image.
        boxes:       Detected line bounding boxes.
        output_path: Where to save the visualization PNG.
    """
    h, w = binary.shape
    projection = np.sum(binary == 0, axis=1).astype(float)

    # Normalize projection for display
    max_proj = projection.max() if projection.max() > 0 else 1
    proj_norm = (projection / max_proj * 200).astype(int)

    # Create side-by-side visualization
    vis_w = w + 220
    vis = np.ones((h, vis_w, 3), dtype=np.uint8) * 255

    # Left: binary image with boxes
    binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(binary_bgr, (x1, y1), (x2, y2), (0, 200, 0), 2)
    vis[:, :w] = binary_bgr

    # Right: projection profile
    for y in range(h):
        bar_len = proj_norm[y]
        cv2.line(vis, (w + 10, y), (w + 10 + bar_len, y), (200, 100, 0), 1)

    # Draw cut lines
    smoothed = smooth_projection(projection, C.overlap_valley_smooth)
    cuts = find_valley_cuts(projection, 3)
    for cut in cuts:
        cv2.line(vis, (0, cut), (vis_w, cut), (0, 0, 255), 1)

    cv2.imwrite(output_path, vis)
    log.debug("Projection visualization saved: %s", output_path)
