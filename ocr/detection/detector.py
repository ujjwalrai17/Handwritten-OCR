"""
Main Detection Controller
=========================
Orchestrates the three-stage text-line detection pipeline:

  Stage 1 — U-Net Line Segmentation
      Predicts a pixel-level mask of text-line regions.
      Gives candidate line bounding boxes.

  Stage 2 — CRAFT Text Refinement
      Runs inside each U-Net candidate region.
      Produces character-level heatmaps to refine region boundaries.

  Stage 3 — Projection Profiling
      Runs inside each CRAFT region.
      Uses horizontal ink-density analysis to find exact line cuts.
      Always runs — it is the final refinement step.

Pipeline flow
-------------
    Input page image
         ↓
    [Stage 1] U-Net  →  candidate regions
         ↓
    [Stage 2] CRAFT  →  refined text regions
         ↓
    [Stage 3] Projection  →  final line boxes
         ↓
    TextLine objects (bbox + crop + difficulty_tag)

Fallback behaviour
------------------
If use_unet=False or checkpoint missing  → skip Stage 1, pass full page to Stage 2
If use_craft=False or checkpoint missing → skip Stage 2, pass U-Net regions to Stage 3
If both disabled                         → Stage 3 runs on full page (classical mode)

Error handling
--------------
If use_unet=True but checkpoint does NOT exist → raises RuntimeError (no silent fallback)
If use_craft=True but checkpoint does NOT exist → raises RuntimeError (no silent fallback)
Set use_unet=False / use_craft=False to explicitly use projection-only mode.
"""

import numpy as np
import cv2
import torch
from dataclasses import dataclass
from pathlib import Path
from PIL import Image

from config.settings import cfg
from ocr.utils.logger import get_logger

log = get_logger(__name__)
C = cfg.detection


# ── TextLine dataclass ────────────────────────────────────────────────────────

@dataclass
class TextLine:
    bbox: tuple           # (x1, y1, x2, y2) in original image coordinates
    crop: np.ndarray      # BGR or grayscale crop for TrOCR
    difficulty_tag: str = "clean"   # "clean" | "hard"


# ── Difficulty tagging ────────────────────────────────────────────────────────

def _tag_difficulty(crop: np.ndarray) -> str:
    """
    Tag a line crop as 'hard' or 'clean' based on ink-density variance.
    High variance = overlapping ascenders/descenders = hard sample.
    """
    if crop is None or crop.size == 0:
        return "clean"
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    ink_col = np.sum(gray == 0, axis=0).astype(float)
    if ink_col.max() == 0:
        return "clean"
    ink_col /= ink_col.max()
    return "hard" if float(np.var(ink_col)) > C.overlap_variance_threshold else "clean"


# ── Stage 1: U-Net ────────────────────────────────────────────────────────────

_unet_model = None


def _load_unet():
    global _unet_model
    if _unet_model is not None:
        return _unet_model

    ckpt = Path(C.unet_checkpoint)
    if not ckpt.exists():
        raise RuntimeError(
            f"U-Net checkpoint not found: '{ckpt}'\n"
            f"Train U-Net first:  python ocr/detection/train_unet.py\n"
            f"Or disable U-Net:   set use_unet=False in config/settings.py"
        )

    from ocr.detection.unet import LineSegUNet
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LineSegUNet(base_ch=32)
    model.load_state_dict(torch.load(str(ckpt), map_location=device, weights_only=True))
    model.to(device).eval()
    _unet_model = model
    log.info("U-Net loaded from %s", ckpt)
    return _unet_model


@torch.inference_mode()
def _run_unet(binary: np.ndarray) -> tuple:
    """
    Run U-Net on the full binary page image.

    Returns:
        prob_mask:   (H, W) float32 probability map
        candidate_boxes: list of (x1, y1, x2, y2) from U-Net mask
    """
    from ocr.detection.postprocessing import unet_mask_to_boxes

    model = _load_unet()
    device = next(model.parameters()).device
    h_orig, w_orig = binary.shape

    resized = cv2.resize(binary, (C.unet_input_width, C.unet_input_height),
                         interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized.astype(np.float32) / 255.0)
    tensor = tensor.unsqueeze(0).unsqueeze(0).to(device)

    prob_mask = model(tensor)[0, 0].cpu().numpy()  # (H, W) in [0,1]

    # Scale back to original resolution
    prob_full = cv2.resize(prob_mask, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
    boxes = unet_mask_to_boxes(prob_full, h_orig, w_orig)

    log.info("U-Net: %d candidate regions", len(boxes))
    return prob_full, boxes


# ── Stage 2: CRAFT ────────────────────────────────────────────────────────────

_craft_model = None


def _load_craft():
    global _craft_model
    if _craft_model is not None:
        return _craft_model

    ckpt = Path(C.craft_checkpoint)
    if not ckpt.exists():
        raise RuntimeError(
            f"CRAFT checkpoint not found: '{ckpt}'\n"
            f"Train CRAFT first:  python ocr/detection/train_craft.py\n"
            f"Or disable CRAFT:   set use_craft=False in config/settings.py"
        )

    from ocr.detection.craft import CRAFTDetector
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CRAFTDetector()
    model.load_state_dict(torch.load(str(ckpt), map_location=device, weights_only=True))
    model.to(device).eval()
    _craft_model = model
    log.info("CRAFT loaded from %s", ckpt)
    return _craft_model


@torch.inference_mode()
def _run_craft(bgr: np.ndarray, candidate_boxes: list) -> tuple:
    """
    Run CRAFT inside each U-Net candidate region.

    For each candidate box, crop the region, run CRAFT, extract
    refined text regions from the character score map.

    Returns:
        char_map_full:  (H, W) float32 character score on full image
        aff_map_full:   (H, W) float32 affinity score on full image
        refined_boxes:  list of (x1, y1, x2, y2) refined regions
    """
    from ocr.detection.postprocessing import craft_scores_to_boxes

    model = _load_craft()
    device = next(model.parameters()).device
    h_orig, w_orig = bgr.shape[:2]

    char_map_full = np.zeros((h_orig, w_orig), dtype=np.float32)
    aff_map_full  = np.zeros((h_orig, w_orig), dtype=np.float32)
    refined_boxes = []

    # If no candidate boxes from U-Net, run CRAFT on full page
    regions = candidate_boxes if candidate_boxes else [(0, 0, w_orig, h_orig)]

    for (rx1, ry1, rx2, ry2) in regions:
        rx1, ry1 = max(0, rx1), max(0, ry1)
        rx2, ry2 = min(w_orig, rx2), min(h_orig, ry2)
        region_bgr = bgr[ry1:ry2, rx1:rx2]
        if region_bgr.size == 0:
            continue

        rh, rw = region_bgr.shape[:2]

        # Normalize and convert to tensor
        rgb = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)

        scores = model(tensor)[0].cpu().numpy()  # (2, H/2, W/2)
        char_score = scores[0]
        aff_score  = scores[1]

        # Scale scores to region size
        char_full = cv2.resize(char_score, (rw, rh), interpolation=cv2.INTER_LINEAR)
        aff_full  = cv2.resize(aff_score,  (rw, rh), interpolation=cv2.INTER_LINEAR)

        # Place into full-image maps
        char_map_full[ry1:ry2, rx1:rx2] = np.maximum(
            char_map_full[ry1:ry2, rx1:rx2], char_full
        )
        aff_map_full[ry1:ry2, rx1:rx2] = np.maximum(
            aff_map_full[ry1:ry2, rx1:rx2], aff_full
        )

        # Extract boxes from this region's char score
        local_boxes = craft_scores_to_boxes(char_score, rh, rw)
        for (lx1, ly1, lx2, ly2) in local_boxes:
            refined_boxes.append((rx1 + lx1, ry1 + ly1, rx1 + lx2, ry1 + ly2))

    if not refined_boxes:
        # CRAFT found nothing — fall back to candidate boxes
        refined_boxes = regions

    log.info("CRAFT: %d refined regions", len(refined_boxes))
    return char_map_full, aff_map_full, refined_boxes


# ── Stage 3: Projection profiling ────────────────────────────────────────────

def _run_projection(binary: np.ndarray, regions: list) -> list:
    """
    Run projection profiling inside each region to find exact line cuts.

    For each region box, crop the binary image and apply horizontal
    projection profiling to split it into individual text lines.

    Returns:
        List of (x1, y1, x2, y2) final line boxes, sorted top-to-bottom.
    """
    from ocr.detection.projection_detector import projection_refine_region

    all_boxes = []
    for region in regions:
        line_boxes = projection_refine_region(binary, region)
        all_boxes.extend(line_boxes)

    all_boxes.sort(key=lambda b: b[1])
    log.info("Projection: %d final line boxes", len(all_boxes))
    return all_boxes


# ── Bleed component reassignment (existing logic, preserved) ──────────────────

def _reassign_bleed_components(binary: np.ndarray, line_boxes: list) -> list:
    if len(line_boxes) < 2:
        return line_boxes

    page_h, page_w = binary.shape
    median_line_h = float(np.median([b[3] - b[1] for b in line_boxes]))
    max_component_h = max(C.min_line_height, median_line_h * 2.5)
    max_component_w = page_w * 0.80

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (binary == 0).astype(np.uint8), connectivity=8
    )
    line_centers = [(b[1] + b[3]) / 2 for b in line_boxes]
    adjusted = [list(b) for b in line_boxes]

    for lbl in range(1, num_labels):
        comp_x  = stats[lbl, cv2.CC_STAT_LEFT]
        comp_y1 = stats[lbl, cv2.CC_STAT_TOP]
        comp_w  = stats[lbl, cv2.CC_STAT_WIDTH]
        comp_h  = stats[lbl, cv2.CC_STAT_HEIGHT]
        comp_area = stats[lbl, cv2.CC_STAT_AREA]

        touches_edge = (
            comp_x <= 1 or comp_y1 <= 1
            or comp_x + comp_w >= page_w - 1
            or comp_y1 + comp_h >= page_h - 1
        )
        if comp_h > max_component_h or comp_w > max_component_w:
            continue
        if touches_edge and comp_area > page_w * 0.02:
            continue

        cy = centroids[lbl][1]
        nearest = int(np.argmin([abs(cy - lc) for lc in line_centers]))
        comp_y2 = comp_y1 + comp_h
        _, orig_y1, _, orig_y2 = line_boxes[nearest]
        expansion_limit = max(C.line_padding * 3, int(median_line_h * 0.35))
        if comp_y2 < orig_y1 - expansion_limit or comp_y1 > orig_y2 + expansion_limit:
            continue
        adjusted[nearest][1] = min(adjusted[nearest][1], comp_y1)
        adjusted[nearest][3] = max(adjusted[nearest][3], comp_y2)

    return [tuple(b) for b in adjusted]


# ── Verify segmentation ───────────────────────────────────────────────────────

def verify_line_segmentation(binary: np.ndarray, detected_lines: list,
                              smooth_window: int = 11,
                              merge_height_ratio: float = 1.8) -> dict:
    from ocr.detection.projection_detector import smooth_projection

    if not detected_lines:
        return {"peak_estimate": 0, "detected_count": 0,
                "merged_crops": [], "warning": "No lines detected."}

    projection = np.sum(binary == 0, axis=1).astype(float)
    smoothed   = smooth_projection(projection, smooth_window)
    peak_val   = smoothed.max()
    peak_estimate = 0
    if peak_val > 0:
        threshold = peak_val * 0.15
        in_peak = False
        for v in smoothed > threshold:
            if v and not in_peak:
                peak_estimate += 1
                in_peak = True
            elif not v:
                in_peak = False

    detected_count = len(detected_lines)
    heights = [ln.bbox[3] - ln.bbox[1] for ln in detected_lines]
    median_h = float(np.median(heights))
    merge_threshold = median_h * merge_height_ratio
    merged_crops = []

    for idx, ln in enumerate(detected_lines):
        x1, y1, x2, y2 = ln.bbox
        h = y2 - y1
        if h > merge_threshold:
            merged_crops.append({"index": idx, "bbox": ln.bbox, "height": int(h)})
            log.warning("Possible merged crop at line %d: height=%dpx", idx, h)

    warning = None
    issues = []
    if peak_estimate > 0 and detected_count < peak_estimate * 0.70:
        issues.append(f"detected {detected_count} but projection suggests ~{peak_estimate}")
    if merged_crops:
        issues.append(f"{len(merged_crops)} crop(s) may be merged lines")
    if issues:
        warning = "Line segmentation warning: " + "; ".join(issues)
        log.warning(warning)

    return {"peak_estimate": peak_estimate, "detected_count": detected_count,
            "merged_crops": merged_crops, "warning": warning}


# ── Main entry point ──────────────────────────────────────────────────────────

def detect_lines(
    binary: np.ndarray,
    bgr: np.ndarray,
    debug_dir: str = None,
) -> list:
    """
    Full three-stage detection pipeline.

    Args:
        binary:    (H, W) binary image — 0=ink, 255=background.
        bgr:       (H, W, 3) color image — used by CRAFT and for crops.
        debug_dir: If set, saves all stage visualizations here.

    Returns:
        List of TextLine objects sorted top-to-bottom.
    """
    from ocr.detection.postprocessing import tighten_to_ink

    h, w = binary.shape
    unet_prob   = None
    unet_boxes  = []
    char_map    = None
    aff_map     = None
    craft_boxes = []

    # ── Stage 1: U-Net ────────────────────────────────────────────────────────
    if C.use_unet:
        unet_prob, unet_boxes = _run_unet(binary)
    else:
        log.info("U-Net disabled — skipping Stage 1")

    # ── Stage 2: CRAFT ────────────────────────────────────────────────────────
    if C.use_craft:
        char_map, aff_map, craft_boxes = _run_craft(bgr, unet_boxes)
    else:
        log.info("CRAFT disabled — skipping Stage 2")
        # Pass U-Net boxes (or full page) directly to projection
        craft_boxes = unet_boxes if unet_boxes else [(0, 0, w, h)]

    # ── Stage 3: Projection profiling ─────────────────────────────────────────
    line_boxes = _run_projection(binary, craft_boxes)
    line_boxes = _reassign_bleed_components(binary, line_boxes)

    # ── Build TextLine objects ─────────────────────────────────────────────────
    p = C.line_padding
    lines = []
    for (x1, y1, x2, y2) in line_boxes:
        x1, y1, x2, y2 = tighten_to_ink(binary, (x1, y1, x2, y2))
        x1c = max(0, x1 - p);  y1c = max(0, y1 - p)
        x2c = min(w, x2 + p);  y2c = min(h, y2 + p)
        binary_crop = binary[y1c:y2c, x1c:x2c]
        color_crop  = bgr[y1c:y2c, x1c:x2c] if bgr.shape[:2] == binary.shape else binary_crop
        if binary_crop.shape[0] >= C.min_line_height and binary_crop.shape[1] >= C.min_line_width:
            tag = _tag_difficulty(binary_crop)
            lines.append(TextLine(bbox=(x1c, y1c, x2c, y2c),
                                  crop=color_crop, difficulty_tag=tag))

    hard = sum(1 for ln in lines if ln.difficulty_tag == "hard")
    log.info("Detected %d lines (%d hard / %d clean)", len(lines), hard, len(lines) - hard)

    verify_line_segmentation(binary, lines)

    # ── Debug visualizations ──────────────────────────────────────────────────
    if debug_dir:
        from ocr.detection.visualizer import (
            save_original, save_unet_outputs, save_craft_outputs,
            save_projection_visualization, save_final_lines,
        )
        save_original(bgr, debug_dir)
        if unet_prob is not None:
            from ocr.detection.postprocessing import unet_mask_to_boxes
            binary_mask = (unet_prob > C.unet_threshold).astype(np.uint8) * 255
            save_unet_outputs(unet_prob, binary_mask, bgr, debug_dir)
        if char_map is not None:
            save_craft_outputs(char_map, aff_map, craft_boxes, bgr, debug_dir)
        save_projection_visualization(binary, [(ln.bbox) for ln in lines], debug_dir)
        save_final_lines(bgr, [ln.bbox for ln in lines], debug_dir)

    return lines


# ── Hierarchical word→line detection (preserved for compatibility) ────────────

def detect_words_hierarchical(bgr: np.ndarray, pil_img: Image.Image) -> list:
    from ocr.detection.word_detector import WordDetector
    from ocr.detection.line_grouper import SpatialLineGrouper
    from PIL import Image as _PILImage

    detector = WordDetector()
    grouper  = SpatialLineGrouper(overlap_ratio=0.6)
    word_regions = detector.detect(pil_img)
    if not word_regions:
        return []

    line_groups = grouper.group(word_regions)
    if not line_groups:
        return []

    lines = []
    word_gap = 8
    for grp in line_groups:
        crops = [w.crop_pil.convert("L") for w in grp.words]
        if not crops:
            continue
        line_h  = max(c.height for c in crops)
        total_w = sum(c.width for c in crops) + word_gap * (len(crops) - 1)
        strip   = _PILImage.new("L", (total_w, line_h), color=255)
        x_off   = 0
        for crop in crops:
            strip.paste(crop, (x_off, (line_h - crop.height) // 2))
            x_off += crop.width + word_gap
        strip_np = np.array(strip)
        if strip_np.shape[0] >= C.min_line_height and strip_np.shape[1] >= C.min_line_width:
            tag = _tag_difficulty(strip_np)
            lines.append(TextLine(bbox=grp.bbox, crop=strip_np, difficulty_tag=tag))

    return lines
