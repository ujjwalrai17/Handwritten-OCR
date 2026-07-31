"""
Text Detection & Line Segmentation
Primary:  CRAFT (persistent singleton, loaded once)
Fallback: Horizontal projection profiling
Output:   List of TextLine objects (bbox + cropped binary image)
"""

import numpy as np
import cv2
from dataclasses import dataclass
from PIL import Image
from config.settings import cfg
from ocr.utils.logger import get_logger

log = get_logger(__name__)
C = cfg.detection


@dataclass
class TextLine:
    bbox: tuple[int, int, int, int]   # (x1, y1, x2, y2)
    crop: np.ndarray                   # cropped binary image


# ── CRAFT Singleton ───────────────────────────────────────────────────────────

_craft = None

def _get_craft():
    global _craft
    if _craft is not None:
        return _craft
    if not C.use_craft:
        return None
    try:
        from craft_text_detector import Craft
        _craft = Craft(output_dir=None, crop_type="box", cuda=False)
        log.info("CRAFT model loaded")
        return _craft
    except Exception as e:
        log.warning("CRAFT unavailable (%s) — using projection profiling", e)
        return None


def _craft_detect(bgr: np.ndarray) -> list[tuple]:
    craft = _get_craft()
    if craft is None:
        return []
    try:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        result = craft.detect_text(rgb)
        boxes = []
        for box in result["boxes"]:
            pts = np.array(box, dtype=np.int32)
            boxes.append((
                int(pts[:, 0].min()), int(pts[:, 1].min()),
                int(pts[:, 0].max()), int(pts[:, 1].max()),
            ))
        return sorted(boxes, key=lambda b: (b[1], b[0]))
    except Exception as e:
        log.warning("CRAFT inference failed: %s", e)
        return []


# ── Projection Profiling ──────────────────────────────────────────────────────

def _projection_detect(binary: np.ndarray) -> list[tuple]:
    h, w = binary.shape

    # Use local minima in projection to find gaps between lines
    projection = np.sum(binary == 0, axis=1).astype(float)

    # Normalize and find valleys (gaps between text lines)
    threshold = projection.mean() * 0.3
    min_px = max(3, int(w * C.projection_min_pixel_ratio))

    in_line, start = False, 0
    boxes = []
    for i, count in enumerate(projection):
        if not in_line and count >= min_px:
            in_line, start = True, i
        elif in_line and count < threshold:
            in_line = False
            if i - start > C.min_line_height:
                boxes.append((0, start, w, i))
    if in_line and h - start > C.min_line_height:
        boxes.append((0, start, w, h))
    return boxes


# ── Word → Line Grouping ──────────────────────────────────────────────────────

def _merge(boxes: list[tuple]) -> tuple:
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def _group_into_lines(word_boxes: list[tuple]) -> list[tuple]:
    if not word_boxes:
        return []
    lines: list[list[tuple]] = []
    for box in sorted(word_boxes, key=lambda b: b[1]):
        x1, y1, x2, y2 = box
        bh = max(1, y2 - y1)
        placed = False
        for line in lines:
            _, ly1, _, ly2 = _merge(line)
            lh = max(1, ly2 - ly1)
            if min(y2, ly2) - max(y1, ly1) > C.word_overlap_threshold * min(bh, lh):
                line.append(box)
                placed = True
                break
        if not placed:
            lines.append([box])
    return sorted([_merge(ln) for ln in lines], key=lambda b: b[1])


# ── Main Entry ────────────────────────────────────────────────────────────────

def detect_lines(binary: np.ndarray, bgr: np.ndarray) -> list[TextLine]:
    """
    Detect text lines in the image.
    Returns list of TextLine sorted top-to-bottom.
    """
    word_boxes = _craft_detect(bgr)
    if word_boxes:
        line_boxes = _group_into_lines(word_boxes)
        log.debug("CRAFT detected %d word boxes → %d lines", len(word_boxes), len(line_boxes))
    else:
        line_boxes = _projection_detect(binary)
        log.debug("Projection profiling detected %d lines", len(line_boxes))

    h, w = binary.shape
    p = C.line_padding
    lines = []
    for (x1, y1, x2, y2) in line_boxes:
        x1c, y1c = max(0, x1 - p), max(0, y1 - p)
        x2c, y2c = min(w, x2 + p), min(h, y2 + p)
        crop = binary[y1c:y2c, x1c:x2c]
        if crop.shape[0] >= C.min_line_height and crop.shape[1] >= C.min_line_width:
            lines.append(TextLine(bbox=(x1c, y1c, x2c, y2c), crop=crop))

    log.info("Segmented %d valid text lines", len(lines))
    return lines
