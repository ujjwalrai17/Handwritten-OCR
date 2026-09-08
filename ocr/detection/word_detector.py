"""
Stage 1 — Word Detection & Per-Word Rotation
=============================================
Detects individual word bounding polygons in a document image and returns
axis-aligned (0°) cropped word images via affine rotation.

Detection priority
------------------
1. CRAFT  — character-region affinity maps; handles tilted / curved words.
2. OpenCV oriented bounding boxes (OBB) — contour-based fallback; zero extra
   dependencies beyond opencv-contrib-python.

Both paths return the same ``WordRegion`` datatype so the rest of the
pipeline is unaffected by which detector ran.

Usage
-----
    detector = WordDetector()
    words = detector.detect(pil_image)
    for w in words:
        w.crop_pil.save(f"word_{w.index}.png")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from ocr.utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Output datatype
# ---------------------------------------------------------------------------

@dataclass
class WordRegion:
    """
    A single detected word region.

    Attributes:
        index:    Zero-based detection order index.
        polygon:  Tight (4, 2) float32 polygon corners (x, y).
        bbox:     Axis-aligned bbox ``(x1, y1, x2, y2)`` in original image coords.
        angle:    Rotation angle in degrees applied to straighten the word.
        crop_pil: Affine-rotated, axis-aligned PIL RGB word crop.
        score:    Detection confidence in ``[0, 1]``.
    """
    index:    int
    polygon:  np.ndarray          # (4, 2) float32
    bbox:     Tuple[int, int, int, int]
    angle:    float
    crop_pil: Image.Image
    score:    float = 1.0


# ---------------------------------------------------------------------------
# Affine rotation helper
# ---------------------------------------------------------------------------

def _rotate_crop(
    image_np: np.ndarray,
    polygon: np.ndarray,
    angle: float,
    padding: int = 4,
) -> Image.Image:
    """
    Extract a word crop by rotating the source image so the word is horizontal.

    Args:
        image_np: (H, W, 3) uint8 RGB source image.
        polygon:  (4, 2) float32 word polygon.
        angle:    Tilt angle in degrees (from minAreaRect).
        padding:  Extra pixels around the tight crop.

    Returns:
        Axis-aligned PIL RGB word crop.
    """
    cx = float(polygon[:, 0].mean())
    cy = float(polygon[:, 1].mean())

    H, W = image_np.shape[:2]
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(
        image_np, M, (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    # Rotate polygon corners to find tight axis-aligned bbox
    ones = np.ones((4, 1), dtype=np.float32)
    rot_pts = (M @ np.hstack([polygon, ones]).T).T  # (4, 2)

    x1 = max(0, int(rot_pts[:, 0].min()) - padding)
    y1 = max(0, int(rot_pts[:, 1].min()) - padding)
    x2 = min(W, int(rot_pts[:, 0].max()) + padding)
    y2 = min(H, int(rot_pts[:, 1].max()) + padding)

    crop = rotated[y1:y2, x1:x2]
    if crop.size == 0:
        crop = image_np[
            max(0, int(polygon[:, 1].min())):int(polygon[:, 1].max()) + 1,
            max(0, int(polygon[:, 0].min())):int(polygon[:, 0].max()) + 1,
        ]
    return Image.fromarray(crop)


# ---------------------------------------------------------------------------
# CRAFT detector
# ---------------------------------------------------------------------------

def _craft_word_boxes(bgr: np.ndarray) -> List[Tuple[np.ndarray, float]]:
    """
    Run CRAFT and return (polygon, score) pairs for each detected word.

    Args:
        bgr: BGR uint8 image.

    Returns:
        List of ``(polygon_4x2_float32, score)`` tuples.
        Empty list if CRAFT is unavailable or fails.
    """
    try:
        from craft_text_detector import Craft
        craft = Craft(output_dir=None, crop_type="poly", cuda=False)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        result = craft.detect_text(rgb)
        craft.unload_craftnet_model()
        craft.unload_refinenet_model()

        out = []
        for box in result.get("boxes", []):
            pts = np.array(box, dtype=np.float32).reshape(4, 2)
            out.append((pts, 1.0))
        log.debug("CRAFT detected %d word boxes.", len(out))
        return out
    except Exception as exc:
        log.warning("CRAFT word detection failed (%s) — using OBB fallback.", exc)
        return []


# ---------------------------------------------------------------------------
# OpenCV OBB fallback
# ---------------------------------------------------------------------------

def _obb_word_boxes(bgr: np.ndarray) -> List[Tuple[np.ndarray, float]]:
    """
    Detect word bounding polygons using OpenCV contour-based oriented bounding
    boxes (OBB).

    Strategy:
      1. Grayscale → CLAHE → Otsu binarisation.
      2. Morphological dilation to merge characters within a word.
      3. Find external contours → minAreaRect → 4-corner polygon.

    Args:
        bgr: BGR uint8 image.

    Returns:
        List of ``(polygon_4x2_float32, score)`` tuples.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Dilate horizontally to connect characters into word blobs
    h, w = binary.shape
    kw = max(10, w // 40)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 3))
    dilated = cv2.dilate(binary, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = (h * w) * 0.0002   # ignore tiny noise blobs
    out = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        rect = cv2.minAreaRect(cnt)          # ((cx,cy), (w,h), angle)
        box  = cv2.boxPoints(rect)           # (4, 2) float32
        out.append((box.astype(np.float32), 1.0))

    log.debug("OBB fallback detected %d word boxes.", len(out))
    return out


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class WordDetector:
    """
    Stage 1 — Word Detection & Per-Word Affine Rotation.

    Detects individual word bounding polygons and returns axis-aligned
    (0°) cropped word images ready for TrOCR recognition.

    Detection priority:
      1. CRAFT  (when ``craft-text-detector`` is installed)
      2. OpenCV oriented bounding boxes (always available)

    Args:
        min_word_width:  Minimum word crop width in pixels (default 8).
        min_word_height: Minimum word crop height in pixels (default 8).
        padding:         Extra pixels around each rotated crop (default 4).

    Example:
        >>> detector = WordDetector()
        >>> words = detector.detect(Image.open("page.jpg"))
        >>> print(len(words), "words detected")
    """

    def __init__(
        self,
        min_word_width: int = 8,
        min_word_height: int = 8,
        padding: int = 4,
    ) -> None:
        self._min_w   = min_word_width
        self._min_h   = min_word_height
        self._padding = padding

    def detect(self, image: Image.Image) -> List[WordRegion]:
        """
        Detect all word regions in a document image.

        Args:
            image: Input PIL RGB image (any size).

        Returns:
            List of :class:`WordRegion` objects in raw detection order.
            Sorting into reading order is handled by :class:`SpatialLineGrouper`.

        Raises:
            ValueError: If *image* is ``None`` or has zero dimensions.
        """
        if image is None:
            raise ValueError("WordDetector.detect: received None image.")
        if image.size[0] == 0 or image.size[1] == 0:
            raise ValueError(f"WordDetector.detect: zero-dimension image {image.size}.")

        rgb_np = np.array(image.convert("RGB"), dtype=np.uint8)
        bgr    = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)

        # Priority 1: CRAFT
        raw = _craft_word_boxes(bgr)
        if not raw:
            # Priority 2: OBB fallback
            raw = _obb_word_boxes(bgr)

        if not raw:
            log.warning("WordDetector: no word boxes found.")
            return []

        regions: List[WordRegion] = []
        idx = 0
        for polygon, score in raw:
            # Compute minAreaRect angle for rotation
            rect  = cv2.minAreaRect(polygon.astype(np.float32))
            angle = rect[2]
            # OpenCV angle convention: normalise to [-45, 45]
            if rect[1][0] < rect[1][1]:
                angle += 90

            # Axis-aligned bbox from polygon
            x1 = max(0, int(polygon[:, 0].min()))
            y1 = max(0, int(polygon[:, 1].min()))
            x2 = min(rgb_np.shape[1], int(polygon[:, 0].max()))
            y2 = min(rgb_np.shape[0], int(polygon[:, 1].max()))

            if (x2 - x1) < self._min_w or (y2 - y1) < self._min_h:
                continue

            crop_pil = _rotate_crop(rgb_np, polygon, angle, self._padding)
            if crop_pil.width < self._min_w or crop_pil.height < self._min_h:
                continue

            regions.append(WordRegion(
                index=idx,
                polygon=polygon,
                bbox=(x1, y1, x2, y2),
                angle=angle,
                crop_pil=crop_pil,
                score=score,
            ))
            idx += 1

        log.info("WordDetector: %d valid word regions extracted.", len(regions))
        return regions
