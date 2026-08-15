"""
Preprocessing Module
Stages: load → validate → resize → grayscale → CLAHE → denoise
        → binarize (Otsu / adaptive fallback) → deskew → stroke-normalize → dilate

Enhancements for cursive/overlapping handwriting:
  - Stroke width normalization via distance-transform thinning
  - Per-line slant correction (Hough-based, not just global moment deskew)
  - Sauvola local binarization for uneven illumination (photographed notes)
"""

import cv2
import numpy as np
from PIL import Image
from config.settings import cfg
from ocr.utils.logger import get_logger

log = get_logger(__name__)
C = cfg.preprocessing


class PreprocessingError(ValueError):
    pass


def load_image(source) -> np.ndarray:
    """Accept file path (str) or PIL Image → BGR numpy array."""
    if isinstance(source, str):
        # Use PIL to handle paths with spaces/unicode on Windows
        try:
            pil = Image.open(source).convert("RGB")
            return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        except Exception:
            raise PreprocessingError(f"Cannot read image: {source}")
    if isinstance(source, Image.Image):
        return cv2.cvtColor(np.array(source.convert("RGB")), cv2.COLOR_RGB2BGR)
    raise PreprocessingError("source must be str path or PIL Image")


def _validate(bgr: np.ndarray):
    if bgr is None or bgr.size == 0:
        raise PreprocessingError("Empty image")
    h, w = bgr.shape[:2]
    if h < C.min_image_side or w < C.min_image_side:
        raise PreprocessingError(f"Image too small: {w}×{h}px")


def _resize(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    if max(h, w) <= C.max_image_side:
        return bgr
    scale = C.max_image_side / max(h, w)
    return cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def _grayscale(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def _clahe(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=C.clahe_clip_limit, tileGridSize=C.clahe_tile_size)
    return clahe.apply(gray)


def _denoise(gray: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(gray, C.gaussian_kernel, sigmaX=1)


def _binarize(gray: np.ndarray) -> np.ndarray:
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg_ratio = np.sum(otsu == 0) / otsu.size
    if fg_ratio < 0.02 or fg_ratio > 0.60:
        log.debug("Otsu fg_ratio=%.3f — switching to adaptive threshold", fg_ratio)
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
        )
    return otsu


def _deskew(binary: np.ndarray) -> np.ndarray:
    """
    Two-pass deskew:
      1. Global moment-based rotation (fast, handles page tilt)
      2. Hough-line refinement for residual slant in cursive strokes
    """
    coords = np.column_stack(np.where(binary == 0))
    if len(coords) < 100:
        return binary

    # Pass 1: moment-based
    angle = cv2.minAreaRect(coords.astype(np.float32))[2]
    if angle < -45:
        angle += 90
    angle = float(np.clip(angle, -C.deskew_max_angle, C.deskew_max_angle))
    if abs(angle) >= 0.5:
        h, w = binary.shape
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        binary = cv2.warpAffine(binary, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)

    # Pass 2: Hough-line slant refinement (catches cursive forward/backward lean)
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=max(30, binary.shape[1] // 8))
    if lines is not None:
        angles = []
        for rho, theta in lines[:, 0]:
            a = np.degrees(theta) - 90
            if abs(a) < C.deskew_max_angle:
                angles.append(a)
        if angles:
            median_angle = float(np.median(angles))
            if abs(median_angle) >= 0.3:
                h, w = binary.shape
                M = cv2.getRotationMatrix2D((w // 2, h // 2), median_angle, 1.0)
                binary = cv2.warpAffine(binary, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)
                log.debug("Hough slant correction: %.2f°", median_angle)
    return binary


def _normalize_stroke_width(binary: np.ndarray) -> np.ndarray:
    """
    Normalize variable stroke widths using distance-transform guided thinning.
    Thick strokes → skeleton → controlled re-dilation to uniform width.
    This reduces the visual impact of overlapping thick/thin strokes.
    """
    ink = (binary == 0).astype(np.uint8)
    if ink.sum() < 50:
        return binary

    # Distance transform on ink pixels
    dist = cv2.distanceTransform(ink, cv2.DIST_L2, 3)
    median_radius = float(np.median(dist[dist > 0])) if dist.max() > 0 else 1.0
    target_radius = max(1.0, min(median_radius, 2.5))  # clamp to [1, 2.5] px

    # Thin to skeleton
    skeleton = cv2.ximgproc.thinning(ink * 255) if hasattr(cv2, 'ximgproc') else ink * 255

    # Re-dilate to target_radius
    r = max(1, int(round(target_radius)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    normalized_ink = cv2.dilate(skeleton, kernel, iterations=1)
    result = np.where(normalized_ink > 0, 0, 255).astype(np.uint8)
    log.debug("Stroke normalization: median_r=%.2f → target_r=%.2f", median_radius, target_radius)
    return result


def _dilate(binary: np.ndarray) -> np.ndarray:
    """Mild morphological closing to reconnect broken thin strokes."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)


def _sauvola_binarize(gray: np.ndarray, window: int = 25, k: float = 0.2) -> np.ndarray:
    """
    Sauvola local binarization — superior to Otsu for photographed notes
    with uneven illumination and shadow gradients.
    """
    gray_f = gray.astype(np.float64)
    # Local mean and std via integral images
    mean = cv2.boxFilter(gray_f, ddepth=-1, ksize=(window, window))
    mean_sq = cv2.boxFilter(gray_f ** 2, ddepth=-1, ksize=(window, window))
    std = np.sqrt(np.maximum(mean_sq - mean ** 2, 0))
    threshold = mean * (1.0 + k * (std / 128.0 - 1.0))
    binary = np.where(gray_f < threshold, 0, 255).astype(np.uint8)
    return binary


def preprocess(source, normalize_strokes: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Full preprocessing pipeline.

    Args:
        source:           str path or PIL.Image
        normalize_strokes: if True, apply stroke-width normalization (slower,
                           helps with variable-width cursive strokes)
    Returns:
        binary (H×W uint8): binarized, deskewed image for detection
        bgr    (H×W×3 uint8): resized color image for CRAFT detection
    Raises:
        PreprocessingError on invalid input
    """
    bgr = load_image(source)
    _validate(bgr)
    bgr = _resize(bgr)
    gray = _grayscale(bgr)
    gray = _clahe(gray)
    gray = _denoise(gray)

    # Choose binarization: Sauvola for photographed notes, Otsu for scans
    binary = _binarize(gray)
    fg_ratio = np.sum(binary == 0) / binary.size
    if fg_ratio < 0.02 or fg_ratio > 0.55:
        log.debug("Switching to Sauvola binarization (fg_ratio=%.3f)", fg_ratio)
        binary = _sauvola_binarize(gray)

    binary = _deskew(binary)
    if normalize_strokes:
        binary = _normalize_stroke_width(binary)
    binary = _dilate(binary)
    log.debug("Preprocessing done: output shape %s", binary.shape)
    return binary, bgr
