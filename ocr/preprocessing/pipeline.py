"""
Preprocessing Module
Stages: load → validate → resize → grayscale → CLAHE → denoise
        → binarize (Otsu / adaptive fallback) → deskew → dilate
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
    Moment-based deskew using cv2.minAreaRect on dark pixel coordinates.
    O(n_dark_pixels) — no rotation loop.
    """
    coords = np.column_stack(np.where(binary == 0))
    if len(coords) < 100:
        return binary
    angle = cv2.minAreaRect(coords.astype(np.float32))[2]
    if angle < -45:
        angle += 90
    angle = float(np.clip(angle, -C.deskew_max_angle, C.deskew_max_angle))
    if abs(angle) < 0.5:
        return binary
    h, w = binary.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(binary, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)


def _dilate(binary: np.ndarray) -> np.ndarray:
    """Mild morphological closing to reconnect broken thin strokes."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)


def preprocess(source) -> tuple[np.ndarray, np.ndarray]:
    """
    Full preprocessing pipeline.

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
    binary = _binarize(gray)
    binary = _deskew(binary)
    binary = _dilate(binary)
    log.debug("Preprocessing done: output shape %s", binary.shape)
    return binary, bgr
