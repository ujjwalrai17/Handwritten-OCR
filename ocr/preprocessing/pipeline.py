"""
Preprocessing Module — Accuracy-first rewrite
Priority: preserve every character stroke, dot, punctuation mark.

Key changes vs previous version:
  - Adaptive CLAHE: only applied when contrast is genuinely low
  - Adaptive denoising: skipped when image is already clean (avoids blurring thin strokes)
  - Multi-candidate binarization: Otsu, Adaptive-Gaussian, Sauvola all computed;
    best selected by component-quality score; all saved for debug
  - Deskew: only applied when angle > deskew_min_angle (avoids unnecessary rotation)
  - Morphological closing: kernel capped at (2,1) — never merges adjacent chars
  - Ruling-line removal: character intersections restored via dilation mask
  - Border removal: threshold raised so text near edges is not erased
  - OCR receives the COLOR crop (not binary) — TrOCR was trained on natural images
"""

import json
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from config.settings import cfg
from ocr.utils.logger import get_logger

log = get_logger(__name__)
C = cfg.preprocessing


# ── Helpers ───────────────────────────────────────────────────────────────────

class PreprocessingError(ValueError):
    pass


def _save_stage(img, debug_dir, name: str):
    """Save a preprocessing stage image; no-op when debug_dir is None."""
    if debug_dir is None:
        return
    Path(debug_dir).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(Path(debug_dir) / name), img)


def load_image(source) -> np.ndarray:
    """Accept file path (str) or PIL Image -> BGR numpy array."""
    if isinstance(source, str):
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
        raise PreprocessingError(f"Image too small: {w}x{h}px")


# ── Stage: Resize ─────────────────────────────────────────────────────────────

def _resize(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    longest = max(h, w)
    shortest = min(h, w)
    # Downscale only if truly oversized
    if longest > C.max_image_side:
        scale = C.max_image_side / longest
        return cv2.resize(bgr, (int(w * scale), int(h * scale)),
                          interpolation=cv2.INTER_AREA)
    # Upscale small images so TrOCR gets enough pixels per character
    if shortest < C.min_resize_side:
        scale = min(C.min_resize_side / shortest, C.max_image_side / longest)
        if scale > 1.05:
            interp = cv2.INTER_LANCZOS4 if scale >= 1.5 else cv2.INTER_CUBIC
            return cv2.resize(bgr, (int(w * scale), int(h * scale)),
                              interpolation=interp)
    return bgr


# ── Stage: Grayscale ──────────────────────────────────────────────────────────

def _grayscale(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


# ── Stage: CLAHE (adaptive — only when contrast is low) ──────────────────────

def _clahe(gray: np.ndarray) -> np.ndarray:
    std = float(np.std(gray))
    if std >= 55.0:
        # Image already has good contrast — CLAHE would add noise
        log.debug("CLAHE skipped (std=%.1f >= 55)", std)
        return gray
    clahe = cv2.createCLAHE(clipLimit=C.clahe_clip_limit,
                             tileGridSize=C.clahe_tile_size)
    result = clahe.apply(gray)
    log.debug("CLAHE applied (std=%.1f)", std)
    return result


# ── Stage: Denoising (adaptive — only when noise is high) ────────────────────

def _denoise(gray: np.ndarray) -> np.ndarray:
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if laplacian_var <= 750.0:
        # Low noise — skip blur to preserve thin strokes and dots
        log.debug("Denoising skipped (laplacian_var=%.1f <= 750)", laplacian_var)
        return gray
    result = cv2.GaussianBlur(gray, C.gaussian_kernel, sigmaX=1)
    log.debug("Gaussian blur applied (laplacian_var=%.1f)", laplacian_var)
    return result


# ── Stage: Binarization — multi-candidate with quality scoring ───────────────

def _sauvola_binarize(gray: np.ndarray, window: int = 25, k: float = 0.2) -> np.ndarray:
    gray_f = gray.astype(np.float64)
    mean = cv2.boxFilter(gray_f, ddepth=-1, ksize=(window, window))
    mean_sq = cv2.boxFilter(gray_f ** 2, ddepth=-1, ksize=(window, window))
    std = np.sqrt(np.maximum(mean_sq - mean ** 2, 0))
    threshold = mean * (1.0 + k * (std / 128.0 - 1.0))
    return np.where(gray_f < threshold, 0, 255).astype(np.uint8)


def _adaptive_gaussian(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    block = max(25, int(min(h, w) * 0.035) | 1)
    block = min(block, 75)
    if block % 2 == 0:
        block += 1
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, block, 11
    )


def _score_binary(binary: np.ndarray) -> float:
    """
    Score a binarization candidate for OCR suitability.
    Higher = better. Penalises: too much/little ink, too many tiny components
    (noise dots), too few components (merged chars), huge blobs (border noise).
    """
    ink = (binary == 0).astype(np.uint8)
    fg = float(ink.mean())
    if fg <= 0.002 or fg >= 0.65:
        return -1000.0
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA] if num_labels > 1 else np.array([1])
    tiny_ratio = float(np.mean(areas <= 2))
    large_ratio = float(np.mean(areas > binary.size * 0.01))
    score = 0.0
    score -= abs(fg - 0.085) * 12.0   # ideal ink density ~8.5%
    score -= tiny_ratio * 1.8          # penalise noise dots
    score -= large_ratio * 4.0         # penalise merged blobs
    if areas.size < 8:
        score -= 2.0                   # too few components
    return score


def _select_binarization(
    gray: np.ndarray, debug_dir=None
) -> tuple:
    """
    Compute Otsu, Adaptive-Gaussian, and Sauvola; save all three for debug;
    return (best_binary, method_name, scores_dict).
    """
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = _adaptive_gaussian(gray)
    sauvola = _sauvola_binarize(gray)

    candidates = {"otsu": otsu, "adaptive_gaussian": adaptive, "sauvola": sauvola}
    _save_stage(otsu,     debug_dir, "06_binarized_otsu.png")
    _save_stage(adaptive, debug_dir, "06_binarized_adaptive_gaussian.png")
    _save_stage(sauvola,  debug_dir, "06_binarized_sauvola.png")

    scores = {name: _score_binary(img) for name, img in candidates.items()}
    best = max(scores, key=scores.get)
    log.debug("Binarization scores: %s → selected: %s", scores, best)
    _save_stage(candidates[best], debug_dir, "07_selected_binarization.png")
    return candidates[best], best, scores


# ── Stage: Deskew ─────────────────────────────────────────────────────────────

def _deskew(binary: np.ndarray) -> tuple:
    """
    Deskew only when a meaningful angle is detected.
    Returns (deskewed_binary, angle_applied).
    """
    coords = np.column_stack(np.where(binary == 0))
    if len(coords) < 100:
        return binary, 0.0

    angle = cv2.minAreaRect(coords.astype(np.float32))[2]
    if angle < -45:
        angle += 90
    angle = float(np.clip(angle, -C.deskew_max_angle, C.deskew_max_angle))

    if abs(angle) < C.deskew_min_angle:
        log.debug("Deskew skipped (angle=%.2f < min=%.2f)", angle, C.deskew_min_angle)
        return binary, 0.0

    h, w = binary.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(binary, M, (w, h),
                              flags=cv2.INTER_LINEAR, borderValue=255)
    log.debug("Deskew applied: %.2f deg", angle)
    return rotated, angle


# ── Stage: Stroke normalization ───────────────────────────────────────────────

def _normalize_stroke_width(binary: np.ndarray) -> np.ndarray:
    ink = (binary == 0).astype(np.uint8)
    if ink.sum() < 50:
        return binary
    dist = cv2.distanceTransform(ink, cv2.DIST_L2, 3)
    median_radius = float(np.median(dist[dist > 0])) if dist.max() > 0 else 1.0
    target_radius = max(1.0, min(median_radius, 2.5))
    skeleton = cv2.ximgproc.thinning(ink * 255) if hasattr(cv2, "ximgproc") else ink * 255
    r = max(1, int(round(target_radius)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    normalized_ink = cv2.dilate(skeleton, kernel, iterations=1)
    return np.where(normalized_ink > 0, 0, 255).astype(np.uint8)


# ── Stage: Morphological closing (minimal — never merges chars) ───────────────

def _dilate(binary: np.ndarray) -> np.ndarray:
    """Tiny closing to reconnect broken thin strokes only."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)


# ── Stage: Border removal (conservative) ─────────────────────────────────────

def _clear_edge_artifacts(binary: np.ndarray) -> np.ndarray:
    """
    Remove dense dark camera-shadow borders at page edges.
    Threshold raised to 0.60 so text near edges is never erased.
    """
    cleaned = binary.copy()
    h, w = cleaned.shape
    max_y_scan = max(1, int(h * 0.08))
    max_x_scan = max(1, int(w * 0.08))
    threshold = 0.60   # only clear rows/cols that are >60% ink (true shadow)

    def _clear_edge(axis, reverse=False):
        rng = range(max_y_scan) if axis == "y" else range(max_x_scan)
        if reverse:
            rng = reversed(list(rng))
        cut = None
        for i in rng:
            row = cleaned[i, :] if axis == "y" else cleaned[:, i]
            if np.mean(row == 0) < threshold:
                break
            cut = i
        if cut is not None:
            if axis == "y" and not reverse:
                cleaned[:cut + 1, :] = 255
            elif axis == "y" and reverse:
                cleaned[cut:, :] = 255
            elif axis == "x" and not reverse:
                cleaned[:, :cut + 1] = 255
            else:
                cleaned[:, cut:] = 255

    _clear_edge("y")
    _clear_edge("y", reverse=True)
    _clear_edge("x")
    _clear_edge("x", reverse=True)
    return cleaned


# ── Stage: Ruling-line removal (character-safe) ───────────────────────────────

def _remove_ruling_lines(binary: np.ndarray, debug_dir=None) -> np.ndarray:
    """
    Remove notebook ruling/margin lines while restoring character intersections.

    Strategy:
      1. Detect long horizontal/vertical lines via morphological opening.
      2. Build a mask of those lines.
      3. Dilate the NON-line ink slightly to recover character strokes that
         cross the line — then subtract only the pure-line pixels.
    """
    h, w = binary.shape
    ink = np.where(binary == 0, 255, 0).astype(np.uint8)

    horizontal_len = max(60, int(w * 0.30))
    vertical_len   = max(60, int(h * 0.30))
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_len, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_len))

    horizontal = cv2.morphologyEx(ink, cv2.MORPH_OPEN, h_kernel)
    vertical   = cv2.morphologyEx(ink, cv2.MORPH_OPEN, v_kernel)
    line_mask  = cv2.bitwise_or(horizontal, vertical)

    _save_stage(cv2.bitwise_not(line_mask), debug_dir, "13_line_mask.png")

    if line_mask.max() == 0:
        # No ruling lines detected
        return binary

    # Recover character pixels that overlap with lines:
    # dilate the non-line ink and use it to punch holes back into the mask
    non_line_ink = cv2.bitwise_and(ink, cv2.bitwise_not(line_mask))
    recover_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    recovered = cv2.dilate(non_line_ink, recover_kernel, iterations=1)
    safe_mask = cv2.bitwise_and(line_mask, cv2.bitwise_not(recovered))

    cleaned = binary.copy()
    cleaned[safe_mask > 0] = 255
    return cleaned


# ── DocumentPreprocessor (used by run_dan) ────────────────────────────────────

class DocumentPreprocessor:
    def __init__(self, n_skew_regions=6, fft_threshold_pct=99.5,
                 tophat_klen_ratio=0.25):
        self._n_regions    = n_skew_regions
        self._fft_pct      = fft_threshold_pct
        self._tophat_ratio = tophat_klen_ratio

    @staticmethod
    def _normalise_illumination(bgr):
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    def _remove_gridlines(self, gray):
        f = np.fft.fft2(gray.astype(np.float32))
        fshift = np.fft.fftshift(f)
        mag = np.abs(fshift)
        H, W = gray.shape
        cy, cx = H // 2, W // 2
        dc_mask = np.ones((H, W), dtype=bool)
        ys, xs = np.ogrid[:H, :W]
        dc_mask[(ys - cy) ** 2 + (xs - cx) ** 2 < 100] = False
        mag_masked = mag.copy()
        mag_masked[~dc_mask] = 0
        if mag_masked.max() > 0:
            thresh = np.percentile(mag_masked[mag_masked > 0], self._fft_pct)
            notch = np.ones((H, W), dtype=np.float32)
            notch[mag_masked > thresh] = 0.0
            img_back = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift * notch)))
            gray = np.clip(img_back, 0, 255).astype(np.uint8)
        inv = cv2.bitwise_not(gray)
        klen = max(20, int(W * self._tophat_ratio))
        kern = cv2.getStructuringElement(cv2.MORPH_RECT, (klen, 1))
        tophat = cv2.morphologyEx(inv, cv2.MORPH_TOPHAT, kern)
        return cv2.bitwise_not(cv2.subtract(inv, tophat))

    def process(self, source):
        bgr = load_image(source)
        _validate(bgr)
        bgr = _resize(bgr)
        bgr = self._normalise_illumination(bgr)
        gray = _grayscale(bgr)
        gray = self._remove_gridlines(gray)
        gray = _denoise(gray)
        binary, _, _ = _select_binarization(gray)
        binary, _ = _deskew(binary)
        binary = _dilate(binary)
        return binary, bgr


# ── Main preprocess() entry point ─────────────────────────────────────────────

def preprocess(
    source,
    normalize_strokes: bool = False,
    debug_dir=None,
) -> tuple:
    """
    Full preprocessing pipeline.

    Returns:
        binary  (HxW uint8)   — clean binary for detection
        bgr     (HxWx3 uint8) — resized colour image (fed to TrOCR)
        meta    (dict)        — stages run, binarization method, scores
    """
    meta = {"stages": [], "binarization_method": "otsu",
            "binarization_scores": {}, "deskew_angle": 0.0}

    # 01 raw
    if isinstance(source, str):
        raw = cv2.cvtColor(np.array(Image.open(source).convert("RGB")),
                           cv2.COLOR_RGB2BGR)
    elif isinstance(source, Image.Image):
        raw = cv2.cvtColor(np.array(source.convert("RGB")), cv2.COLOR_RGB2BGR)
    else:
        raw = source
    _save_stage(raw, debug_dir, "01_raw_image.png")

    bgr = load_image(source)
    _validate(bgr)
    _save_stage(bgr, debug_dir, "02_loaded_image.png")
    meta["stages"].append("load")

    bgr = _resize(bgr)
    _save_stage(bgr, debug_dir, "03_resized.png")
    meta["stages"].append("resize")

    gray = _grayscale(bgr)
    _save_stage(gray, debug_dir, "04_grayscale.png")
    meta["stages"].append("grayscale")

    gray = _clahe(gray)
    _save_stage(gray, debug_dir, "05_clahe.png")
    meta["stages"].append("clahe")

    gray = _denoise(gray)
    _save_stage(gray, debug_dir, "05b_denoised.png")
    meta["stages"].append("gaussian_denoising")

    # Multi-candidate binarization — all three saved, best selected
    binary, method, scores = _select_binarization(gray, debug_dir)
    meta["binarization_method"] = method
    meta["binarization_scores"] = scores
    meta["stages"].append("binarization")

    binary, angle = _deskew(binary)
    _save_stage(binary, debug_dir, "08_deskewed.png")
    meta["deskew_angle"] = angle
    meta["stages"].append("deskew")

    if normalize_strokes:
        binary = _normalize_stroke_width(binary)
        _save_stage(binary, debug_dir, "09_stroke_normalized.png")
        meta["stages"].append("stroke_normalization")

    binary = _dilate(binary)
    _save_stage(binary, debug_dir, "10_morphological_closing.png")
    meta["stages"].append("morphological_closing")

    binary = _clear_edge_artifacts(binary)
    _save_stage(binary, debug_dir, "11_borders_removed.png")
    meta["stages"].append("remove_borders")

    _save_stage(binary, debug_dir, "12_before_line_removal.png")
    binary = _remove_ruling_lines(binary, debug_dir)
    _save_stage(binary, debug_dir, "14_after_line_removal.png")
    meta["stages"].append("remove_ruling_lines")

    _save_stage(binary, debug_dir, "15_final_clean_binary.png")

    log.debug("Preprocessing done: %s  method=%s  angle=%.2f",
              binary.shape, method, angle)
    return binary, bgr, meta
