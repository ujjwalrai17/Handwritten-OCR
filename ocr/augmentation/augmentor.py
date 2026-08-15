"""
Augmentation Module for Cursive Handwriting
Generates synthetic training samples that simulate the specific failure modes
of messy, fast cursive writing:

  1. Elastic distortion      — warps strokes to simulate natural writing variation
  2. Stroke jitter           — per-pixel noise on ink pixels
  3. Line-bleed simulation   — copies ascender/descender pixels from adjacent lines
  4. Random slant            — shear transform for forward/backward lean
  5. Stroke width variation  — dilation/erosion to vary pen thickness
  6. Baseline drift          — sinusoidal vertical shift of text baseline

Usage:
    from ocr.augmentation.augmentor import augment_line, augment_page
    aug_img = augment_line(binary_crop)
    aug_page = augment_page(binary_page, line_boxes)
"""

import cv2
import numpy as np
from config.settings import cfg
from ocr.utils.logger import get_logger

log = get_logger(__name__)
A = cfg.augmentation


# ── Elastic Distortion ────────────────────────────────────────────────────────

def elastic_distort(image: np.ndarray, alpha: float = None, sigma: float = None) -> np.ndarray:
    """
    Elastic deformation (Simard et al. 2003).
    Simulates natural handwriting stroke variation and connected ligatures.
    """
    alpha = alpha or A.elastic_alpha
    sigma = sigma or A.elastic_sigma
    h, w = image.shape[:2]

    rng = np.random.default_rng()
    dx = cv2.GaussianBlur(rng.uniform(-1, 1, (h, w)).astype(np.float32), (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur(rng.uniform(-1, 1, (h, w)).astype(np.float32), (0, 0), sigma) * alpha

    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = np.clip(x + dx, 0, w - 1).astype(np.float32)
    map_y = np.clip(y + dy, 0, h - 1).astype(np.float32)

    return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderValue=255)


# ── Stroke Jitter ─────────────────────────────────────────────────────────────

def stroke_jitter(binary: np.ndarray, sigma: float = None) -> np.ndarray:
    """Add Gaussian noise to ink pixels to simulate pen pressure variation."""
    sigma = sigma or A.stroke_jitter_sigma
    noise = np.random.normal(0, sigma * 20, binary.shape).astype(np.int16)
    jittered = np.clip(binary.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    # Re-binarize
    _, result = cv2.threshold(jittered, 127, 255, cv2.THRESH_BINARY)
    return result


# ── Line Bleed Simulation ─────────────────────────────────────────────────────

def simulate_line_bleed(
    page: np.ndarray,
    line_boxes: list[tuple],
    prob: float = None,
    max_shift: int = None,
) -> np.ndarray:
    """
    Simulate ascender/descender bleed between adjacent lines.
    For each pair of adjacent lines, with probability `prob`, copy a strip
    of ink from the bottom of line[i] into the top of line[i+1] (and vice versa).

    Args:
        page:       full-page binary image
        line_boxes: list of (x1,y1,x2,y2) line bounding boxes, sorted top-to-bottom
        prob:       probability of applying bleed to each adjacent pair
        max_shift:  maximum pixel shift for the bleed strip
    Returns:
        augmented page image
    """
    prob = prob or A.line_bleed_prob
    max_shift = max_shift or A.line_bleed_max_shift
    if len(line_boxes) < 2:
        return page

    result = page.copy()
    rng = np.random.default_rng()

    for i in range(len(line_boxes) - 1):
        if rng.random() > prob:
            continue
        _, y1_top, _, y2_top = line_boxes[i]
        _, y1_bot, _, y2_bot = line_boxes[i + 1]

        shift = rng.integers(1, max_shift + 1)

        # Bleed bottom strip of upper line into top of lower line
        strip_h = min(shift, y2_top - y1_top, y2_bot - y1_bot)
        if strip_h < 1:
            continue

        upper_strip = page[y2_top - strip_h: y2_top, :]
        # Overlay: ink pixels (0) from upper strip bleed into lower line top
        target_y1 = y1_bot
        target_y2 = min(y1_bot + strip_h, result.shape[0])
        bleed_h = target_y2 - target_y1
        result[target_y1:target_y2, :] = np.minimum(
            result[target_y1:target_y2, :],
            upper_strip[:bleed_h, :]
        )

    return result


# ── Random Slant ──────────────────────────────────────────────────────────────

def random_slant(binary: np.ndarray, angle_range: tuple = None) -> np.ndarray:
    """Apply random shear (slant) to simulate forward/backward cursive lean."""
    angle_range = angle_range or A.slant_range
    angle = float(np.random.uniform(*angle_range))
    h, w = binary.shape
    shear = np.tan(np.radians(angle))
    M = np.float32([[1, shear, 0], [0, 1, 0]])
    new_w = int(w + abs(shear) * h)
    result = cv2.warpAffine(binary, M, (new_w, h), borderValue=255)
    # Crop back to original width
    if result.shape[1] > w:
        start = (result.shape[1] - w) // 2
        result = result[:, start: start + w]
    return result


# ── Stroke Width Variation ────────────────────────────────────────────────────

def vary_stroke_width(binary: np.ndarray, scale_range: tuple = None) -> np.ndarray:
    """Randomly dilate or erode strokes to simulate different pen widths."""
    scale_range = scale_range or A.stroke_width_range
    scale = float(np.random.uniform(*scale_range))
    r = max(1, int(round(scale)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r, r))
    if scale >= 1.0:
        return cv2.dilate(binary, kernel, iterations=1)
    else:
        return cv2.erode(binary, kernel, iterations=1)


# ── Baseline Drift ────────────────────────────────────────────────────────────

def baseline_drift(binary: np.ndarray, amplitude: int = 3, frequency: float = 0.02) -> np.ndarray:
    """
    Apply sinusoidal vertical shift column-by-column to simulate baseline drift.
    Mimics the natural up-and-down movement of fast handwriting.
    """
    h, w = binary.shape
    result = np.full_like(binary, 255)
    phase = np.random.uniform(0, 2 * np.pi)
    for x in range(w):
        shift = int(amplitude * np.sin(2 * np.pi * frequency * x + phase))
        src_y = np.arange(h)
        dst_y = np.clip(src_y + shift, 0, h - 1)
        result[dst_y, x] = binary[src_y, x]
    return result


# ── Composite Augmentation ────────────────────────────────────────────────────

def augment_line(
    binary: np.ndarray,
    elastic: bool = True,
    jitter: bool = True,
    slant: bool = True,
    stroke_width: bool = True,
    drift: bool = True,
) -> np.ndarray:
    """
    Apply a random subset of augmentations to a single line crop.
    Each transform is applied with 50% probability to create variety.
    """
    if not A.enabled:
        return binary

    rng = np.random.default_rng()

    if elastic and rng.random() < 0.5:
        binary = elastic_distort(binary)
    if slant and rng.random() < 0.5:
        binary = random_slant(binary)
    if stroke_width and rng.random() < 0.5:
        binary = vary_stroke_width(binary)
    if drift and rng.random() < 0.4:
        binary = baseline_drift(binary)
    if jitter and rng.random() < 0.3:
        binary = stroke_jitter(binary)

    return binary


def augment_page(
    page: np.ndarray,
    line_boxes: list[tuple],
    n_variants: int = 1,
) -> list[np.ndarray]:
    """
    Generate `n_variants` augmented versions of a full page.
    Applies line-bleed simulation at the page level (requires line context),
    then per-line augmentations.

    Returns list of augmented page images.
    """
    variants = []
    for _ in range(n_variants):
        aug = simulate_line_bleed(page, line_boxes)
        aug = elastic_distort(aug, alpha=A.elastic_alpha * 0.5, sigma=A.elastic_sigma)
        variants.append(aug)
    return variants
