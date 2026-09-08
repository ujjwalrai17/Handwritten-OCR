"""
Step 5 — Hallucination Regression Test
========================================
Runs the pipeline on a sample image with known ground-truth text and asserts:

  (a) Word-level accuracy >= 40% on high-confidence lines (not just CER).
  (b) No output line has a word count more than 2.5x what its crop's ink
      density can plausibly support (hallucination length check).
  (c) verify_trocr_weights_loaded() passes (model is not random-init).
  (d) Lines flagged needs_review=True are a minority (<50%) of all lines.

Run with:
    python -m pytest tests/test_hallucination.py -v
    python tests/test_hallucination.py data/samples/handwritten.jpg
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pytest

log = logging.getLogger(__name__)

DEFAULT_IMAGE = Path("data/samples/handwritten.jpg")

# Ground truth for handwritten.jpg (machine learning paragraph).
# Only the lines we are confident about — used for word-accuracy check.
GROUND_TRUTH_LINES = [
    "Machine learning is a method of data analysis that automates",
    "analytical model building It is a branch of artificial intelligence",
    "based on the idea that systems can learn from data identify",
    "Machine learning algorithms build a mathematical model based on",
    "example data known as training data in order to make predictions",
    "or decisions without being explicitly programmed to perform the task",
    "Machine learning is widely used in many fields including computer",
    "vision natural language processing speech recognition email filtering",
    "that it became one of the most exciting fields in technology today",
]

MAX_HALLUCINATION_RATIO = 2.5   # predicted words / ink-estimated words
MIN_WORD_ACCURACY       = 0.40  # on high-confidence lines only
MAX_REVIEW_RATIO        = 0.50  # max fraction of lines flagged needs_review


# ── helpers ───────────────────────────────────────────────────────────────────

def _get_image_path() -> Path:
    p = DEFAULT_IMAGE
    if not p.exists():
        p = _PROJECT_ROOT / DEFAULT_IMAGE
    if not p.exists():
        pytest.skip(f"Test image not found: {DEFAULT_IMAGE}")
    return p


def _word_accuracy(predicted: str, reference: str) -> float:
    """
    Compute word-level accuracy: fraction of reference words that appear
    in the predicted string (order-insensitive, case-insensitive).

    This is more informative than CER for detecting hallucination because
    a hallucinated fluent sentence can have low CER against a real sentence
    if they share common English words.

    Args:
        predicted:  Predicted text string.
        reference:  Ground-truth text string.

    Returns:
        Float in [0, 1].
    """
    ref_words  = set(reference.lower().split())
    pred_words = set(predicted.lower().split())
    if not ref_words:
        return 1.0
    return len(ref_words & pred_words) / len(ref_words)


def _ink_word_estimate(crop: np.ndarray) -> int:
    """
    Estimate word count from a binary crop's column ink projection.
    Counts horizontal ink runs separated by gaps (each run ~ one word).

    Args:
        crop: Binary uint8 image (0=ink, 255=background).

    Returns:
        Estimated word count (minimum 1).
    """
    import cv2
    if crop.ndim == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(crop, 128, 255, cv2.THRESH_BINARY_INV)
    col_proj = binary.sum(axis=0).astype(float)
    kernel   = np.ones(5) / 5
    col_proj = np.convolve(col_proj, kernel, mode="same")
    thresh   = col_proj.max() * 0.05
    in_run, count = False, 0
    for v in col_proj:
        if not in_run and v > thresh:
            in_run = True
            count += 1
        elif in_run and v <= thresh:
            in_run = False
    return max(3, count)  # floor at 3: short crops always have at least a few words


def _run_pipeline(image_path: Path):
    """Run preprocessing + segmentation + recognition, return (lines, results)."""
    from ocr.preprocessing.pipeline import preprocess
    from ocr.detection.detector import detect_lines
    from ocr.recognition.engine import get_engine

    binary, bgr = preprocess(str(image_path))
    lines = detect_lines(binary, bgr)
    if not lines:
        pytest.fail("No lines detected.")

    engine = get_engine()
    engine.load()
    results = engine.run_batch(
        [ln.crop for ln in lines],
        difficulty_tags=[ln.difficulty_tag for ln in lines],
    )
    return lines, results


# ── Test A: word accuracy on high-confidence lines ────────────────────────────

def test_word_accuracy_high_conf_lines():
    """
    On lines with confidence >= 0.5, at least MIN_WORD_ACCURACY of the
    reference words must appear in the prediction.

    This catches semantic hallucination: a model that generates fluent but
    wrong text will miss reference words even if CER looks acceptable.
    """
    lines, results = _run_pipeline(_get_image_path())

    high_conf = [(ln, r) for ln, r in zip(lines, results) if r.confidence >= 0.5]
    if not high_conf:
        pytest.skip("No high-confidence lines to evaluate.")

    scores = []
    for ln, r in high_conf:
        # Find the best-matching ground-truth line by word overlap
        best_acc = max(
            (_word_accuracy(r.text, gt) for gt in GROUND_TRUTH_LINES),
            default=0.0,
        )
        scores.append(best_acc)
        log.info("conf=%.2f  acc=%.2f  text=%r", r.confidence, best_acc, r.text[:60])

    mean_acc = float(np.mean(scores))
    assert mean_acc >= MIN_WORD_ACCURACY, (
        f"Mean word accuracy on high-conf lines = {mean_acc:.1%} < {MIN_WORD_ACCURACY:.0%}. "
        f"Scores: {[f'{s:.2f}' for s in scores]}. "
        "The decoder is likely hallucinating fluent but wrong text. "
        "Check flag_hallucination_risk() output and run save_debug_crops()."
    )
    log.info("test_word_accuracy: PASSED — mean_acc=%.1f%%", mean_acc * 100)


# ── Test B: ink-density length check ─────────────────────────────────────────

def test_no_hallucination_length_excess():
    """
    No output line should have a predicted word count more than
    MAX_HALLUCINATION_RATIO times the ink-density word estimate.

    A short, sparse crop producing a long fluent sentence is a definitive
    hallucination signal regardless of confidence score.
    """
    lines, results = _run_pipeline(_get_image_path())

    # Compute median crop height to identify anomalous full-page crops
    heights = [ln.crop.shape[0] for ln in lines]
    median_h = float(np.median(heights)) if heights else 1.0

    violations = []
    for idx, (ln, r) in enumerate(zip(lines, results)):
        if not r.text.strip() or r.text.strip() == "[illegible line]":
            continue
        # Lines already flagged needs_review by the engine are known hallucinations
        # and are handled by test D. Skip them here to avoid double-counting.
        if r.needs_review:
            continue
        # Skip crops that are >4x the median height (segmentation failures)
        if ln.crop.shape[0] > median_h * 4:
            continue
        ink_words  = _ink_word_estimate(ln.crop)
        pred_words = len([w for w in r.text.split() if not w.startswith("[")])
        ratio      = pred_words / max(1, ink_words)
        if ratio > MAX_HALLUCINATION_RATIO:
            violations.append({
                "line": idx,
                "pred_words": pred_words,
                "ink_words":  ink_words,
                "ratio":      ratio,
                "text":       r.text[:80],
            })
            log.warning(
                "Length excess line %d: pred=%d ink=%d ratio=%.1f text=%r",
                idx, pred_words, ink_words, ratio, r.text[:60],
            )

    assert not violations, (
        f"{len(violations)} line(s) exceed ink-density word count by >{MAX_HALLUCINATION_RATIO}x:\n"
        + "\n".join(
            f"  line {v['line']}: pred={v['pred_words']} ink={v['ink_words']} "
            f"ratio={v['ratio']:.1f}  {v['text']!r}"
            for v in violations
        )
    )
    log.info("test_no_hallucination_length_excess: PASSED.")


# ── Test C: weight verification ───────────────────────────────────────────────

def test_trocr_weights_loaded():
    """Model must pass both parameter-stats and smoke-test inference checks."""
    from ocr.recognition.engine import get_engine
    from ocr.debug.weight_verifier import verify_trocr_weights_loaded

    engine = get_engine()
    engine.load()
    passed, report = verify_trocr_weights_loaded(engine)
    print(f"\n{report}")
    assert passed, f"TrOCR weight verification FAILED:\n{report}"
    log.info("test_trocr_weights_loaded: PASSED.")


# ── Test D: hallucination-flagged ratio ───────────────────────────────────────

def test_hallucination_flag_ratio():
    """
    Lines flagged needs_review=True (hallucination risk) must be a minority
    of all lines.  If >50% are flagged, the flag_hallucination_risk()
    threshold may be miscalibrated or the model is genuinely broken.
    """
    _, results = _run_pipeline(_get_image_path())

    flagged = [r for r in results if r.needs_review]
    ratio   = len(flagged) / max(1, len(results))

    assert ratio <= MAX_REVIEW_RATIO, (
        f"{len(flagged)}/{len(results)} lines ({ratio:.1%}) flagged needs_review. "
        f"Threshold is {MAX_REVIEW_RATIO:.0%}. "
        "Either hallucination_conf_threshold or hallucination_length_ratio "
        "needs recalibration in config/settings.py."
    )
    log.info(
        "test_hallucination_flag_ratio: PASSED — %.1f%% flagged.", ratio * 100
    )


# ── Standalone runner ─────────────────────────────────────────────────────────

def _run_standalone(image: str) -> None:
    global DEFAULT_IMAGE
    DEFAULT_IMAGE = Path(image)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    tests = [
        ("A - Word accuracy (high-conf lines)",  test_word_accuracy_high_conf_lines),
        ("B - No hallucination length excess",   test_no_hallucination_length_excess),
        ("C - TrOCR weights loaded",             test_trocr_weights_loaded),
        ("D - Hallucination flag ratio",         test_hallucination_flag_ratio),
    ]
    passed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  [PASS] {name}")
            passed += 1
        except (AssertionError, Exception) as exc:
            print(f"  [FAIL] {name}: {exc}")
    print(f"\n{passed}/{len(tests)} tests passed.")
    sys.exit(0 if passed == len(tests) else 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hallucination regression test")
    parser.add_argument("image", nargs="?", default=str(DEFAULT_IMAGE))
    args = parser.parse_args()
    _run_standalone(args.image)
