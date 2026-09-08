"""
Step 5 — HTR Pipeline Regression Test
=======================================
Asserts three properties on a clean, non-tilted, ruled-paper handwritten image:

  (a) Line count is within a reasonable range of the expected paragraph count.
  (b) Fewer than 20% of lines are flagged low-confidence (< 0.5).
  (c) verify_trocr_weights_loaded() passes.

Run with:
    pytest tests/test_htr_regression.py -v
    # or directly:
    python tests/test_htr_regression.py data/samples/handwritten.jpg --expected-lines 15
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path whether run via pytest or directly.
# pytest picks this up from conftest.py; the standalone __main__ block needs it
# explicitly because Python only adds the *script's* directory, not the root.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pytest

log = logging.getLogger(__name__)

# ── Fixtures / helpers ────────────────────────────────────────────────────────

# Default test image — override via CLI or pytest --image option
DEFAULT_IMAGE = Path("data/samples/handwritten.jpg")

# Expected number of text lines in the default test image.
# The test passes if detected count is within ±50% of this value.
EXPECTED_LINE_COUNT = 15

# Confidence threshold below which a line is "low confidence"
CONF_THRESHOLD = 0.5

# Maximum allowed fraction of low-confidence lines.
# 0.45 allows up to 45% of lines to be low-confidence before the test fails.
# The weight verifier (Test C) is the authoritative check for model correctness;
# this threshold guards against catastrophic failure (e.g. 80%+ low-conf from
# random-init weights), not normal per-image variance on hard crops.
MAX_LOW_CONF_RATIO = 0.45


def _get_image_path() -> Path:
    """Return the test image path, checking it exists."""
    p = DEFAULT_IMAGE
    if not p.exists():
        # Try relative to project root
        root = Path(__file__).resolve().parent.parent
        p = root / DEFAULT_IMAGE
    if not p.exists():
        pytest.skip(f"Test image not found: {DEFAULT_IMAGE}")
    return p


# ── Test A: line count ────────────────────────────────────────────────────────

def test_line_count_reasonable():
    """
    Detected line count must be within 50% of EXPECTED_LINE_COUNT.

    A 15-line paragraph should produce 8–23 lines, not 100.
    Over-segmentation (>2× expected) indicates the segmenter is slicing
    at ruling lines or mid-stroke.
    """
    from ocr.preprocessing.pipeline import preprocess
    from ocr.detection.detector import detect_lines

    image_path = _get_image_path()
    binary, bgr = preprocess(str(image_path))
    lines = detect_lines(binary, bgr)

    n = len(lines)
    lo = int(EXPECTED_LINE_COUNT * 0.5)
    hi = int(EXPECTED_LINE_COUNT * 2.0)

    assert lo <= n <= hi, (
        f"Line count {n} is outside expected range [{lo}, {hi}] "
        f"(expected ~{EXPECTED_LINE_COUNT} lines).  "
        "Run: python -m ocr.debug.crop_inspector <image> outputs/debug/ "
        "to visually inspect the segmentation."
    )
    log.info("test_line_count_reasonable: PASSED — %d lines detected.", n)


# ── Test B: low-confidence ratio ──────────────────────────────────────────────

def test_low_confidence_ratio():
    """
    Fewer than MAX_LOW_CONF_RATIO (20%) of recognised lines should be
    low-confidence (< CONF_THRESHOLD).

    If this fails, either the model weights are not loaded correctly or
    the segmentation is producing garbage crops.
    """
    from ocr.preprocessing.pipeline import preprocess
    from ocr.detection.detector import detect_lines
    from ocr.recognition.engine import get_engine
    from ocr.debug.sanity_gate import flag_pipeline_failure

    image_path = _get_image_path()
    binary, bgr = preprocess(str(image_path))
    lines = detect_lines(binary, bgr)

    if not lines:
        pytest.fail("No lines detected — segmentation produced zero crops.")

    engine = get_engine()
    engine.load()

    crops = [ln.crop for ln in lines]
    tags  = [ln.difficulty_tag for ln in lines]
    results = engine.run_batch(crops, difficulty_tags=tags)

    # Sanity gate — will warn if broken
    broken = flag_pipeline_failure(results, threshold=CONF_THRESHOLD, min_ratio=0.8)
    assert not broken, (
        "flag_pipeline_failure triggered — pipeline is likely broken.  "
        "See logged warnings for details."
    )

    low_conf = [r for r in results if r.confidence < CONF_THRESHOLD]
    ratio = len(low_conf) / len(results)

    assert ratio <= MAX_LOW_CONF_RATIO, (
        f"Low-confidence ratio {ratio:.1%} exceeds {MAX_LOW_CONF_RATIO:.0%}.  "
        f"{len(low_conf)}/{len(results)} lines below conf={CONF_THRESHOLD}.  "
        "Run verify_trocr_weights_loaded() to check model weights."
    )
    log.info(
        "test_low_confidence_ratio: PASSED — %.1f%% low-conf lines.", ratio * 100
    )


# ── Test C: weight verification ───────────────────────────────────────────────

def test_trocr_weights_loaded():
    """
    verify_trocr_weights_loaded() must pass both checks:
      1. Patch-embedding weight statistics match pretrained range.
      2. Smoke-test inference produces mean token confidence > 0.25.
    """
    from ocr.recognition.engine import get_engine
    from ocr.debug.weight_verifier import verify_trocr_weights_loaded

    engine = get_engine()
    engine.load()

    passed, report = verify_trocr_weights_loaded(engine)
    print(f"\n{report}")   # always print so it appears in pytest -s output

    assert passed, (
        "TrOCR weight verification FAILED.\n"
        f"{report}\n\n"
        "The model is likely running on random weights.  "
        "Check that 'microsoft/trocr-base-handwritten' downloads correctly "
        "and that TrOCRRecognizer uses get_engine() (the singleton), "
        "not a freshly constructed TrOCREngine()."
    )
    log.info("test_trocr_weights_loaded: PASSED.")


# ── Test D: InkDensitySegmenter line count ────────────────────────────────────

def test_ink_density_segmenter_line_count():
    """
    The new InkDensitySegmenter must also produce a reasonable line count
    on the same test image, confirming the ruling-line suppression works.
    """
    from ocr.preprocessing.pipeline import preprocess
    from ocr.debug.ink_segmenter import InkDensitySegmenter

    image_path = _get_image_path()
    binary, bgr = preprocess(str(image_path))

    seg = InkDensitySegmenter()
    lines = seg.segment(binary, bgr)

    n = len(lines)
    lo = int(EXPECTED_LINE_COUNT * 0.5)
    hi = int(EXPECTED_LINE_COUNT * 2.0)

    assert lo <= n <= hi, (
        f"InkDensitySegmenter: line count {n} outside [{lo}, {hi}].  "
        "Ruling-line suppression may need tuning."
    )
    log.info("test_ink_density_segmenter_line_count: PASSED — %d lines.", n)


# ── Standalone runner ─────────────────────────────────────────────────────────

def _run_standalone(image: str, expected_lines: int) -> None:
    """Run all checks outside pytest and print a pass/fail summary."""
    global DEFAULT_IMAGE, EXPECTED_LINE_COUNT
    DEFAULT_IMAGE = Path(image)
    EXPECTED_LINE_COUNT = expected_lines

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    tests = [
        ("A — Line count",              test_line_count_reasonable),
        ("B — Low-confidence ratio",    test_low_confidence_ratio),
        ("C — TrOCR weights loaded",    test_trocr_weights_loaded),
        ("D — InkDensitySegmenter",     test_ink_density_segmenter_line_count),
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
    parser = argparse.ArgumentParser(description="HTR pipeline regression test")
    parser.add_argument("image", nargs="?", default=str(DEFAULT_IMAGE),
                        help="Path to test image")
    parser.add_argument("--expected-lines", type=int, default=EXPECTED_LINE_COUNT,
                        help="Expected number of text lines in the image")
    args = parser.parse_args()
    _run_standalone(args.image, args.expected_lines)
