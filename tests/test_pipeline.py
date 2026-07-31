"""
Unit Tests — covers all core modules
Run with: pytest tests/ -v
"""

import numpy as np
import pytest
from PIL import Image

from ocr.preprocessing.pipeline import preprocess, PreprocessingError
from ocr.postprocessing.corrector import (
    correct_text, postprocess, keyword_search, DocumentResult
)
from ocr.recognition.engine import LineResult, TrOCREngine
from ocr.evaluation.metrics import compute_cer, compute_wer, evaluate
from ocr.utils.output_writer import save_text, save_confidence_csv


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def text_image():
    img = np.ones((100, 300, 3), dtype=np.uint8) * 255
    img[30:70, 50:250] = 0
    return Image.fromarray(img)


@pytest.fixture
def tiny_image():
    return Image.fromarray(np.ones((5, 5, 3), dtype=np.uint8) * 255)


@pytest.fixture
def sample_line_results():
    return [
        LineResult(text="hello world", confidence=0.92, token_confidences=[0.92, 0.91]),
        LineResult(text="the quick brown fox", confidence=0.80, token_confidences=[0.80]),
        LineResult(text="low conf line", confidence=0.50, token_confidences=[0.50]),
        LineResult(text="", confidence=0.0, token_confidences=[]),
    ]


# ── Preprocessing ─────────────────────────────────────────────────────────────

def test_preprocess_returns_correct_shapes(text_image):
    binary, bgr = preprocess(text_image)
    assert binary.ndim == 2
    assert bgr.ndim == 3
    assert binary.shape[:2] == bgr.shape[:2]


def test_preprocess_binary_values(text_image):
    binary, _ = preprocess(text_image)
    assert set(np.unique(binary)).issubset({0, 255})


def test_preprocess_rejects_tiny_image(tiny_image):
    with pytest.raises(PreprocessingError):
        preprocess(tiny_image)


def test_preprocess_rejects_bad_path():
    with pytest.raises(PreprocessingError):
        preprocess("does_not_exist.jpg")


def test_preprocess_resizes_large_image():
    large = Image.fromarray(np.ones((3000, 4000, 3), dtype=np.uint8) * 200)
    binary, bgr = preprocess(large)
    assert max(binary.shape) <= 2048


# ── Post-processing ───────────────────────────────────────────────────────────

def test_correct_text_preserves_proper_noun():
    assert correct_text("London is great") == correct_text("London is great")
    assert "London" in correct_text("London is great")


def test_correct_text_skips_numbers():
    result = correct_text("2024 was a year")
    assert "2024" in result


def test_postprocess_filters_empty_lines(sample_line_results):
    doc = postprocess(sample_line_results)
    assert all(ln.raw_text.strip() for ln in doc.lines)


def test_postprocess_confidence_flags(sample_line_results):
    doc = postprocess(sample_line_results)
    for ln in doc.lines:
        if ln.confidence >= 0.75:
            assert not ln.is_low_confidence
        else:
            assert ln.is_low_confidence


def test_document_full_text(sample_line_results):
    doc = postprocess(sample_line_results)
    assert isinstance(doc.full_text, str)
    assert len(doc.full_text) > 0


def test_document_mean_confidence(sample_line_results):
    doc = postprocess(sample_line_results)
    assert 0.0 <= doc.mean_confidence <= 1.0


def test_keyword_search_finds_match(sample_line_results):
    doc = postprocess(sample_line_results)
    matches = keyword_search(doc, "hello")
    assert len(matches) >= 1
    assert matches[0]["line_index"] == 0


def test_keyword_search_case_insensitive(sample_line_results):
    doc = postprocess(sample_line_results)
    assert keyword_search(doc, "HELLO") == keyword_search(doc, "hello")


def test_keyword_search_empty_returns_empty(sample_line_results):
    doc = postprocess(sample_line_results)
    assert keyword_search(doc, "") == []


def test_keyword_search_no_match(sample_line_results):
    doc = postprocess(sample_line_results)
    assert keyword_search(doc, "zzznomatch") == []


# ── Evaluation Metrics ────────────────────────────────────────────────────────

def test_cer_perfect():
    assert compute_cer(["hello"], ["hello"]) == 0.0


def test_cer_completely_wrong():
    cer = compute_cer(["abc"], ["xyz"])
    assert cer == 1.0


def test_wer_perfect():
    assert compute_wer(["hello world"], ["hello world"]) == 0.0


def test_wer_one_word_wrong():
    wer = compute_wer(["hello world"], ["hello earth"])
    assert 0.0 < wer <= 1.0


def test_evaluate_returns_metric_result(sample_line_results):
    doc = postprocess(sample_line_results)
    preds = [ln.corrected_text for ln in doc.lines]
    refs = preds[:]   # perfect prediction
    metrics = evaluate(preds, refs, inference_time_sec=1.5)
    assert metrics.cer == 0.0
    assert metrics.wer == 0.0
    assert metrics.accuracy == 1.0


def test_evaluate_empty_lists():
    metrics = evaluate([], [], inference_time_sec=0.0)
    assert metrics.num_samples == 0


# ── Recognition Engine (no model load) ───────────────────────────────────────

def test_engine_valid_crop():
    engine = TrOCREngine()
    assert engine._is_valid(np.ones((32, 200), dtype=np.uint8))


def test_engine_rejects_tiny_crop():
    engine = TrOCREngine()
    assert not engine._is_valid(np.ones((4, 4), dtype=np.uint8))


def test_engine_rejects_none():
    engine = TrOCREngine()
    assert not engine._is_valid(None)


# ── Output Writer ─────────────────────────────────────────────────────────────

def test_save_text_creates_file(tmp_path, sample_line_results):
    doc = postprocess(sample_line_results)
    out = tmp_path / "out.txt"
    save_text(doc, out)
    assert out.exists()
    assert out.read_text(encoding="utf-8") == doc.full_text


def test_save_confidence_csv_creates_file(tmp_path, sample_line_results):
    doc = postprocess(sample_line_results)
    out = tmp_path / "conf.csv"
    save_confidence_csv(doc, out)
    assert out.exists()
    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("line_id")
    assert len(lines) == len(doc.lines) + 1   # header + data rows
