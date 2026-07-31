"""
OCR Pipeline Orchestrator
Connects all stages: preprocess → detect → recognize → postprocess
Used by main.py, predict.py, and train.py (for evaluation)
"""

import time
from pathlib import Path
from PIL import Image

from ocr.preprocessing.pipeline import preprocess, PreprocessingError
from ocr.detection.detector import detect_lines
from ocr.recognition.engine import get_engine
from ocr.postprocessing.corrector import postprocess, DocumentResult
from ocr.utils.logger import get_logger

log = get_logger(__name__)


def run(source, source_path: str = "") -> tuple[DocumentResult, float]:
    """
    Run the full OCR pipeline on a single image.

    Args:
        source:      str (file path) or PIL.Image
        source_path: display label for output (original filename)

    Returns:
        (DocumentResult, inference_time_seconds)
    """
    t0 = time.perf_counter()

    try:
        binary, bgr = preprocess(source)
    except PreprocessingError as e:
        log.error("Preprocessing failed: %s", e)
        return DocumentResult(source_path=source_path), 0.0

    lines = detect_lines(binary, bgr)
    if not lines:
        log.warning("No text lines detected in: %s", source_path)
        return DocumentResult(source_path=source_path), 0.0

    engine = get_engine()
    line_results = engine.run_batch([ln.crop for ln in lines])

    doc = postprocess(line_results, source_path=source_path)

    elapsed = time.perf_counter() - t0
    log.info("Pipeline done in %.2fs — %d lines recognized", elapsed, len(doc.lines))
    return doc, elapsed


def run_folder(folder: str) -> list[tuple[DocumentResult, float]]:
    """Run pipeline on all images in a folder. Returns list of (doc, time)."""
    exts = {".jpg", ".jpeg", ".png", ".tiff", ".bmp"}
    paths = [p for p in Path(folder).iterdir() if p.suffix.lower() in exts]
    paths.sort()
    log.info("Found %d images in '%s'", len(paths), folder)
    return [run(str(p), source_path=str(p)) for p in paths]


def run_pdf(pdf_path: str) -> list[tuple[DocumentResult, float]]:
    """Convert PDF pages to images and run pipeline on each page."""
    from ocr.utils.pdf_reader import pdf_to_images
    pages = pdf_to_images(pdf_path)
    log.info("Processing %d pages from '%s'", len(pages), pdf_path)
    return [run(page, source_path=f"{pdf_path} [page {i+1}]")
            for i, page in enumerate(pages)]
