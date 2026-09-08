"""
OCR Pipeline Orchestrator
Connects all stages: preprocess → detect → recognize → postprocess
Used by main.py, predict.py, and train.py (for evaluation)
"""

import time
from pathlib import Path
from PIL import Image

from config.settings import cfg
from ocr.preprocessing.pipeline import preprocess, PreprocessingError
from ocr.detection.detector import detect_lines, detect_words_hierarchical
from ocr.recognition.engine import get_engine
from ocr.postprocessing.corrector import postprocess, DocumentResult
from ocr.utils.logger import get_logger

log = get_logger(__name__)


def run(source, source_path: str = "", normalize_strokes: bool = False) -> tuple[DocumentResult, float]:
    """
    Run the full OCR pipeline on a single image.

    Args:
        source:            str (file path) or PIL.Image
        source_path:       display label for output (original filename)
        normalize_strokes: enable stroke-width normalization (slower, helps
                           with variable-width cursive and overlapping strokes)

    Returns:
        (DocumentResult, inference_time_seconds)
    """
    t0 = time.perf_counter()

    try:
        binary, bgr, _prep_meta = preprocess(source, normalize_strokes=normalize_strokes)
    except PreprocessingError as e:
        log.error("Preprocessing failed: %s", e)
        return DocumentResult(source_path=source_path), 0.0

    lines = detect_lines(binary, bgr)
    if not lines:
        log.warning("No text lines detected in: %s", source_path)
        return DocumentResult(source_path=source_path), 0.0

    engine = get_engine()
    difficulty_tags = [ln.difficulty_tag for ln in lines]
    line_results = engine.run_batch([ln.crop for ln in lines],
                                    difficulty_tags=difficulty_tags)

    doc = postprocess(line_results, source_path=source_path)

    elapsed = time.perf_counter() - t0
    log.info("Pipeline done in %.2fs — %d lines recognized", elapsed, len(doc.lines))
    return doc, elapsed


def run_hierarchical(
    source,
    source_path: str = "",
    normalize_strokes: bool = False,
) -> tuple[DocumentResult, float]:
    """
    Run the hierarchical word-and-line OCR pipeline on a single image.

    Detects individual word bounding boxes, groups them into lines via
    spatial Y-centroid clustering, then feeds each line strip to TrOCR.
    More robust than full-line extraction for tilted / overlapping text.

    Args:
        source:            str (file path) or PIL.Image.
        source_path:       display label for output.
        normalize_strokes: enable stroke-width normalization.

    Returns:
        (DocumentResult, inference_time_seconds)
    """
    from PIL import Image as _Image
    t0 = time.perf_counter()

    try:
        binary, bgr, _prep_meta = preprocess(source, normalize_strokes=normalize_strokes)
    except PreprocessingError as e:
        log.error("Preprocessing failed: %s", e)
        return DocumentResult(source_path=source_path), 0.0

    # Build PIL image for WordDetector (needs colour)
    if isinstance(source, str):
        pil_img = _Image.open(source).convert("RGB")
    elif isinstance(source, _Image.Image):
        pil_img = source.convert("RGB")
    else:
        import cv2 as _cv2
        import numpy as _np
        pil_img = _Image.fromarray(_cv2.cvtColor(bgr, _cv2.COLOR_BGR2RGB))

    lines = detect_words_hierarchical(bgr, pil_img)
    if not lines:
        log.warning("Hierarchical: no lines found, falling back to standard detect.")
        lines = detect_lines(binary, bgr)
    if not lines:
        log.warning("No text lines detected in: %s", source_path)
        return DocumentResult(source_path=source_path), 0.0

    engine = get_engine()
    difficulty_tags = [ln.difficulty_tag for ln in lines]
    line_results = engine.run_batch(
        [ln.crop for ln in lines], difficulty_tags=difficulty_tags
    )

    doc     = postprocess(line_results, source_path=source_path)
    elapsed = time.perf_counter() - t0
    log.info(
        "Hierarchical pipeline done in %.2fs — %d lines recognized",
        elapsed, len(doc.lines),
    )
    return doc, elapsed


def run_dan(
    source,
    source_path: str = "",
    confidence_threshold: float = 0.6,
    dan_weights: str = None,
    maskrcnn_weights: str = None,
    save_txt: bool = True,
    save_pdf: bool = True,
    save_csv: bool = True,
) -> tuple[DocumentResult, float]:
    """
    Run the DAN-style segmentation-free HTR pipeline and save outputs.

    Phase 1: DocumentPreprocessor (illumination + gridline removal + per-region skew).
    Phase 2: FullPageHTR (DAN CNN+Transformer full-page recognizer).
    Phase 3: RobustLineSegmenter (Mask-RCNN fallback if confidence is low).
    Phase 4: HTRRouter (confidence-based routing between phases 2 and 3).

    Args:
        source:               str file path or PIL.Image.
        source_path:          display label for output.
        confidence_threshold: Route to line fallback below this confidence.
        dan_weights:          Path to FullPageHTR checkpoint or None.
        maskrcnn_weights:     Path to RobustLineSegmenter checkpoint or None.
        save_txt:             Save recognized_text.txt to outputs/results/.
        save_pdf:             Save searchable_output.pdf to outputs/pdfs/.
        save_csv:             Save confidence_scores.csv to outputs/results/.

    Returns:
        (DocumentResult, inference_time_seconds)
    """
    from PIL import Image as _Image
    from ocr.preprocessing.pipeline import DocumentPreprocessor
    from ocr.recognition.htr_router import HTRRouter
    from ocr.recognition.engine import LineResult
    from ocr.utils.output_writer import save_text, save_pdf as _save_pdf, save_confidence_csv
    from pathlib import Path as _Path

    t0 = time.perf_counter()

    # Phase 1: adaptive preprocessing
    prep = DocumentPreprocessor()
    try:
        _, bgr = prep.process(source)
    except Exception as e:
        log.error("DocumentPreprocessor failed: %s", e)
        return DocumentResult(source_path=source_path), 0.0

    import cv2 as _cv2
    pil_img = _Image.fromarray(_cv2.cvtColor(bgr, _cv2.COLOR_BGR2RGB))

    # Phases 2-4: route through DAN → fallback
    router = HTRRouter(
        device="auto",
        confidence_threshold=confidence_threshold,
        dan_weights=dan_weights,
        maskrcnn_weights=maskrcnn_weights,
    )
    routed = router.route(pil_img)

    # Wrap into DocumentResult for compatibility with existing output writers
    line_results = [
        LineResult(text=t, confidence=c)
        for t, c in zip(routed.lines, routed.confidences)
    ]
    doc     = postprocess(line_results, source_path=source_path)
    elapsed = time.perf_counter() - t0

    log.info(
        "DAN pipeline done in %.2fs — %d lines [route=%s, mean_conf=%.3f]",
        elapsed, len(doc.lines), routed.route_used, routed.mean_confidence,
    )

    # Derive stem for named output files
    stem = _Path(source_path or source if isinstance(source, str) else "output").stem

    if save_txt:
        save_text(doc, cfg.paths.results_dir / f"{stem}_text.txt")
    if save_csv:
        save_confidence_csv(doc, cfg.paths.results_dir / f"{stem}_confidence.csv")
    if save_pdf:
        _save_pdf(doc, cfg.paths.pdfs_dir / f"{stem}_ocr.pdf")

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
