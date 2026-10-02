"""
Handwritten OCR — Main CLI Entry Point

Usage:
    python main.py --image  sample.jpg
    python main.py --folder test_images/
    python main.py --pdf    document.pdf
    python main.py --image  sample.jpg --save-txt
    python main.py --image  sample.jpg --save-csv
    python main.py --image  sample.jpg --save-pdf
    python main.py --image  sample.jpg --evaluate --ground-truth gt.txt
    python main.py --folder images/    --model checkpoints/best_model
"""

import argparse
import sys
import json
from datetime import datetime
from pathlib import Path

from config.settings import cfg
from ocr.pipeline import run, run_folder, run_pdf
from ocr.utils.output_writer import save_text, save_confidence_csv, save_pdf, print_results
from ocr.evaluation.metrics import evaluate, save_evaluation_report
from ocr.utils.logger import get_logger

log = get_logger("main")


def parse_args():
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Handwritten Text Recognition using TrOCR",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",  type=str, metavar="PATH", help="Single image file")
    group.add_argument("--folder", type=str, metavar="PATH", help="Folder of images")
    group.add_argument("--pdf",    type=str, metavar="PATH", help="PDF document")

    parser.add_argument("--save-txt",  action="store_true", help="Save recognized_text.txt")
    parser.add_argument("--save-csv",  action="store_true", help="Save confidence_scores.csv")
    parser.add_argument("--save-pdf",  action="store_true", help="Save searchable_output.pdf")

    parser.add_argument("--evaluate",     action="store_true",
                        help="Compute CER/WER (requires --ground-truth)")
    parser.add_argument("--ground-truth", type=str, metavar="PATH",
                        help="Ground truth .txt file (one line per image line)")

    parser.add_argument("--model", type=str, default=None,
                        help="Path to fine-tuned checkpoint (default: HuggingFace pretrained)")
    parser.add_argument("--normalize-strokes", action="store_true",
                        help="Enable stroke-width normalization")
    parser.add_argument("--ensemble", choices=["trocr_only", "confidence"],
                        default=None, help="Override ensemble blend mode")

    return parser.parse_args()


def _load_ground_truth(path: str) -> list[str]:
    return Path(path).read_text(encoding="utf-8").strip().splitlines()


def _make_sample_folder(image_path: str) -> Path:
    stem = Path(image_path).stem
    sample_dir = Path("output") / stem
    (sample_dir / "preprocessing").mkdir(parents=True, exist_ok=True)
    (sample_dir / "extracted").mkdir(parents=True, exist_ok=True)
    return sample_dir


def _write_metadata(sample_dir: Path, image_path: str, prep_meta: dict):
    meta = {
        "input_image": Path(image_path).name,
        "binarization_method": prep_meta.get("binarization_method", "otsu"),
        "preprocessing_stages": prep_meta.get("stages", []),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    (sample_dir / "processing_metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )


def _run_with_debug(args, sample_dir: Path):
    """Run TrOCR pipeline with preprocessing debug output into sample_dir."""
    from ocr.preprocessing.pipeline import preprocess, PreprocessingError
    from ocr.detection.detector import detect_lines
    from ocr.recognition.engine import get_engine
    from ocr.postprocessing.corrector import postprocess, DocumentResult
    import time

    debug_dir = sample_dir / "preprocessing"
    t0 = time.perf_counter()
    try:
        binary, bgr, prep_meta = preprocess(
            args.image,
            normalize_strokes=args.normalize_strokes,
            debug_dir=debug_dir,
        )
    except PreprocessingError as e:
        log.error("Preprocessing failed: %s", e)
        return DocumentResult(source_path=args.image), 0.0, {}

    det_dir = sample_dir / "detection"
    det_dir.mkdir(parents=True, exist_ok=True)

    lines = detect_lines(binary, bgr, debug_dir=str(det_dir))
    if not lines:
        return DocumentResult(source_path=args.image), 0.0, prep_meta

    try:
        import cv2 as _cv2
        vis = bgr.copy()
        for ln in lines:
            x1, y1, x2, y2 = ln.bbox
            _cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        _cv2.imwrite(str(det_dir / "final_detection_boxes.png"), vis)
        for idx, ln in enumerate(lines):
            _cv2.imwrite(str(det_dir / f"line_{idx:03d}.png"), ln.crop)
    except Exception as _e:
        log.debug("Detection vis failed: %s", _e)

    engine = get_engine()
    difficulty_tags = [ln.difficulty_tag for ln in lines]
    line_results = engine.run_batch(
        [ln.crop for ln in lines], difficulty_tags=difficulty_tags
    )
    doc = postprocess(line_results, source_path=args.image)
    elapsed = time.perf_counter() - t0
    return doc, elapsed, prep_meta


def _handle_single(args):
    stem = Path(args.image).stem
    sample_dir = _make_sample_folder(args.image)
    prep_meta = {}

    doc, elapsed, prep_meta = _run_with_debug(args, sample_dir)

    print_results(doc)

    extracted = sample_dir / "extracted"
    txt_path = save_text(doc, extracted / f"{stem}.txt")
    print(f"  TXT saved : {txt_path}")
    csv_path = save_confidence_csv(doc, extracted / f"{stem}.csv")
    print(f"  CSV saved : {csv_path}")
    pdf_path = save_pdf(doc, extracted / f"{stem}.pdf")
    print(f"  PDF saved : {pdf_path}")
    print(f"  Sample dir: {sample_dir}")

    _write_metadata(sample_dir, args.image, prep_meta)

    if args.evaluate:
        if not args.ground_truth:
            log.error("--evaluate requires --ground-truth")
            sys.exit(1)
        refs  = _load_ground_truth(args.ground_truth)
        preds = [ln.corrected_text for ln in doc.lines]
        tags  = [ln.difficulty_tag  for ln in doc.lines]
        flags = [ln.needs_review    for ln in doc.lines]
        metrics = evaluate(preds, refs, inference_time_sec=elapsed,
                           difficulty_tags=tags, needs_review_flags=flags)
        print("\n-- Evaluation Results ------------------------------------------")
        print(metrics)
        save_evaluation_report(preds, refs, metrics,
                               difficulty_tags=tags, needs_review_flags=flags)


def _handle_folder(args):
    results = run_folder(args.folder)
    all_preds = []

    for doc, elapsed in results:
        print_results(doc)
        stem = Path(doc.source_path).stem
        if args.save_txt:
            save_text(doc, cfg.paths.results_dir / f"{stem}_text.txt")
        if args.save_csv:
            save_confidence_csv(doc, cfg.paths.results_dir / f"{stem}_confidence.csv")
        if args.save_pdf:
            save_pdf(doc, cfg.paths.pdfs_dir / f"{stem}_ocr.pdf")
        all_preds.extend(ln.corrected_text for ln in doc.lines)

    if args.evaluate and args.ground_truth:
        all_refs   = _load_ground_truth(args.ground_truth)
        total_time = sum(t for _, t in results)
        metrics    = evaluate(all_preds, all_refs, inference_time_sec=total_time)
        print("\n── Evaluation Results ──────────────────────────")
        print(metrics)
        save_evaluation_report(all_preds, all_refs, metrics)


def _handle_pdf(args):
    results = run_pdf(args.pdf)
    for i, (doc, elapsed) in enumerate(results, 1):
        print(f"\n{'='*60}")
        print(f"  PAGE {i}")
        print_results(doc)
        if args.save_txt:
            save_text(doc, cfg.paths.results_dir / f"page_{i:02d}_text.txt")
        if args.save_csv:
            save_confidence_csv(doc, cfg.paths.results_dir / f"page_{i:02d}_confidence.csv")
        if args.save_pdf:
            save_pdf(doc, cfg.paths.pdfs_dir / f"page_{i:02d}_ocr.pdf")


def main():
    args = parse_args()

    if args.model:
        cfg.model.name = args.model
        log.info("Using model checkpoint: %s", args.model)

    if args.ensemble:
        cfg.ensemble.blend_mode = args.ensemble
        log.info("Ensemble mode: %s", args.ensemble)

    if args.image:
        _handle_single(args)
    elif args.folder:
        _handle_folder(args)
    elif args.pdf:
        _handle_pdf(args)


if __name__ == "__main__":
    main()
