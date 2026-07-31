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

    # ── Input (mutually exclusive) ────────────────────────────────────────────
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",  type=str, metavar="PATH", help="Single image file")
    group.add_argument("--folder", type=str, metavar="PATH", help="Folder of images")
    group.add_argument("--pdf",    type=str, metavar="PATH", help="PDF document")

    # ── Output options ────────────────────────────────────────────────────────
    parser.add_argument("--save-txt",  action="store_true", help="Save recognized_text.txt")
    parser.add_argument("--save-csv",  action="store_true", help="Save confidence_scores.csv")
    parser.add_argument("--save-pdf",  action="store_true", help="Save searchable_output.pdf")

    # ── Evaluation ────────────────────────────────────────────────────────────
    parser.add_argument("--evaluate",      action="store_true",
                        help="Compute CER/WER (requires --ground-truth)")
    parser.add_argument("--ground-truth",  type=str, metavar="PATH",
                        help="Ground truth .txt file (one line per image line)")

    # ── Model ─────────────────────────────────────────────────────────────────
    parser.add_argument("--model", type=str, default=None,
                        help="Path to fine-tuned checkpoint (default: HuggingFace pretrained)")

    return parser.parse_args()


def _load_ground_truth(path: str) -> list[str]:
    return Path(path).read_text(encoding="utf-8").strip().splitlines()


def _handle_single(args):
    doc, elapsed = run(args.image, source_path=args.image)
    print_results(doc)

    if args.save_txt:
        save_text(doc)
    if args.save_csv:
        save_confidence_csv(doc)
    if args.save_pdf:
        save_pdf(doc)

    if args.evaluate:
        if not args.ground_truth:
            log.error("--evaluate requires --ground-truth")
            sys.exit(1)
        refs = _load_ground_truth(args.ground_truth)
        preds = [ln.corrected_text for ln in doc.lines]
        metrics = evaluate(preds, refs, inference_time_sec=elapsed)
        print("\n── Evaluation Results ──────────────────────────")
        print(metrics)
        save_evaluation_report(preds, refs, metrics)


def _handle_folder(args):
    results = run_folder(args.folder)
    all_preds, all_refs = [], []

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
        all_refs = _load_ground_truth(args.ground_truth)
        total_time = sum(t for _, t in results)
        metrics = evaluate(all_preds, all_refs, inference_time_sec=total_time)
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

    if args.image:
        _handle_single(args)
    elif args.folder:
        _handle_folder(args)
    elif args.pdf:
        _handle_pdf(args)


if __name__ == "__main__":
    main()
