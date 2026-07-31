"""
Batch Prediction Script
Runs OCR on a folder of images and saves all outputs.
Usage:
    python predict.py --folder data/samples/
    python predict.py --folder data/samples/ --save-pdf
    python predict.py --folder data/samples/ --model checkpoints/best_model
"""

import argparse
import csv
from pathlib import Path

from config.settings import cfg
from ocr.pipeline import run_folder
from ocr.utils.output_writer import save_text, save_confidence_csv, save_pdf, print_results
from ocr.utils.logger import get_logger

log = get_logger("predict")


def parse_args():
    parser = argparse.ArgumentParser(description="Batch OCR prediction")
    parser.add_argument("--folder",    required=True, help="Folder of images to process")
    parser.add_argument("--save-pdf",  action="store_true", help="Generate searchable PDFs")
    parser.add_argument("--model",     type=str, default=None,
                        help="Path to fine-tuned model checkpoint")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.model:
        cfg.model.name = args.model
        log.info("Using model: %s", args.model)

    results = run_folder(args.folder)

    summary_rows = []
    for doc, elapsed in results:
        print_results(doc)
        stem = Path(doc.source_path).stem if doc.source_path else "result"
        save_text(doc, cfg.paths.results_dir / f"{stem}_text.txt")
        save_confidence_csv(doc, cfg.paths.results_dir / f"{stem}_confidence.csv")
        if args.save_pdf:
            save_pdf(doc, cfg.paths.pdfs_dir / f"{stem}_ocr.pdf")
        summary_rows.append({
            "file": doc.source_path,
            "lines": len(doc.lines),
            "mean_confidence": f"{doc.mean_confidence:.4f}",
            "inference_sec": f"{elapsed:.2f}",
        })

    if not summary_rows:
        log.warning("No images found in folder: %s", args.folder)
        return

    # Write batch summary CSV
    summary_path = cfg.paths.results_dir / "batch_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    log.info("Batch summary saved: %s", summary_path)


if __name__ == "__main__":
    main()
