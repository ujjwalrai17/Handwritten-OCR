"""
Output Writer
Generates all output files from a DocumentResult:
  1. recognized_text.txt
  2. confidence_scores.csv
  3. searchable_output.pdf  (optional)
"""

import csv
import io
import time
from pathlib import Path
from config.settings import cfg
from ocr.utils.logger import get_logger

log = get_logger(__name__)


def save_text(doc, output_path: Path = None) -> Path:
    """Write full recognized text to .txt file."""
    path = output_path or (cfg.paths.results_dir / "recognized_text.txt")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc.full_text, encoding="utf-8")
    log.info("Text saved: %s", path)
    return path


def save_confidence_csv(doc, output_path: Path = None) -> Path:
    """Write per-line confidence scores to CSV."""
    path = output_path or (cfg.paths.results_dir / "confidence_scores.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["line_id", "text", "confidence", "low_confidence", "raw_text"])
        for i, ln in enumerate(doc.lines):
            writer.writerow([
                i + 1,
                ln.corrected_text,
                f"{ln.confidence:.4f}",
                ln.is_low_confidence,
                ln.raw_text,
            ])
    log.info("Confidence CSV saved: %s", path)
    return path


def save_pdf(doc, output_path: Path = None) -> Path:
    """
    Generate a searchable PDF.
    Low-confidence lines printed in red with confidence % shown inline.
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas as rl_canvas
    from reportlab.lib import colors

    path = output_path or (cfg.paths.pdfs_dir / "searchable_output.pdf")
    path.parent.mkdir(parents=True, exist_ok=True)

    buf = io.BytesIO()
    c = rl_canvas.Canvas(buf, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, height - 40, "Handwritten OCR — Recognized Text")
    c.setFont("Helvetica", 10)
    c.setFillColor(colors.grey)
    c.drawString(40, height - 56, f"Source: {doc.source_path}   |   "
                                   f"Mean confidence: {doc.mean_confidence:.1%}")

    if not doc.lines:
        c.setFont("Helvetica", 11)
        c.setFillColor(colors.grey)
        c.drawString(40, height - 80, "No text recognized.")
        c.save()
        path.write_bytes(buf.getvalue())
        return path

    c.setFont("Helvetica", 11)
    y = height - 76
    for ln in doc.lines:
        if y < 60:
            c.showPage()
            c.setFont("Helvetica", 11)
            y = height - 40
        c.setFillColor(colors.red if ln.is_low_confidence else colors.black)
        c.drawString(40, y, f"{ln.corrected_text}  [{ln.confidence:.0%}]")
        y -= 18

    c.save()
    # Write via temp file to avoid PermissionError if PDF is open
    import tempfile, shutil, os
    tmp = path.parent / (path.stem + "_tmp.pdf")
    try:
        tmp.write_bytes(buf.getvalue())
        shutil.move(str(tmp), str(path))
    except PermissionError:
        ts = int(time.time())
        alt = path.parent / f"{path.stem}_{ts}.pdf"
        tmp.rename(alt)
        path = alt
        log.warning("PDF in use, saved as: %s", path)
    log.info("PDF saved: %s", path)
    return path


def print_results(doc):
    """Pretty-print results to terminal."""
    sep = "-" * 60
    print(f"\n{sep}")
    print(f"  SOURCE     : {doc.source_path or 'stdin'}")
    print(f"  LINES      : {len(doc.lines)}")
    print(f"  CONFIDENCE : {doc.mean_confidence:.1%}  "
          f"({doc.low_confidence_count} low-confidence lines)")
    print(sep)
    for i, ln in enumerate(doc.lines, 1):
        flag = " [LOW]" if ln.is_low_confidence else ""
        print(f"  [{i:02d}] ({ln.confidence:.0%}){flag}  {ln.corrected_text}")
    print(sep + "\n")
