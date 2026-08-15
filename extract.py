"""
extract.py — Primary LLM-based handwritten text extraction CLI.

Primary flow  : image → Groq vision API → structured lines → TXT / CSV / PDF
Fallback/check: --cross-check also runs the local TrOCR/CRNN pipeline
                (ocr.pipeline) and highlights lines where the two outputs
                disagree, so you only need to review uncertain lines.

Usage:
    python extract.py --image notes.jpg
    python extract.py --image notes.jpg --review
    python extract.py --image notes.jpg --output-dir ./results --cross-check
"""

import argparse
import base64
import csv
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

# ── Dependency imports (fail fast with a clear message) ───────────────────────

try:
    from dotenv import load_dotenv
except ImportError:
    sys.exit("Missing dependency: pip install python-dotenv")

try:
    from groq import Groq, RateLimitError, APIError
except ImportError:
    sys.exit("Missing dependency: pip install groq")

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
except ImportError:
    sys.exit("Missing dependency: pip install reportlab")

# ── Prompts ───────────────────────────────────────────────────────────────────

# System message: locks the model into transcription-only output mode.
# Explicitly forbids thinking blocks, reasoning, drafts, and commentary.
SYSTEM_PROMPT = (
    "You are a handwriting transcription engine. "
    "Output ONLY the final transcribed lines, one per line. "
    "No reasoning, no explanation, no commentary, no repeated drafts. "
    "Just the transcribed text lines and nothing else."
)

USER_PROMPT = (
    "Transcribe every line of handwritten text in this image, one line per output line. "
    "If a word is unreadable write [word?]. "
    "Output the transcription lines only — nothing else."
)


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class Line:
    number: int
    text: str
    flagged: bool   # True if the line contains at least one [word?] marker


def parse_lines(raw_text: str) -> list[Line]:
    """Split GPT response into Line objects, detecting [word?] flags."""
    lines = []
    for i, text in enumerate(raw_text.splitlines(), start=1):
        text = text.rstrip()
        if not text:
            continue
        flagged = "[" in text and "?" in text
        lines.append(Line(number=i, text=text, flagged=flagged))
    return lines


# ── Image encoding ────────────────────────────────────────────────────────────

def encode_image(path: Path) -> tuple[str, str]:
    """Return (base64_string, media_type) for a jpg/png image."""
    suffix = path.suffix.lower()
    media_type = "image/jpeg" if suffix in {".jpg", ".jpeg"} else "image/png"
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8"), media_type


# ── Response cleaner ─────────────────────────────────────────────────────────

# Tagged reasoning blocks (e.g. Qwen <think>...</think>)
_THINK_TAG_RE = re.compile(
    r"<(think|thinking|reasoning|internal|scratchpad)[^>]*>.*?</\1>",
    re.IGNORECASE | re.DOTALL,
)

# Lines that are clearly model meta-commentary, not transcription content.
# Matches lines that START with these reasoning-leak patterns.
_META_LINE_RE = re.compile(
    r"^("
    r"the user wants|let me|let'?s|wait[,.]?|actually[,.]?"
    r"|looking at|i can see|i notice|i'll|i will|i need"
    r"|step \d|line \d+:|\d+\.|analyzing|re-examining"
    r"|upon|here is|here are|the (image|text|handwriting)"
    r"|ok[,.]?|okay[,.]?|sure[,.]?"
    r")",
    re.IGNORECASE,
)

def clean_response(text: str) -> str:
    """
    Three-layer safety net applied to every model response before saving:
    1. Strip tagged reasoning blocks (<think>, <reasoning>, etc.)
    2. Drop lines that are clearly model meta-commentary / reasoning prose
    3. Deduplicate: if the model repeated a line while self-correcting,
       keep only the LAST occurrence (the final corrected version)
    """
    # Layer 1 — remove tagged blocks
    text = _THINK_TAG_RE.sub("", text)
    text = re.sub(r"</?think[^>]*>", "", text, flags=re.IGNORECASE)

    # Layer 2 — drop meta-commentary lines
    kept = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if _META_LINE_RE.match(stripped):
            continue
        kept.append(stripped)

    # Layer 3 — deduplicate repeated lines, keeping last occurrence
    # (reasoning models often repeat the same line several times with edits;
    #  the last version is the model's final answer)
    seen: dict[str, int] = {}          # normalised_line -> index in `kept`
    for idx, line in enumerate(kept):
        key = re.sub(r"\s+", " ", line.lower())   # normalise whitespace + case
        seen[key] = idx                             # overwrite → last wins
    deduped = [kept[i] for i in sorted(seen.values())]

    return "\n".join(deduped)


# ── Groq vision call ─────────────────────────────────────────────────────────

def call_groq_vision(image_path: Path, api_key: str) -> str:
    """Send image to Groq and return clean transcription string."""
    client = Groq(api_key=api_key)
    b64, media_type = encode_image(image_path)

    print("Sending to Groq vision API...")
    try:
        response = client.chat.completions.create(
            model="qwen/qwen3.6-27b",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": USER_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{media_type};base64,{b64}"},
                        },
                    ],
                },
            ],
            max_tokens=4096,
            temperature=0,
        )
    except RateLimitError:
        sys.exit("Error: Groq rate limit reached. Wait a moment and retry.")
    except APIError as e:
        sys.exit(f"Error: Groq API error — {e}")

    raw = response.choices[0].message.content.strip()
    return clean_response(raw)


# ── Optional review step ──────────────────────────────────────────────────────

def interactive_review(text: str) -> str:
    """
    Open the transcription in the user's default editor via a temp file.
    Returns the (possibly edited) text after the editor closes.
    """
    print("\n--- Extracted text (opening in editor for review) ---")
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(text)
        tmp_path = tmp.name

    editor = (
        os.environ.get("VISUAL")
        or os.environ.get("EDITOR")
        or ("notepad" if sys.platform == "win32" else "nano")
    )
    subprocess.call([editor, tmp_path])

    with open(tmp_path, encoding="utf-8") as f:
        edited = f.read()
    os.unlink(tmp_path)
    return edited.strip()


# ── Export writers ────────────────────────────────────────────────────────────

def write_txt(lines: list[Line], path: Path) -> None:
    print(f"Writing TXT... → {path}")
    path.write_text("\n".join(ln.text for ln in lines), encoding="utf-8")


def write_csv(lines: list[Line], path: Path) -> None:
    print(f"Writing CSV... → {path}")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["line_number", "text", "flagged"])
        for ln in lines:
            writer.writerow([ln.number, ln.text, ln.flagged])


def write_pdf(lines: list[Line], path: Path, title: str) -> None:
    print(f"Writing PDF... -> {path}")
    doc = SimpleDocTemplate(str(path), pagesize=A4,
                            leftMargin=40, rightMargin=40,
                            topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    normal_style = styles["Normal"]
    normal_style.fontSize = 11
    normal_style.leading = 16

    story = [Paragraph(title, styles["Title"]), Spacer(1, 12)]
    last_idx = len(lines) - 1
    for i, ln in enumerate(lines):
        text = ln.text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        # Last paragraph rendered in red as the final-review highlight
        if i == last_idx:
            para = Paragraph(f'<font color="red">{text}</font>', normal_style)
        else:
            para = Paragraph(text, normal_style)
        story.append(para)
    doc.build(story)


# ── Cross-check against local HTR pipeline ────────────────────────────────────

def run_cross_check(image_path: Path, gpt_lines: list[Line]) -> None:
    """
    Run the existing TrOCR/CRNN pipeline on the same image and print any
    lines where the two outputs differ (CER > threshold).
    This is purely informational — it does not modify the export files.
    """
    print("\nRunning cross-check with local HTR pipeline...")
    try:
        from ocr.pipeline import run as htr_run
    except ImportError as e:
        print(f"  [cross-check skipped] Could not import local pipeline: {e}")
        return

    try:
        doc, elapsed = htr_run(str(image_path), source_path=str(image_path))
    except Exception as e:
        print(f"  [cross-check skipped] Pipeline error: {e}")
        return

    htr_texts = [ln.text for ln in doc.lines]
    gpt_texts = [ln.text for ln in gpt_lines]

    disagreements = []
    for i, (gpt_text, htr_text) in enumerate(
        zip(gpt_texts, htr_texts), start=1
    ):
        cer = _cer(gpt_text, htr_text)
        if cer > 0.30:                   # flag lines with >30% character error rate
            disagreements.append((i, gpt_text, htr_text, cer))

    if not disagreements:
        print("  ✓ No significant disagreements between GPT-4V and HTR pipeline.")
        return

    print(f"\n  ⚠  {len(disagreements)} line(s) where GPT-4V and HTR disagree "
          f"(CER > 30%) — review these:\n")
    print(f"  {'Line':<6}  {'CER':>5}  {'GPT-4V':<45}  HTR")
    print("  " + "-" * 90)
    for line_no, gpt_t, htr_t, cer in disagreements:
        gpt_display = (gpt_t[:42] + "...") if len(gpt_t) > 45 else gpt_t
        htr_display = (htr_t[:42] + "...") if len(htr_t) > 45 else htr_t
        print(f"  {line_no:<6}  {cer:>4.0%}  {gpt_display:<45}  {htr_display}")

    if len(htr_texts) != len(gpt_texts):
        print(f"\n  Note: line counts differ — GPT-4V: {len(gpt_texts)}, "
              f"HTR: {len(htr_texts)}. Comparison is truncated to the shorter.")


def _cer(a: str, b: str) -> float:
    """Levenshtein-based character error rate."""
    if not b:
        return 1.0
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[:], i
        for j in range(1, n + 1):
            dp[j] = (prev[j - 1] if a[i - 1] == b[j - 1]
                     else 1 + min(prev[j], dp[j - 1], prev[j - 1]))
    return dp[n] / max(1, n)


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    load_dotenv()                        # load .env if present

    parser = argparse.ArgumentParser(
        description="Extract handwritten text via Groq vision API and export to TXT/CSV/PDF."
    )
    parser.add_argument("--image", required=True, help="Path to input image (jpg/png)")
    parser.add_argument("--output-dir", default="./output",
                        help="Directory for output files (default: ./output)")
    parser.add_argument("--review", action="store_true",
                        help="Open extracted text in editor before exporting")
    parser.add_argument("--cross-check", action="store_true",
                        help="Also run local HTR pipeline and highlight disagreements against Groq output")
    args = parser.parse_args()

    # ── Validate inputs ───────────────────────────────────────────────────────
    image_path = Path(args.image)
    if not image_path.exists():
        sys.exit(f"Error: image not found — {image_path}")
    if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        sys.exit("Error: image must be a .jpg or .png file")

    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        sys.exit(
            "Error: GROQ_API_KEY not set.\n"
            "  Set it in your shell:  export GROQ_API_KEY=gsk_...\n"
            "  Or create a .env file: GROQ_API_KEY=gsk_..."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem                # e.g. "notes" from "notes.jpg"

    # ── Extract ───────────────────────────────────────────────────────────────
    print(f"Reading image: {image_path}")
    raw_text = call_groq_vision(image_path, api_key)

    # ── Optional review ───────────────────────────────────────────────────────
    if args.review:
        raw_text = interactive_review(raw_text)

    lines = parse_lines(raw_text)
    if not lines:
        sys.exit("Error: Groq returned an empty transcription.")

    flagged_count = sum(1 for ln in lines if ln.flagged)
    print(f"Transcribed {len(lines)} line(s), {flagged_count} flagged as ambiguous.")

    # ── Export ────────────────────────────────────────────────────────────────
    write_txt(lines, output_dir / f"{stem}_output.txt")
    write_csv(lines, output_dir / f"{stem}_output.csv")
    write_pdf(lines, output_dir / f"{stem}_output.pdf", title=f"Transcription — {stem}")

    # ── Optional cross-check ──────────────────────────────────────────────────
    if args.cross_check:
        run_cross_check(image_path, lines)

    print(f"\nDone. Files saved to {output_dir}/")


if __name__ == "__main__":
    main()
