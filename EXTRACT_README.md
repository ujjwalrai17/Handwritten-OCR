# extract.py — LLM-Based Handwritten Text Extraction

Sends a handwritten notes image to **GPT-4V** (via the OpenAI API) and exports
the transcription as **TXT**, **CSV**, and **PDF**.

The existing TrOCR/CRNN pipeline is preserved as an optional cross-checker
(`--cross-check`), not replaced.

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

New packages added for `extract.py`:

| Package | Purpose |
|---|---|
| `openai>=1.30.0` | GPT-4V API client |
| `python-dotenv>=1.0.0` | Load API key from `.env` |
| `fpdf2>=2.7.0` | Client-side PDF generation |

### 2. Set your OpenAI API key

**Option A — environment variable (recommended):**

```bash
# Linux / macOS
export OPENAI_API_KEY=sk-...

# Windows (cmd)
set OPENAI_API_KEY=sk-...

# Windows (PowerShell)
$env:OPENAI_API_KEY="sk-..."
```

**Option B — `.env` file** in the project root:

```
OPENAI_API_KEY=sk-...
```

The key is read at runtime and never written to disk or logged.

---

## Usage

### Basic extraction

```bash
python extract.py --image data/samples/notes.jpg
```

Outputs to `./output/`:
- `notes_output.txt` — plain text transcript
- `notes_output.csv` — line-by-line with `line_number`, `text`, `flagged` columns
- `notes_output.pdf` — formatted PDF (ambiguous lines shaded grey)

### Custom output directory

```bash
python extract.py --image notes.jpg --output-dir ./results
```

### Interactive review before export

```bash
python extract.py --image notes.jpg --review
```

Opens the extracted text in your default editor (`$EDITOR` / `$VISUAL`, or
`notepad` on Windows, `nano` on Linux/macOS). Save and close to proceed to export.

### Cross-check against local HTR pipeline

```bash
python extract.py --image notes.jpg --cross-check
```

Runs the same image through the local TrOCR/CRNN pipeline and prints any lines
where the two outputs disagree by more than 30% CER — so you only need to
manually review the uncertain lines, not the whole document.

Requires the full project dependencies (`torch`, `transformers`, etc.) to be
installed. If the local pipeline is unavailable, the cross-check is skipped
gracefully and the export still completes.

### All flags together

```bash
python extract.py --image notes.jpg --output-dir ./results --review --cross-check
```

---

## Output format

### CSV columns

| Column | Description |
|---|---|
| `line_number` | 1-based line index |
| `text` | Transcribed text for that line |
| `flagged` | `True` if the line contains at least one `[word?]` ambiguity marker |

### Ambiguity markers

When GPT-4V cannot confidently read a word, it wraps it in brackets:
`the quick [brwon?] fox`. These lines are marked `flagged=True` in the CSV
and shaded grey in the PDF.

---

## Error messages

| Message | Fix |
|---|---|
| `OPENAI_API_KEY not set` | Set the env var or create a `.env` file |
| `image not found` | Check the `--image` path |
| `OpenAI rate limit reached` | Wait ~60 s and retry |
| `OpenAI API error` | Check your API key and account quota |
| `GPT-4V returned an empty transcription` | Image may be blank or unreadable |

---

## Architecture

```
extract.py
    │
    ├── encode_image()        read file → base64
    ├── call_gpt4v()          base64 + prompt → OpenAI API → raw text
    ├── parse_lines()         raw text → list[Line(number, text, flagged)]
    ├── interactive_review()  optional: open in $EDITOR, return edited text
    ├── write_txt/csv/pdf()   export to output dir
    └── run_cross_check()     optional: ocr.pipeline.run() → diff vs GPT lines
                              (uses existing TrOCR/CRNN pipeline unchanged)
```
