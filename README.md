# Handwritten OCR — Research Project

Terminal-based Handwritten Text Recognition using TrOCR, OpenCV, and CRAFT.

## Quick Start

```bash
pip install -r requirements.txt

# Single image
python main.py --image data/samples/sample.jpg

# Save all outputs
python main.py --image data/samples/sample.jpg --save-txt --save-csv --save-pdf

# Folder of images
python main.py --folder data/samples/ --save-txt --save-csv

# PDF document
python main.py --pdf document.pdf --save-txt

# Evaluate against ground truth
python main.py --image sample.jpg --evaluate --ground-truth gt.txt

# Fine-tune on IAM dataset
python train.py

# Fine-tune with custom settings
python train.py --epochs 5 --lr 3e-5 --batch 4

# Batch predict with fine-tuned model
python predict.py --folder data/samples/ --model checkpoints/best_model

# Run tests
pytest tests/ -v
```

## Project Structure

```
Handwritten-OCR/
├── main.py                        # Primary CLI entry point
├── train.py                       # TrOCR fine-tuning pipeline
├── predict.py                     # Batch prediction script
├── requirements.txt
│
├── config/
│   └── settings.py                # All hyperparameters and paths
│
├── ocr/
│   ├── pipeline.py                # Orchestrator (connects all stages)
│   ├── preprocessing/
│   │   └── pipeline.py            # Grayscale→CLAHE→Binarize→Deskew
│   ├── detection/
│   │   └── detector.py            # CRAFT + projection profiling
│   ├── recognition/
│   │   └── engine.py              # TrOCR batched inference + confidence
│   ├── postprocessing/
│   │   └── corrector.py           # SymSpell + DocumentResult
│   ├── evaluation/
│   │   └── metrics.py             # CER, WER, Accuracy
│   ├── dataset/
│   │   └── iam_dataset.py         # IAM + local dataset loaders
│   └── utils/
│       ├── logger.py              # Structured logging
│       ├── output_writer.py       # TXT, CSV, PDF output
│       └── pdf_reader.py          # PDF → PIL Image converter
│
├── tests/
│   └── test_pipeline.py           # 20 unit tests
│
├── data/
│   ├── samples/                   # Test images
│   ├── raw/                       # Raw training data
│   └── processed/                 # Preprocessed data
│
├── checkpoints/                   # Saved model checkpoints
├── outputs/
│   ├── results/                   # TXT, CSV outputs
│   └── pdfs/                      # Searchable PDFs
├── logs/                          # ocr.log
├── notebooks/                     # Jupyter experiments
├── scripts/                       # Helper scripts
└── docs/                          # Documentation
```

## Pipeline

```
Image / PDF / Folder
        ↓
[1] Preprocessing    — Grayscale, CLAHE, Denoise, Otsu/Adaptive, Deskew, Dilate
        ↓
[2] Text Detection   — CRAFT (primary) → Projection Profiling (fallback)
        ↓
[3] Line Segmentation — Word→Line grouping, sorted top-to-bottom
        ↓
[4] TrOCR Recognition — Batched inference, beam search, confidence extraction
        ↓
[5] Post-processing  — SymSpell correction, DocumentResult assembly
        ↓
[6] Evaluation       — CER, WER, Accuracy (if ground truth provided)
        ↓
[7] Output           — Terminal print + TXT + CSV + PDF
```

## Model

- TrOCR: `microsoft/trocr-base-handwritten`
- Dataset: IAM Handwriting Database (via HuggingFace `Teklia/IAM-line`)
- Expected CER: 3–5% after fine-tuning on IAM

## Output Files

| File | Location | Description |
|---|---|---|
| `recognized_text.txt` | `outputs/results/` | Full recognized text |
| `confidence_scores.csv` | `outputs/results/` | Per-line confidence |
| `evaluation_report.csv` | `outputs/results/` | CER/WER per sample |
| `searchable_output.pdf` | `outputs/pdfs/` | Searchable PDF |
| `ocr.log` | `logs/` | Full debug log |
