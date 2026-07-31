from pathlib import Path

inits = [
    "ocr/__init__.py",
    "ocr/preprocessing/__init__.py",
    "ocr/detection/__init__.py",
    "ocr/recognition/__init__.py",
    "ocr/postprocessing/__init__.py",
    "ocr/evaluation/__init__.py",
    "ocr/dataset/__init__.py",
    "ocr/utils/__init__.py",
    "config/__init__.py",
    "tests/__init__.py",
]

for p in inits:
    Path(p).write_text("", encoding="utf-8")
    print(f"OK: {p}")
