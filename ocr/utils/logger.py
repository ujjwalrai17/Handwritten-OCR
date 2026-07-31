"""
Centralized logger for the OCR pipeline.
Writes to both terminal (colored) and logs/ocr.log simultaneously.
"""

import logging
import sys
from pathlib import Path
from config.settings import cfg

cfg.paths.logs_dir.mkdir(parents=True, exist_ok=True)

_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE = "%H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # Terminal handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter(_FMT, _DATE))

    # File handler
    fh = logging.FileHandler(cfg.paths.logs_dir / "ocr.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(_FMT, _DATE))

    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger
