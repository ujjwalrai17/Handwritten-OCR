"""
PDF Reader Utility
Converts each page of a PDF into a PIL Image for the OCR pipeline.
"""

from PIL import Image
from ocr.utils.logger import get_logger

log = get_logger(__name__)


def pdf_to_images(pdf_path: str, dpi: int = 200) -> list[Image.Image]:
    """
    Convert every page of a PDF to a PIL Image at the given DPI.
    Requires: pdf2image + poppler
    """
    try:
        from pdf2image import convert_from_path
        pages = convert_from_path(pdf_path, dpi=dpi)
        log.info("PDF '%s': %d pages extracted at %d DPI", pdf_path, len(pages), dpi)
        return pages
    except ImportError:
        log.error("pdf2image not installed. Run: pip install pdf2image")
        raise
    except Exception as e:
        log.error("Failed to read PDF '%s': %s", pdf_path, e)
        raise
