"""
Post-Processing Module
- Thread-safe SymSpell spell correction
- Punctuation preserved
- Numbers, proper nouns, short words bypassed
- Paragraph reconstruction from line results
"""

import re
import threading
from dataclasses import dataclass, field
from config.settings import cfg
from ocr.recognition.engine import LineResult
from ocr.utils.logger import get_logger

log = get_logger(__name__)
C = cfg.postprocessing

# ── Thread-Safe SymSpell ──────────────────────────────────────────────────────

_sym = None
_lock = threading.Lock()


def _get_symspell():
    global _sym
    if _sym is not None:
        return _sym
    with _lock:
        if _sym is None:
            from symspellpy import SymSpell
            from importlib.resources import files
            s = SymSpell(max_dictionary_edit_distance=C.max_edit_distance, prefix_length=7)
            path = str(files("symspellpy").joinpath("frequency_dictionary_en_82_765.txt"))
            s.load_dictionary(path, term_index=0, count_index=1)
            _sym = s
            log.info("SymSpell dictionary loaded")
    return _sym


_WHITELIST: set[str] = set()

def add_to_whitelist(words: list[str]):
    _WHITELIST.update(w.lower() for w in words)


def _correctable(word: str) -> bool:
    if len(word) < 3:
        return False
    if word[0].isupper():
        return False
    if re.search(r"[0-9@#$%^&*_\-/\\]", word):
        return False
    if word.lower() in _WHITELIST:
        return False
    return True


def _correct_word(word: str) -> str:
    m = re.match(r"^([^a-zA-Z]*)([a-zA-Z']+)([^a-zA-Z]*)$", word)
    if not m:
        return word
    pre, core, suf = m.group(1), m.group(2), m.group(3)
    if not _correctable(core):
        return word
    from symspellpy import Verbosity
    suggestions = _get_symspell().lookup(core.lower(), Verbosity.CLOSEST,
                                         max_edit_distance=C.max_edit_distance)
    corrected = suggestions[0].term if suggestions else core
    return pre + corrected + suf


# OCR noise patterns: leading symbols like " # Ell and trailing ... artifacts
_NOISE_PREFIX = re.compile(r'^["#\u201c\u201d\u2018\u2019\s]*(?:Ell\s+)?')
_NOISE_SUFFIX = re.compile(r'[\s\.]{2,}$')
_MULTI_SPACE  = re.compile(r' {2,}')


def _clean_ocr_noise(text: str) -> str:
    """Remove common TrOCR hallucination artifacts from line text."""
    text = _NOISE_PREFIX.sub('', text)
    text = _NOISE_SUFFIX.sub('', text)
    text = _MULTI_SPACE.sub(' ', text)
    return text.strip()


def correct_text(text: str) -> str:
    """Spell-correct text, preserving [illegible] placeholders intact."""
    if not C.spell_correction:
        return _clean_ocr_noise(text)
    # Split on [illegible] spans, only correct the non-placeholder parts
    parts = re.split(r"(\[illegible[^\]]*\])", text)
    corrected_parts = []
    for part in parts:
        if part.startswith("[") and part.endswith("]"):
            corrected_parts.append(part)  # preserve placeholder verbatim
        else:
            cleaned = _clean_ocr_noise(part)
            corrected_parts.append(" ".join(_correct_word(w) for w in cleaned.split()))
    return " ".join(p for p in corrected_parts if p).strip()


# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class ProcessedLine:
    raw_text: str
    corrected_text: str
    confidence: float
    is_low_confidence: bool
    needs_review: bool = False
    difficulty_tag: str = "clean"
    token_confidences: list[float] = field(default_factory=list)
    word_confidences: list = field(default_factory=list)  # list[WordConfidence]


@dataclass
class DocumentResult:
    lines: list[ProcessedLine] = field(default_factory=list)
    source_path: str = ""

    @property
    def full_text(self) -> str:
        return "\n".join(ln.corrected_text for ln in self.lines)

    @property
    def mean_confidence(self) -> float:
        if not self.lines:
            return 0.0
        return sum(ln.confidence for ln in self.lines) / len(self.lines)

    @property
    def low_confidence_count(self) -> int:
        return sum(1 for ln in self.lines if ln.is_low_confidence)


# ── Main ──────────────────────────────────────────────────────────────────────

def keyword_search(doc, keyword: str) -> list[dict]:
    """Search recognized text for a keyword across all lines."""
    keyword_lower = keyword.lower().strip()
    if not keyword_lower:
        return []
    return [
        {"line_index": i, "line_text": ln.corrected_text, "confidence": ln.confidence}
        for i, ln in enumerate(doc.lines)
        if keyword_lower in ln.corrected_text.lower()
    ]


def postprocess(line_results: list[LineResult], source_path: str = "") -> DocumentResult:
    doc = DocumentResult(source_path=source_path)
    for r in line_results:
        if not r.text.strip():
            continue
        corrected = correct_text(r.text)
        doc.lines.append(ProcessedLine(
            raw_text=r.text,
            corrected_text=corrected,
            confidence=r.confidence,
            is_low_confidence=r.confidence < C.confidence_threshold,
            needs_review=r.needs_review,
            difficulty_tag=r.difficulty_tag,
            token_confidences=r.token_confidences,
            word_confidences=r.word_confidences,
        ))
    log.debug("Postprocessed %d lines (%.1f%% low-confidence)",
              len(doc.lines),
              100 * doc.low_confidence_count / max(1, len(doc.lines)))
    return doc
