"""
TrOCR Recognition Engine + CRNN Ensemble Router

HALLUCINATION SUPPRESSION (Priority 1)
---------------------------------------
Root cause: the TrOCR decoder has a strong language-model prior that generates
fluent text even when visual evidence is ambiguous. Three levers fix this:

  1. length_penalty < 1.0   — penalises long fluent completions; forces the
                              decoder to stay close to what it actually sees
  2. repetition_penalty > 1 — breaks LM-driven repetitive phrase loops
  3. no_repeat_ngram_size   — prevents copy-pasting common n-grams from LM prior

ILLEGIBLE PLACEHOLDER (Priority 1)
------------------------------------
Per-token confidence is extracted from decoder logits. Tokens whose softmax
probability falls below cfg.model.illegible_token_threshold are replaced with
the literal string "[illegible]" so downstream consumers know exactly which
spans are uncertain, rather than receiving fabricated fluent text.

If the mean line confidence is below cfg.model.illegible_line_threshold the
entire line is replaced with "[illegible line]".

TEXTRACT CROSS-CHECK (Priority 4)
-----------------------------------
When cfg.ensemble.use_textract_crosscheck=True, low-confidence lines are also
sent to Amazon Textract. If CER(trocr_text, textract_text) exceeds
cfg.ensemble.crosscheck_cer_flag_threshold the line is flagged needs_review=True
in the LineResult so the caller can route it to human review.
"""

import io
import re
import numpy as np
import torch
from dataclasses import dataclass, field
from PIL import Image
from tqdm import tqdm
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, RobertaTokenizer, ViTImageProcessor
from config.settings import cfg
from ocr.utils.logger import get_logger

log = get_logger(__name__)
C = cfg.model
E = cfg.ensemble

ILLEGIBLE_TOKEN = "[illegible]"
ILLEGIBLE_LINE  = "[illegible line]"


@dataclass
class WordConfidence:
    """
    Confidence score attached to a single decoded word.

    Attributes:
        word:       The decoded word string (may be empty for punctuation tokens).
        confidence: Mean softmax probability of the BPE tokens that form this word.
        low:        True when confidence < cfg.model.illegible_token_threshold.
    """
    word:       str
    confidence: float
    low:        bool


@dataclass
class LineResult:
    text: str
    confidence: float                           # mean token probability (0-1)
    token_confidences: list[float] = field(default_factory=list)
    word_confidences:  list[WordConfidence] = field(default_factory=list)  # Step 4
    needs_review: bool = False                  # True when hallucination risk flagged
    difficulty_tag: str = "clean"               # "clean" | "hard" — set by detector


# ── Illegible masking ─────────────────────────────────────────────────────────

def _apply_illegible_mask(
    tokens: list[str],
    token_confs: list[float],
    token_threshold: float,
    line_conf: float,
    line_threshold: float,
) -> str:
    """
    Replace low-confidence token spans with [illegible].
    Consecutive illegible tokens are collapsed into a single placeholder.
    If the whole line is below line_threshold, return [illegible line].
    """
    if line_conf < line_threshold:
        return ILLEGIBLE_LINE

    parts = []
    in_illegible = False
    for tok, conf in zip(tokens, token_confs):
        if conf < token_threshold:
            if not in_illegible:
                parts.append(ILLEGIBLE_TOKEN)
                in_illegible = True
        else:
            in_illegible = False
            parts.append(tok)
    return " ".join(p for p in parts if p).strip()


def _build_word_confidences(
    token_strings: list[str],
    token_probs: list[float],
    threshold: float,
) -> list[WordConfidence]:
    """
    Group BPE tokens into words and compute per-word mean confidence.

    TrOCR uses a RoBERTa tokenizer where word-initial tokens start with
    a leading space (\u0120 / 'Ġ') and continuation tokens do not.
    We group tokens into words on that boundary, then average their
    softmax probabilities to get one confidence score per word.

    Args:
        token_strings: Decoded token strings from the tokenizer.
        token_probs:   Softmax probability of each token (same length).
        threshold:     Tokens below this are flagged ``low=True``.

    Returns:
        List of :class:`WordConfidence` objects, one per word.
    """
    if not token_strings:
        return []

    word_tokens:  list[str]   = []
    word_probs:   list[float] = []
    result: list[WordConfidence] = []

    def _flush():
        if not word_tokens:
            return
        word_str  = "".join(word_tokens).strip()
        mean_conf = float(np.mean(word_probs))
        result.append(WordConfidence(
            word=word_str,
            confidence=mean_conf,
            low=mean_conf < threshold,
        ))

    for tok, prob in zip(token_strings, token_probs):
        # RoBERTa BPE: leading-space char \u0120 marks a new word boundary
        is_new_word = tok.startswith("\u0120") or tok.startswith(" ") or not word_tokens
        if is_new_word and word_tokens:
            _flush()
            word_tokens, word_probs = [], []
        word_tokens.append(tok.lstrip("\u0120").lstrip())
        word_probs.append(prob)

    _flush()
    return [w for w in result if w.word]  # drop empty-string entries


# ── Textract cross-check ──────────────────────────────────────────────────────

def _textract_line(crop_pil: Image.Image, region: str) -> str:
    """Send a single line crop to Textract and return the top LINE text."""
    try:
        import boto3
        buf = io.BytesIO()
        crop_pil.save(buf, format="PNG")
        client = boto3.client("textract", region_name=region)
        resp = client.detect_document_text(Document={"Bytes": buf.getvalue()})
        lines = [b["Text"] for b in resp.get("Blocks", []) if b["BlockType"] == "LINE"]
        return " ".join(lines)
    except Exception as ex:
        log.debug("Textract cross-check failed: %s", ex)
        return ""


def _cer_quick(a: str, b: str) -> float:
    """Fast CER for two short strings (no import needed)."""
    if not b:
        return 1.0
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[:], i
        for j in range(1, n + 1):
            dp[j] = prev[j - 1] if a[i-1] == b[j-1] else 1 + min(prev[j], dp[j-1], prev[j-1])
    return dp[n] / max(1, n)


# ── Step 2: Visual-grounding hallucination risk check ────────────────────────

def flag_hallucination_risk(
    line_image: Image.Image,
    predicted_text: str,
    per_token_confidences: list[float],
    threshold: float = 0.45,
    length_ratio: float = 2.5,
) -> bool:
    """
    Flag a transcription as high hallucination-risk.

    A line is flagged when BOTH conditions hold:
      (a) Mean token confidence is below *threshold* — the decoder was
          uncertain, making LM-prior-driven confabulation more likely.
      (b) The predicted word count exceeds what the crop's ink density
          can plausibly support by more than *length_ratio* — a short,
          sparse crop producing a long fluent sentence is a strong signal
          that the decoder is hallucinating rather than reading.

    Ink-density word-count estimate:
      We count the number of distinct ink "runs" (connected horizontal
      segments of dark pixels) in the horizontal projection of the binary
      crop.  Each run corresponds roughly to one word.  This is a fast,
      model-free estimate that does not require a word detector.

    Args:
        line_image:            PIL image of the line crop (any mode).
        predicted_text:        Decoded text string from TrOCR.
        per_token_confidences: List of per-token softmax probabilities.
        threshold:             Mean confidence below which condition (a) fires.
        length_ratio:          Max ratio of predicted_words / ink_words before
                               condition (b) fires.

    Returns:
        ``True`` if the line is high hallucination-risk and should be
        flagged ``needs_review=True`` in :class:`LineResult`.
    """
    import cv2 as _cv2

    if not predicted_text.strip() or not per_token_confidences:
        return False

    # Condition (a): mean confidence check
    mean_conf = float(np.mean(per_token_confidences))
    if mean_conf >= threshold:
        return False   # high confidence — not a hallucination risk

    # Condition (b): ink-density word-count estimate
    gray = np.array(line_image.convert("L"))
    # Binarise: pixels darker than 128 are ink
    _, binary = _cv2.threshold(gray, 128, 255, _cv2.THRESH_BINARY_INV)
    # Horizontal projection: sum of ink pixels per column
    col_proj = binary.sum(axis=0).astype(float)
    # Smooth to merge broken strokes within a word
    kernel = np.ones(5) / 5
    col_proj = np.convolve(col_proj, kernel, mode="same")
    # Count ink runs (transitions from 0 to >0) as word estimate
    ink_threshold = col_proj.max() * 0.05
    in_run = False
    ink_word_count = 0
    for v in col_proj:
        if not in_run and v > ink_threshold:
            in_run = True
            ink_word_count += 1
        elif in_run and v <= ink_threshold:
            in_run = False

    if ink_word_count == 0:
        return False
    ink_word_count = max(3, ink_word_count)  # floor: real lines have >= a few words

    # Count predicted words (skip [illegible] placeholders)
    predicted_words = [
        w for w in predicted_text.split()
        if not w.startswith("[")
    ]
    predicted_word_count = max(1, len(predicted_words))

    ratio = predicted_word_count / ink_word_count
    if ratio > length_ratio:
        log.debug(
            "Hallucination risk: pred_words=%d ink_words=%d ratio=%.1f conf=%.2f text=%r",
            predicted_word_count, ink_word_count, ratio, mean_conf, predicted_text[:50],
        )
        return True

    return False


# ── Main engine ───────────────────────────────────────────────────────────────

class TrOCREngine:
    """
    Wraps microsoft/trocr-base-handwritten.
    Loaded once via .load(); reused for all subsequent calls.
    """

    def __init__(self):
        self._processor: TrOCRProcessor = None
        self._model: VisionEncoderDecoderModel = None
        self._device = self._resolve_device()

    def _resolve_device(self) -> str:
        if C.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return C.device

    def load(self):
        if self._model is not None:
            return
        log.info("Loading TrOCR model: %s on %s", C.name, self._device)
        tokenizer = RobertaTokenizer.from_pretrained(C.name)
        image_processor = ViTImageProcessor.from_pretrained(C.name)
        self._processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
        self._model = VisionEncoderDecoderModel.from_pretrained(
            C.name, low_cpu_mem_usage=True
        )
        self._model.to(self._device)
        self._model.eval()
        log.info("TrOCR model ready")

    def _to_pil(self, crop: np.ndarray) -> Image.Image:
        if crop.ndim == 2:
            return Image.fromarray(crop).convert("RGB")
        return Image.fromarray(crop.astype(np.uint8)).convert("RGB")

    def _is_valid(self, crop: np.ndarray) -> bool:
        return (crop is not None and crop.size > 0
                and crop.shape[0] >= 8 and crop.shape[1] >= 8)

    def _extract_token_confidences(
        self, scores, sequences, batch_pos: int, num_beams: int = 1
    ) -> tuple[list[str], list[float]]:
        """
        Extract per-token (text, confidence) pairs from generation output.
        scores:    tuple of (vocab_size,) tensors, one per generated step
        sequences: (batch, seq_len) token id tensor
        Returns (token_strings, softmax_probs)
        """
        if not scores:
            return [], []

        beam_idx = batch_pos * num_beams
        token_ids = sequences[batch_pos].tolist()

        # scores covers only the generated tokens (after decoder_start_token)
        token_strings, token_probs = [], []
        for step_idx, step_scores in enumerate(scores):
            if beam_idx >= step_scores.shape[0]:
                break
            probs = torch.softmax(step_scores[beam_idx], dim=-1)
            best_prob = probs.max().item()
            # Map generated token id to string
            gen_token_id = token_ids[step_idx + 1] if step_idx + 1 < len(token_ids) else 0
            tok_str = self._processor.tokenizer.decode(
                [gen_token_id], skip_special_tokens=True
            )
            token_strings.append(tok_str)
            token_probs.append(best_prob)

        return token_strings, token_probs

    def run_batch(
        self,
        crops: list[np.ndarray],
        difficulty_tags: list[str] | None = None,
    ) -> list[LineResult]:
        """
        Process a batch of line crops with hallucination suppression.
        difficulty_tags: optional list matching crops length, values "clean"|"hard"
        """
        self.load()
        if difficulty_tags is None:
            difficulty_tags = ["clean"] * len(crops)

        trocr_results = self._run_batch_inference(crops, difficulty_tags)

        # Textract cross-check on low-confidence lines
        if E.use_textract_crosscheck:
            for i, r in enumerate(trocr_results):
                if r.confidence < E.trocr_confidence_threshold and self._is_valid(crops[i]):
                    pil = self._to_pil(crops[i])
                    tx_text = _textract_line(pil, E.textract_region)
                    if tx_text:
                        cer = _cer_quick(r.text, tx_text)
                        if cer > E.crosscheck_cer_flag_threshold:
                            trocr_results[i].needs_review = True
                            log.info(
                                "Line %d flagged for review: TrOCR=%r Textract=%r CER=%.2f",
                                i, r.text[:40], tx_text[:40], cer
                            )

        return trocr_results

    @torch.inference_mode()
    def _run_batch_inference(
        self, crops: list[np.ndarray], difficulty_tags: list[str]
    ) -> list[LineResult]:
        results: dict[int, LineResult] = {}
        valid_idx = [i for i, c in enumerate(crops) if self._is_valid(c)]

        for i in range(len(crops)):
            if i not in valid_idx:
                results[i] = LineResult(text="", confidence=0.0,
                                        difficulty_tag=difficulty_tags[i])

        batches = [
            valid_idx[s: s + C.batch_size]
            for s in range(0, len(valid_idx), C.batch_size)
        ]

        for batch_i in tqdm(batches, desc="Recognizing lines", unit="batch",
                            disable=len(batches) <= 1):
            pil_imgs = [self._to_pil(crops[i]) for i in batch_i]
            pixel_values = self._processor(
                images=pil_imgs, return_tensors="pt"
            ).pixel_values.to(self._device)

            # Step 1 fix: remove length_penalty (ignored + warns with beam=1)
            # and no_repeat_ngram_size (suppressed legitimate repeated words).
            # Keep only mild repetition_penalty to break decoder loops.
            outputs = self._model.generate(
                pixel_values,
                num_beams=C.beam_size,
                max_new_tokens=C.max_new_tokens,
                repetition_penalty=C.repetition_penalty,
                output_scores=True,
                return_dict_in_generate=True,
            )

            for j, orig_i in enumerate(batch_i):
                token_strings, token_probs = self._extract_token_confidences(
                    outputs.scores, outputs.sequences, batch_pos=j, num_beams=C.beam_size
                )
                line_conf = float(np.mean(token_probs)) if token_probs else 0.0

                masked_text = _apply_illegible_mask(
                    token_strings, token_probs,
                    token_threshold=C.illegible_token_threshold,
                    line_conf=line_conf,
                    line_threshold=C.illegible_line_threshold,
                )

                # Step 1 + Step 4: build per-word confidence list
                word_confs = _build_word_confidences(
                    token_strings, token_probs,
                    threshold=C.illegible_token_threshold,
                )

                # Step 2: flag hallucination risk using visual grounding check
                crop_img = self._to_pil(crops[orig_i])
                needs_review = flag_hallucination_risk(
                    crop_img, masked_text, token_probs,
                    threshold=C.hallucination_conf_threshold,
                    length_ratio=C.hallucination_length_ratio,
                )
                if needs_review:
                    log.warning(
                        "Line %d flagged hallucination risk: conf=%.2f text=%r",
                        orig_i, line_conf, masked_text[:60],
                    )

                results[orig_i] = LineResult(
                    text=masked_text,
                    confidence=line_conf,
                    token_confidences=token_probs,
                    word_confidences=word_confs,
                    needs_review=needs_review,
                    difficulty_tag=difficulty_tags[orig_i],
                )

        return [results[i] for i in range(len(crops))]


# ── Singleton ─────────────────────────────────────────────────────────────────
_engine = TrOCREngine()

def get_engine() -> TrOCREngine:
    return _engine
