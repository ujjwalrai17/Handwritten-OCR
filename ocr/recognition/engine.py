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
class LineResult:
    text: str
    confidence: float                        # mean token probability (0-1)
    token_confidences: list[float] = field(default_factory=list)
    needs_review: bool = False               # True when cross-check flags disagreement
    difficulty_tag: str = "clean"            # "clean" | "hard" — set by detector


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
        self._model = VisionEncoderDecoderModel.from_pretrained(C.name)
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

        if E.blend_mode == "trocr_only":
            return trocr_results

        from ocr.recognition.crnn.engine import get_crnn_engine
        crnn = get_crnn_engine()

        if E.blend_mode == "crnn_only":
            if not crnn.available:
                log.warning("CRNN engine not available, falling back to TrOCR")
                return trocr_results
            crnn_results = crnn.run_batch(crops)
            return [
                LineResult(text=t, confidence=c, difficulty_tag=difficulty_tags[i])
                for i, (t, c) in enumerate(crnn_results)
            ]

        # confidence mode: CRNN fallback for low-confidence lines
        if not crnn.available:
            return trocr_results

        low_conf_idx = [
            i for i, r in enumerate(trocr_results)
            if r.confidence < E.trocr_confidence_threshold
        ]
        if low_conf_idx:
            low_crops = [crops[i] for i in low_conf_idx]
            crnn_results = crnn.run_batch(low_crops)
            for j, orig_i in enumerate(low_conf_idx):
                crnn_text, crnn_conf = crnn_results[j]
                if crnn_conf > trocr_results[orig_i].confidence and crnn_text.strip():
                    log.debug("Line %d: CRNN (%.2f) > TrOCR (%.2f)", orig_i, crnn_conf,
                              trocr_results[orig_i].confidence)
                    trocr_results[orig_i] = LineResult(
                        text=crnn_text, confidence=crnn_conf,
                        difficulty_tag=difficulty_tags[orig_i]
                    )

        # Textract cross-check on still-low-confidence lines
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

            outputs = self._model.generate(
                pixel_values,
                num_beams=C.beam_size,
                max_new_tokens=C.max_new_tokens,
                # ── Hallucination suppression ──────────────────────────────
                length_penalty=C.length_penalty,
                repetition_penalty=C.repetition_penalty,
                no_repeat_ngram_size=C.no_repeat_ngram_size,
                # ──────────────────────────────────────────────────────────
                output_scores=True,
                return_dict_in_generate=True,
            )

            for j, orig_i in enumerate(batch_i):
                token_strings, token_probs = self._extract_token_confidences(
                    outputs.scores, outputs.sequences, batch_pos=j, num_beams=C.beam_size
                )
                line_conf = float(np.mean(token_probs)) if token_probs else 0.0

                # Apply illegible masking — this is the core anti-hallucination step
                masked_text = _apply_illegible_mask(
                    token_strings, token_probs,
                    token_threshold=C.illegible_token_threshold,
                    line_conf=line_conf,
                    line_threshold=C.illegible_line_threshold,
                )

                results[orig_i] = LineResult(
                    text=masked_text,
                    confidence=line_conf,
                    token_confidences=token_probs,
                    difficulty_tag=difficulty_tags[orig_i],
                )

        return [results[i] for i in range(len(crops))]


# ── Singleton ─────────────────────────────────────────────────────────────────
_engine = TrOCREngine()

def get_engine() -> TrOCREngine:
    return _engine
