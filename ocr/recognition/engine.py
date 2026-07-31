"""
TrOCR Recognition Engine
- Lazy model loading (loaded once, reused)
- Batched inference for GPU efficiency
- Per-token confidence extraction from decoder logits (novelty feature)
"""

import numpy as np
import torch
from dataclasses import dataclass, field
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, RobertaTokenizer, ViTImageProcessor
from config.settings import cfg
from ocr.utils.logger import get_logger

log = get_logger(__name__)
C = cfg.model


@dataclass
class LineResult:
    text: str
    confidence: float                        # mean token probability (0–1)
    token_confidences: list[float] = field(default_factory=list)


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

    def _extract_confidence(self, scores, batch_pos: int = 0, num_beams: int = 1) -> list[float]:
        """
        Extract per-token confidence from beam search decoder scores.
        scores: tuple of tensors per generation step, shape [batch*num_beams, vocab]
        """
        if not scores:
            return []
        # With beam search, sequences are interleaved: index = batch_pos * num_beams
        beam_idx = batch_pos * num_beams
        return [
            torch.softmax(step[beam_idx], dim=-1).max().item()
            for step in scores
        ]

    def run_batch(self, crops: list[np.ndarray]) -> list[LineResult]:
        """
        Process a batch of line crops simultaneously.
        Invalid crops return empty LineResult without breaking the batch.
        """
        self.load()  # ensure model is loaded before inference_mode
        return self._run_batch_inference(crops)

    @torch.inference_mode()
    def _run_batch_inference(self, crops: list[np.ndarray]) -> list[LineResult]:
        results: dict[int, LineResult] = {}
        valid_idx = [i for i, c in enumerate(crops) if self._is_valid(c)]

        for i in range(len(crops)):
            if i not in valid_idx:
                results[i] = LineResult(text="", confidence=0.0)

        for start in range(0, len(valid_idx), C.batch_size):
            batch_i = valid_idx[start: start + C.batch_size]
            pil_imgs = [self._to_pil(crops[i]) for i in batch_i]

            pixel_values = self._processor(
                images=pil_imgs, return_tensors="pt"
            ).pixel_values.to(self._device)

            outputs = self._model.generate(
                pixel_values,
                num_beams=C.beam_size,
                max_new_tokens=C.max_new_tokens,
                output_scores=True,
                return_dict_in_generate=True,
            )

            for j, orig_i in enumerate(batch_i):
                text = self._processor.tokenizer.decode(
                    outputs.sequences[j], skip_special_tokens=True
                )
                confs = self._extract_confidence(outputs.scores, batch_pos=j, num_beams=C.beam_size)
                results[orig_i] = LineResult(
                    text=text,
                    confidence=float(np.mean(confs)) if confs else 0.0,
                    token_confidences=confs,
                )

        return [results[i] for i in range(len(crops))]


# ── Singleton ─────────────────────────────────────────────────────────────────
_engine = TrOCREngine()

def get_engine() -> TrOCREngine:
    return _engine
