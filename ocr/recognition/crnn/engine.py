"""
CRNN Inference Engine
Loads a trained CRNN+CTC checkpoint and runs batched inference on line crops.
Used by the ensemble router in engine.py.
"""

import numpy as np
import torch
import cv2
from pathlib import Path
from PIL import Image

from config.settings import cfg
from ocr.recognition.crnn.model import CRNN, VOCAB_SIZE, decode_ctc
from ocr.utils.logger import get_logger

log = get_logger(__name__)
C = cfg.crnn


class CRNNEngine:
    def __init__(self, checkpoint_path: str | None = None):
        self._model: CRNN | None = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._ckpt = checkpoint_path

    def load(self):
        if self._model is not None:
            return
        if not self._ckpt or not Path(self._ckpt).exists():
            log.warning("CRNN checkpoint not found at '%s' — CRNN engine disabled", self._ckpt)
            return
        log.info("Loading CRNN model from %s on %s", self._ckpt, self._device)
        self._model = CRNN(vocab_size=VOCAB_SIZE).to(self._device)
        state = torch.load(self._ckpt, map_location=self._device, weights_only=True)
        self._model.load_state_dict(state)
        self._model.eval()
        log.info("CRNN model ready")

    @property
    def available(self) -> bool:
        return self._model is not None

    def _preprocess_crop(self, crop: np.ndarray) -> torch.Tensor:
        """Resize to fixed height, normalize to [0,1], return (1,1,H,W) tensor."""
        if crop.ndim == 3:
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        h, w = crop.shape
        target_h = C.input_height
        scale = target_h / h
        target_w = max(1, int(w * scale))
        resized = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(resized).float() / 255.0
        return tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    @torch.inference_mode()
    def run_batch(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        """
        Returns list of (text, confidence) for each crop.
        Confidence = mean softmax probability of the argmax token sequence.
        """
        if not self.available:
            return [("", 0.0)] * len(crops)

        results = []
        for crop in crops:
            if crop is None or crop.size == 0:
                results.append(("", 0.0))
                continue
            x = self._preprocess_crop(crop).to(self._device)
            logits = self._model(x)          # (T, 1, vocab)
            probs = torch.softmax(logits, dim=-1)  # (T, 1, vocab)
            best_probs, best_idx = probs.max(dim=-1)  # (T, 1)
            indices = best_idx[:, 0].cpu().tolist()
            conf = float(best_probs[:, 0].mean().cpu())
            text = decode_ctc(indices)
            results.append((text, conf))
        return results


# ── Singleton ─────────────────────────────────────────────────────────────────
_crnn_engine: CRNNEngine | None = None


def get_crnn_engine(checkpoint_path: str | None = None) -> CRNNEngine:
    global _crnn_engine
    if _crnn_engine is None:
        path = checkpoint_path or str(cfg.paths.checkpoints_dir / "crnn_best.pth")
        _crnn_engine = CRNNEngine(path)
        _crnn_engine.load()
    return _crnn_engine
