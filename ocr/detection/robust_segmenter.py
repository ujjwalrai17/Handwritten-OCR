"""
Phase 3 — Robust Line Segmentation (Fallback)
==============================================
Instance-segmentation based polygon mask extraction that explicitly handles
curved baselines and touching ascenders/descenders.

Used only when FullPageHTR confidence falls below the routing threshold.

Architecture
------------
Mask-RCNN with ResNet-50-FPN backbone (torchvision).
  - Each predicted instance = one text line polygon mask.
  - Handles curved baselines via per-pixel mask (not just axis-aligned bbox).
  - Touching ascenders/descenders are separated by per-instance masks.

Fallback chain
--------------
  1. Mask-RCNN  (when checkpoint provided)
  2. Existing overlap-aware projection profiling (always available)

Usage
-----
    segmenter = RobustLineSegmenter(device="auto")
    lines = segmenter.segment(pil_page_image)
    for line in lines:
        line.crop_pil.save(f"line_{line.index}.png")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from ocr.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class SegmentedLine:
    """
    A single segmented text line.

    Attributes:
        index:    Zero-based reading-order index.
        bbox:     Axis-aligned ``(x1, y1, x2, y2)`` bounding box.
        polygon:  Tight ``(N, 2)`` float32 polygon (handles curved baselines).
        crop_pil: Masked PIL RGB crop (background set to white).
        score:    Detection confidence in ``[0, 1]``.
    """
    index:    int
    bbox:     Tuple[int, int, int, int]
    polygon:  np.ndarray
    crop_pil: Image.Image
    score:    float = 1.0


class RobustLineSegmenter:
    """
    Phase 3 — Robust Line Segmentation (Mask-RCNN fallback).

    Detects text-line polygon masks using Mask-RCNN when a fine-tuned
    checkpoint is available, falling back to overlap-aware projection
    profiling otherwise.

    Args:
        device:          ``"cuda"``, ``"cpu"``, or ``"auto"``.
        weights_path:    Path to Mask-RCNN ``.pth`` checkpoint or ``None``.
        score_threshold: Minimum instance confidence to keep (default 0.5).
        padding:         Extra pixels around each crop (default 4).

    Example:
        >>> seg = RobustLineSegmenter(device="auto")
        >>> lines = seg.segment(Image.open("page.jpg"))
        >>> print(len(lines), "lines segmented")
    """

    def __init__(
        self,
        device: str = "auto",
        weights_path: Optional[str] = None,
        score_threshold: float = 0.5,
        padding: int = 4,
    ) -> None:
        if device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        self._score_thresh = score_threshold
        self._padding      = padding
        self._model        = self._build_maskrcnn(weights_path)

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def _build_maskrcnn(self, weights_path: Optional[str]):
        """
        Build and optionally load a Mask-RCNN model for text-line segmentation.

        Args:
            weights_path: Path to checkpoint or ``None``.

        Returns:
            Loaded model on self._device, or ``None`` if torchvision unavailable.
        """
        try:
            from torchvision.models.detection import maskrcnn_resnet50_fpn
            from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
            from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

            model = maskrcnn_resnet50_fpn(weights=None, num_classes=2)
            in_feat = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, 2)
            in_feat_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
            model.roi_heads.mask_predictor = MaskRCNNPredictor(in_feat_mask, 256, 2)

            if weights_path is not None:
                from pathlib import Path
                p = Path(weights_path)
                if p.exists():
                    state = torch.load(str(p), map_location=self._device, weights_only=True)
                    model.load_state_dict(state)
                    log.info("RobustLineSegmenter: Mask-RCNN weights loaded from '%s'.", p)
                else:
                    log.warning("RobustLineSegmenter: checkpoint '%s' not found.", p)

            model.to(self._device).eval()
            return model
        except ImportError as exc:
            log.warning("RobustLineSegmenter: torchvision unavailable (%s) — projection fallback.", exc)
            return None

    # ------------------------------------------------------------------
    # Crop helper
    # ------------------------------------------------------------------

    def _polygon_crop(
        self,
        rgb_np: np.ndarray,
        polygon: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> Image.Image:
        """
        Extract a masked crop using the polygon boundary.

        Pixels outside the polygon are set to white (background).

        Args:
            rgb_np:  (H, W, 3) uint8 RGB source image.
            polygon: (N, 2) float32 polygon.
            bbox:    (x1, y1, x2, y2) bounding box.

        Returns:
            Masked PIL RGB crop.
        """
        H, W = rgb_np.shape[:2]
        p    = self._padding
        x1, y1, x2, y2 = bbox
        x1c = max(0, x1 - p); y1c = max(0, y1 - p)
        x2c = min(W, x2 + p); y2c = min(H, y2 + p)

        mask = np.zeros((H, W), dtype=np.uint8)
        pts  = polygon.astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [pts], 255)

        bg  = np.full_like(rgb_np, 255)
        out = np.where(mask[:, :, None] > 0, rgb_np, bg)
        return Image.fromarray(out[y1c:y2c, x1c:x2c])

    # ------------------------------------------------------------------
    # Detection paths
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def _maskrcnn_segment(self, pil_img: Image.Image) -> List[SegmentedLine]:
        """
        Run Mask-RCNN instance segmentation.

        Args:
            pil_img: RGB PIL page image.

        Returns:
            List of :class:`SegmentedLine` sorted top-to-bottom.
        """
        if self._model is None:
            return []

        rgb_np = np.array(pil_img.convert("RGB"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(rgb_np).permute(2, 0, 1).unsqueeze(0).to(self._device)
        preds  = self._model(tensor)[0]

        keep   = preds["scores"] >= self._score_thresh
        boxes  = preds["boxes"][keep].cpu().numpy()
        masks  = preds["masks"][keep].cpu().numpy()
        scores = preds["scores"][keep].cpu().numpy()

        rgb_uint8 = np.array(pil_img.convert("RGB"), dtype=np.uint8)
        lines: List[SegmentedLine] = []

        for i, (box, mask, score) in enumerate(zip(boxes, masks, scores)):
            x1, y1, x2, y2 = map(int, box)
            bin_mask = (mask[0] > 0.5).astype(np.uint8) * 255
            contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            poly = contours[0].reshape(-1, 2).astype(np.float32)
            crop = self._polygon_crop(rgb_uint8, poly, (x1, y1, x2, y2))
            lines.append(SegmentedLine(
                index=i, bbox=(x1, y1, x2, y2),
                polygon=poly, crop_pil=crop, score=float(score),
            ))

        lines.sort(key=lambda ln: ln.bbox[1])
        for i, ln in enumerate(lines):
            ln.index = i
        return lines

    def _projection_segment(self, pil_img: Image.Image) -> List[SegmentedLine]:
        """
        Fallback: overlap-aware projection profiling via existing detector.

        Args:
            pil_img: RGB PIL page image.

        Returns:
            List of :class:`SegmentedLine` sorted top-to-bottom.
        """
        from ocr.preprocessing.pipeline import preprocess
        from ocr.detection.detector import detect_lines

        binary, bgr = preprocess(pil_img)
        text_lines  = detect_lines(binary, bgr)
        rgb_np      = np.array(pil_img.convert("RGB"), dtype=np.uint8)
        lines: List[SegmentedLine] = []

        for tl in text_lines:
            x1, y1, x2, y2 = tl.bbox
            poly = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
            crop = Image.fromarray(rgb_np[y1:y2, x1:x2])
            lines.append(SegmentedLine(
                index=tl.bbox[1],   # temp; re-indexed below
                bbox=tl.bbox, polygon=poly, crop_pil=crop,
            ))

        lines.sort(key=lambda ln: ln.bbox[1])
        for i, ln in enumerate(lines):
            ln.index = i
        return lines

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def segment(self, image: Image.Image) -> List[SegmentedLine]:
        """
        Segment text lines from a page image.

        Tries Mask-RCNN first; falls back to projection profiling.

        Args:
            image: Input PIL RGB page image.

        Returns:
            List of :class:`SegmentedLine` objects sorted top-to-bottom.

        Raises:
            ValueError: If *image* is ``None`` or has zero dimensions.
        """
        if image is None:
            raise ValueError("RobustLineSegmenter.segment: received None image.")
        if image.size[0] == 0 or image.size[1] == 0:
            raise ValueError(f"RobustLineSegmenter.segment: zero-dimension {image.size}.")

        lines = self._maskrcnn_segment(image)
        if lines:
            log.info("RobustLineSegmenter: Mask-RCNN → %d lines.", len(lines))
            return lines

        lines = self._projection_segment(image)
        log.info("RobustLineSegmenter: projection fallback → %d lines.", len(lines))
        return lines
