"""
Stage 2 — Spatial Line Grouping & Reading Order Sorting
========================================================
Groups detected word bounding boxes into horizontal text lines using a
Y-centroid overlap sweep, then sorts words left-to-right within each line.

Algorithm
---------
1. Sort all word boxes by their vertical centroid (y_mid).
2. Sweep top-to-bottom: assign each word to an existing line group if its
   vertical range overlaps the group's running height range by more than
   ``overlap_ratio`` of the shorter height.  Otherwise open a new group.
3. Within each group, sort words by their left edge (x1) → reading order.
4. Merge each group's word bboxes into a single line bbox.

This is equivalent to a single-pass DBSCAN on the Y axis with a dynamic
epsilon derived from word heights — no sklearn dependency required.

Usage
-----
    grouper = SpatialLineGrouper()
    lines   = grouper.group(word_regions)
    for line in lines:
        print([w.index for w in line.words])
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from ocr.detection.word_detector import WordRegion
from ocr.utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Output datatype
# ---------------------------------------------------------------------------

@dataclass
class TextLineGroup:
    """
    A group of word regions that belong to the same text line.

    Attributes:
        line_index: Zero-based reading-order line index (top-to-bottom).
        words:      Left-to-right ordered list of :class:`WordRegion` objects.
        bbox:       Merged axis-aligned bbox ``(x1, y1, x2, y2)`` of the line.
    """
    line_index: int
    words:      List[WordRegion] = field(default_factory=list)
    bbox:       Tuple[int, int, int, int] = (0, 0, 0, 0)

    @property
    def text_from_words(self) -> str:
        """Space-joined word texts (populated after recognition)."""
        return " ".join(getattr(w, "text", "") for w in self.words)


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class SpatialLineGrouper:
    """
    Stage 2 — Spatial Line Grouping & Reading Order Sorting.

    Groups word bounding boxes into horizontal text lines by vertical
    centroid overlap, then sorts each line left-to-right.

    Args:
        overlap_ratio: Minimum fractional vertical overlap between a word
                       and an existing line group to assign the word to that
                       group (default 0.4 = 40 %).
        min_words:     Minimum number of words per line to keep the group
                       (default 1 — keeps single-word lines).

    Example:
        >>> grouper = SpatialLineGrouper()
        >>> lines   = grouper.group(word_regions)
        >>> for line in lines:
        ...     print(f"Line {line.line_index}: {len(line.words)} words")
    """

    def __init__(
        self,
        overlap_ratio: float = 0.4,
        min_words: int = 1,
    ) -> None:
        self._overlap_ratio = overlap_ratio
        self._min_words     = min_words

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _y_mid(word: WordRegion) -> float:
        """Vertical centroid of a word bbox."""
        return (word.bbox[1] + word.bbox[3]) / 2.0

    @staticmethod
    def _merge_bbox(words: List[WordRegion]) -> Tuple[int, int, int, int]:
        """Merge word bboxes into a single enclosing bbox."""
        x1 = min(w.bbox[0] for w in words)
        y1 = min(w.bbox[1] for w in words)
        x2 = max(w.bbox[2] for w in words)
        y2 = max(w.bbox[3] for w in words)
        return (x1, y1, x2, y2)

    @staticmethod
    def _median_word_height(words: List[WordRegion]) -> float:
        """Median word height across all words — used as stable line-height estimate."""
        heights = [float(w.bbox[3] - w.bbox[1]) for w in words]
        heights.sort()
        mid = len(heights) // 2
        return max(1.0, heights[mid] if heights else 1.0)

    def _overlaps(
        self,
        word: WordRegion,
        group_y1: float,
        group_y2: float,
        line_height: float,
    ) -> bool:
        """
        Return True if *word*'s vertical centroid is within ``overlap_ratio``
        of one median word-height from the group's centroid.

        Using centroid distance (not bbox overlap) prevents the group bbox
        from growing unboundedly and swallowing words from other lines.

        Args:
            word:        Candidate word region.
            group_y1:    Current group top Y.
            group_y2:    Current group bottom Y.
            line_height: Median word height used as distance threshold.

        Returns:
            True if the word belongs to this line group.
        """
        wy_mid  = (word.bbox[1] + word.bbox[3]) / 2.0
        grp_mid = (group_y1 + group_y2) / 2.0
        return abs(wy_mid - grp_mid) < line_height * self._overlap_ratio

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def group(self, words: List[WordRegion]) -> List[TextLineGroup]:
        """
        Group word regions into horizontal text lines.

        Args:
            words: List of :class:`WordRegion` objects (any order).

        Returns:
            List of :class:`TextLineGroup` objects sorted top-to-bottom,
            with words within each group sorted left-to-right.
            Returns an empty list if *words* is empty.
        """
        if not words:
            return []

        # Compute stable line-height from median word height
        line_height = self._median_word_height(words)

        # Sort words top-to-bottom by vertical centroid
        sorted_words = sorted(words, key=self._y_mid)

        # Each group stores: [words_list, running_y1, running_y2]
        groups: List[List] = []

        for word in sorted_words:
            wy1, wy2 = float(word.bbox[1]), float(word.bbox[3])
            placed = False
            for grp in groups:
                if self._overlaps(word, grp[1], grp[2], line_height):
                    grp[0].append(word)
                    grp[1] = min(grp[1], wy1)
                    grp[2] = max(grp[2], wy2)
                    placed = True
                    break
            if not placed:
                groups.append([[word], wy1, wy2])

        # Filter, sort left-to-right within each group, build output
        result: List[TextLineGroup] = []
        # Sort groups top-to-bottom by their top Y
        groups.sort(key=lambda g: g[1])

        for line_idx, (grp_words, _, _) in enumerate(groups):
            if len(grp_words) < self._min_words:
                continue
            # Left-to-right sort by word left edge
            grp_words.sort(key=lambda w: w.bbox[0])
            bbox = self._merge_bbox(grp_words)
            result.append(TextLineGroup(
                line_index=line_idx,
                words=grp_words,
                bbox=bbox,
            ))

        log.info(
            "SpatialLineGrouper: %d words → %d lines.",
            len(words), len(result),
        )
        return result
