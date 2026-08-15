"""
Evaluation Module
Computes: CER, WER, Accuracy, Inference Time
Split by difficulty tag: "clean" vs "hard" (overlapping strokes)

Per-difficulty split is the key metric for measuring whether targeted
fine-tuning actually improves the hard cases, not just the easy average.
"""

import csv
from dataclasses import dataclass, field
from pathlib import Path
from config.settings import cfg
from ocr.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class MetricResult:
    cer: float
    wer: float
    accuracy: float
    num_samples: int
    inference_time_sec: float
    # Per-difficulty breakdown
    cer_clean: float = 0.0
    wer_clean: float = 0.0
    num_clean: int = 0
    cer_hard: float = 0.0
    wer_hard: float = 0.0
    num_hard: int = 0
    # Review flagging
    num_needs_review: int = 0

    def __str__(self):
        lines = [
            f"  CER (overall)  : {self.cer:.4f}  ({self.cer*100:.2f}%)",
            f"  WER (overall)  : {self.wer:.4f}  ({self.wer*100:.2f}%)",
            f"  Accuracy (~)   : {self.accuracy:.4f}  ({self.accuracy*100:.2f}%)",
            f"  Samples        : {self.num_samples}",
            f"  Inference time : {self.inference_time_sec:.2f}s",
        ]
        if self.num_clean > 0:
            lines.append(
                f"  CER [clean]    : {self.cer_clean:.4f}  "
                f"({self.num_clean} samples)"
            )
        if self.num_hard > 0:
            lines.append(
                f"  CER [hard]     : {self.cer_hard:.4f}  "
                f"({self.num_hard} samples)  <-- target metric"
            )
        if self.num_needs_review > 0:
            lines.append(f"  Needs review   : {self.num_needs_review} lines flagged")
        return "\n".join(lines)


def _edit_distance(a, b) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            dp[j] = prev[j-1] if a[i-1] == b[j-1] else 1 + min(prev[j], dp[j-1], prev[j-1])
    return dp[n]


def compute_cer(predictions: list[str], references: list[str]) -> float:
    if not references:
        return 0.0
    total_dist, total_len = 0, 0
    for pred, ref in zip(predictions, references):
        total_dist += _edit_distance(pred, ref)
        total_len += max(1, len(ref))
    return total_dist / total_len


def compute_wer(predictions: list[str], references: list[str]) -> float:
    if not references:
        return 0.0
    total_dist, total_len = 0, 0
    for pred, ref in zip(predictions, references):
        total_dist += _edit_distance(pred.split(), ref.split())
        total_len += max(1, len(ref.split()))
    return total_dist / total_len


def evaluate(
    predictions: list[str],
    references: list[str],
    inference_time_sec: float = 0.0,
    difficulty_tags: list[str] | None = None,
    needs_review_flags: list[bool] | None = None,
) -> MetricResult:
    """
    Compute CER/WER overall and split by difficulty tag.

    Args:
        predictions:       model output strings
        references:        ground truth strings
        difficulty_tags:   list of "clean"|"hard" per sample
        needs_review_flags: list of bool per sample (from cross-check)
    """
    cer = compute_cer(predictions, references)
    wer = compute_wer(predictions, references)

    result = MetricResult(
        cer=cer,
        wer=wer,
        accuracy=max(0.0, 1.0 - wer),
        num_samples=len(predictions),
        inference_time_sec=inference_time_sec,
    )

    # Per-difficulty split
    if difficulty_tags and len(difficulty_tags) == len(predictions):
        clean_preds = [p for p, t in zip(predictions, difficulty_tags) if t == "clean"]
        clean_refs  = [r for r, t in zip(references,  difficulty_tags) if t == "clean"]
        hard_preds  = [p for p, t in zip(predictions, difficulty_tags) if t == "hard"]
        hard_refs   = [r for r, t in zip(references,  difficulty_tags) if t == "hard"]

        result.num_clean  = len(clean_preds)
        result.cer_clean  = compute_cer(clean_preds, clean_refs)
        result.wer_clean  = compute_wer(clean_preds, clean_refs)
        result.num_hard   = len(hard_preds)
        result.cer_hard   = compute_cer(hard_preds, hard_refs)
        result.wer_hard   = compute_wer(hard_preds, hard_refs)

    if needs_review_flags:
        result.num_needs_review = sum(1 for f in needs_review_flags if f)

    return result


def save_evaluation_report(
    predictions: list[str],
    references: list[str],
    metrics: MetricResult,
    output_path: Path = None,
    difficulty_tags: list[str] | None = None,
    needs_review_flags: list[bool] | None = None,
):
    """Save per-sample + aggregate metrics to CSV, including difficulty column."""
    path = output_path or (cfg.paths.results_dir / "evaluation_report.csv")
    path.parent.mkdir(parents=True, exist_ok=True)

    tags  = difficulty_tags or ["unknown"] * len(predictions)
    flags = needs_review_flags or [False] * len(predictions)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_id", "difficulty", "needs_review",
            "reference", "prediction", "sample_cer", "sample_wer"
        ])
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            s_cer = compute_cer([pred], [ref])
            s_wer = compute_wer([pred], [ref])
            writer.writerow([
                i + 1, tags[i], flags[i],
                ref, pred, f"{s_cer:.4f}", f"{s_wer:.4f}"
            ])

        writer.writerow([])
        writer.writerow(["AGGREGATE", "all", "",  "", "",
                         f"CER={metrics.cer:.4f}", f"WER={metrics.wer:.4f}"])
        if metrics.num_clean > 0:
            writer.writerow(["AGGREGATE", "clean", "", "", "",
                             f"CER={metrics.cer_clean:.4f}",
                             f"WER={metrics.wer_clean:.4f}"])
        if metrics.num_hard > 0:
            writer.writerow(["AGGREGATE", "hard", "", "", "",
                             f"CER={metrics.cer_hard:.4f}",
                             f"WER={metrics.wer_hard:.4f}"])

    log.info("Evaluation report saved: %s", path)
    return path
