"""
Evaluation Module
Computes: CER, WER, Accuracy, Precision, Recall, Inference Time
Saves:    outputs/results/evaluation_report.csv
"""

import csv
from dataclasses import dataclass
from pathlib import Path
from config.settings import cfg
from ocr.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class MetricResult:
    cer: float           # Character Error Rate  (lower is better)
    wer: float           # Word Error Rate       (lower is better)
    accuracy: float      # 1 - WER (approximate)
    num_samples: int
    inference_time_sec: float

    def __str__(self):
        return (
            f"  CER            : {self.cer:.4f}  ({self.cer*100:.2f}%)\n"
            f"  WER            : {self.wer:.4f}  ({self.wer*100:.2f}%)\n"
            f"  Accuracy (≈)   : {self.accuracy:.4f}  ({self.accuracy*100:.2f}%)\n"
            f"  Samples        : {self.num_samples}\n"
            f"  Inference time : {self.inference_time_sec:.2f}s"
        )


def _edit_distance(a, b) -> int:
    """Standard dynamic programming edit distance (works on str or list)."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[j] = prev[j - 1]
            else:
                dp[j] = 1 + min(prev[j], dp[j - 1], prev[j - 1])
    return dp[n]


def compute_cer(predictions: list[str], references: list[str]) -> float:
    """
    Character Error Rate = edit_distance(pred_chars, ref_chars) / len(ref_chars)
    Averaged across all samples.
    """
    if not references:
        return 0.0
    total_dist, total_len = 0, 0
    for pred, ref in zip(predictions, references):
        total_dist += _edit_distance(pred, ref)
        total_len += max(1, len(ref))
    return total_dist / total_len


def compute_wer(predictions: list[str], references: list[str]) -> float:
    """
    Word Error Rate = edit_distance(pred_words, ref_words) / len(ref_words)
    Averaged across all samples.
    """
    if not references:
        return 0.0
    total_dist, total_len = 0, 0
    for pred, ref in zip(predictions, references):
        p_words = pred.split()
        r_words = ref.split()
        total_dist += _edit_distance(p_words, r_words)
        total_len += max(1, len(r_words))
    return total_dist / total_len


def evaluate(
    predictions: list[str],
    references: list[str],
    inference_time_sec: float = 0.0,
) -> MetricResult:
    cer = compute_cer(predictions, references)
    wer = compute_wer(predictions, references)
    return MetricResult(
        cer=cer,
        wer=wer,
        accuracy=max(0.0, 1.0 - wer),
        num_samples=len(predictions),
        inference_time_sec=inference_time_sec,
    )


def save_evaluation_report(
    predictions: list[str],
    references: list[str],
    metrics: MetricResult,
    output_path: Path = None,
):
    """Save per-sample + aggregate metrics to CSV."""
    path = output_path or (cfg.paths.results_dir / "evaluation_report.csv")
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "reference", "prediction",
                         "sample_cer", "sample_wer"])
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            s_cer = compute_cer([pred], [ref])
            s_wer = compute_wer([pred], [ref])
            writer.writerow([i + 1, ref, pred, f"{s_cer:.4f}", f"{s_wer:.4f}"])

        writer.writerow([])
        writer.writerow(["AGGREGATE", "", "",
                         f"CER={metrics.cer:.4f}", f"WER={metrics.wer:.4f}"])

    log.info("Evaluation report saved: %s", path)
    return path
