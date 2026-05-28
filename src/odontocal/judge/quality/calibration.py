from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ReliabilitySummary:
    bin_edges: NDArray[np.float64]
    bin_counts: NDArray[np.int64]
    bin_accuracy: NDArray[np.float64]
    bin_confidence: NDArray[np.float64]


def reliability_bins(
    confidences: NDArray[np.float64],
    correct: NDArray[np.bool_],
    n_bins: int = 15,
) -> ReliabilitySummary:
    if confidences.ndim != 1 or correct.ndim != 1:
        raise ValueError("confidences and correct must be 1-D arrays")
    if confidences.shape != correct.shape:
        raise ValueError("confidences and correct must have the same length")
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    edges = np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float64)
    idx = np.clip(np.digitize(confidences, edges[1:-1], right=False), 0, n_bins - 1)
    counts = np.zeros(n_bins, dtype=np.int64)
    acc = np.zeros(n_bins, dtype=np.float64)
    conf = np.zeros(n_bins, dtype=np.float64)
    for b in range(n_bins):
        mask = idx == b
        n = int(mask.sum())
        counts[b] = n
        if n == 0:
            continue
        acc[b] = float(correct[mask].mean())
        conf[b] = float(confidences[mask].mean())
    return ReliabilitySummary(
        bin_edges=edges, bin_counts=counts, bin_accuracy=acc, bin_confidence=conf
    )


def expected_calibration_error(
    confidences: NDArray[np.float64],
    correct: NDArray[np.bool_],
    n_bins: int = 15,
) -> float:
    summary = reliability_bins(confidences, correct, n_bins=n_bins)
    total = int(summary.bin_counts.sum())
    if total == 0:
        return 0.0
    weights = summary.bin_counts.astype(np.float64) / float(total)
    gap = np.abs(summary.bin_accuracy - summary.bin_confidence)
    return float((weights * gap).sum())


def maximum_calibration_error(
    confidences: NDArray[np.float64],
    correct: NDArray[np.bool_],
    n_bins: int = 15,
) -> float:
    summary = reliability_bins(confidences, correct, n_bins=n_bins)
    non_empty = summary.bin_counts > 0
    if not non_empty.any():
        return 0.0
    gap = np.abs(summary.bin_accuracy - summary.bin_confidence)
    return float(gap[non_empty].max())


def brier_score(
    probabilities: NDArray[np.float64],
    targets: NDArray[np.int64],
) -> float:
    if probabilities.ndim != 1:
        raise ValueError("probabilities must be 1-D (probability of the predicted class)")
    if targets.ndim != 1 or targets.shape != probabilities.shape:
        raise ValueError("targets must match probabilities length")
    binary_truth = targets.astype(np.float64)
    diff = probabilities - binary_truth
    return float(np.mean(diff * diff))
