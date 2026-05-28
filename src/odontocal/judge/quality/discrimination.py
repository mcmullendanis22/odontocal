from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def accuracy(predictions: NDArray[np.int64], targets: NDArray[np.int64]) -> float:
    if predictions.shape != targets.shape:
        raise ValueError("predictions and targets must have the same shape")
    if predictions.size == 0:
        return 0.0
    return float((predictions == targets).mean())


def sensitivity_specificity(
    predicted_positive: NDArray[np.bool_],
    truth_positive: NDArray[np.bool_],
) -> tuple[float, float]:
    pos = int(truth_positive.sum())
    neg = int((~truth_positive).sum())
    tp = int(np.logical_and(predicted_positive, truth_positive).sum())
    tn = int(np.logical_and(~predicted_positive, ~truth_positive).sum())
    sens = float(tp / pos) if pos else 0.0
    spec = float(tn / neg) if neg else 0.0
    return sens, spec


def f1_score(
    predicted_positive: NDArray[np.bool_],
    truth_positive: NDArray[np.bool_],
) -> float:
    tp = int(np.logical_and(predicted_positive, truth_positive).sum())
    fp = int(np.logical_and(predicted_positive, ~truth_positive).sum())
    fn = int(np.logical_and(~predicted_positive, truth_positive).sum())
    denom = 2 * tp + fp + fn
    if denom == 0:
        return 0.0
    return float(2 * tp / denom)


def macro_f1(predictions: NDArray[np.int64], targets: NDArray[np.int64], n_classes: int) -> float:
    if predictions.shape != targets.shape:
        raise ValueError("predictions and targets must have the same shape")
    if n_classes < 2:
        raise ValueError("n_classes must be at least 2")
    per_class: list[float] = []
    for k in range(n_classes):
        per_class.append(f1_score(predictions == k, targets == k))
    return float(np.mean(per_class))


def auc_roc(scores: NDArray[np.float64], truth_positive: NDArray[np.bool_]) -> float:
    if scores.shape != truth_positive.shape:
        raise ValueError("scores and truth_positive must have the same shape")
    n_pos = int(truth_positive.sum())
    n_neg = int((~truth_positive).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    n = scores.shape[0]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = avg
        i = j + 1
    sum_ranks_pos = float(ranks[truth_positive].sum())
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))
