from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def net_benefit(
    predicted_positive: NDArray[np.bool_],
    truth_positive: NDArray[np.bool_],
    threshold: float,
) -> float:
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be strictly within (0, 1)")
    if predicted_positive.shape != truth_positive.shape:
        raise ValueError("predicted_positive and truth_positive must have the same length")
    n = predicted_positive.shape[0]
    if n == 0:
        return 0.0
    tp = int(np.logical_and(predicted_positive, truth_positive).sum())
    fp = int(np.logical_and(predicted_positive, ~truth_positive).sum())
    odds = threshold / (1.0 - threshold)
    return float(tp / n) - float(fp / n) * float(odds)


def referral_rate(referred: NDArray[np.bool_]) -> float:
    n = referred.shape[0]
    if n == 0:
        return 0.0
    return float(referred.sum() / n)


def error_capture_rate(
    referred: NDArray[np.bool_],
    is_error: NDArray[np.bool_],
) -> float:
    if referred.shape != is_error.shape:
        raise ValueError("referred and is_error must have the same length")
    total_errors = int(is_error.sum())
    if total_errors == 0:
        return 0.0
    captured = int(np.logical_and(referred, is_error).sum())
    return float(captured / total_errors)
