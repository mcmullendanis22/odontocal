from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def holm_bonferroni(p_values: NDArray[np.float64], alpha: float = 0.05) -> NDArray[np.bool_]:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if p_values.ndim != 1:
        raise ValueError("p_values must be 1-D")
    n = p_values.shape[0]
    if n == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p_values, kind="mergesort")
    sorted_p = p_values[order]
    thresholds = alpha / (n - np.arange(n))
    sorted_reject = np.zeros(n, dtype=bool)
    failed = False
    for i in range(n):
        if failed:
            sorted_reject[i] = False
            continue
        if sorted_p[i] <= thresholds[i]:
            sorted_reject[i] = True
        else:
            failed = True
    reject = np.zeros(n, dtype=bool)
    reject[order] = sorted_reject
    return reject


def benjamini_hochberg(p_values: NDArray[np.float64], alpha: float = 0.05) -> NDArray[np.bool_]:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    n = p_values.shape[0]
    if n == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p_values, kind="mergesort")
    sorted_p = p_values[order]
    thresholds = alpha * (np.arange(1, n + 1) / n)
    below = sorted_p <= thresholds
    if not below.any():
        return np.zeros(n, dtype=bool)
    cutoff = int(np.max(np.flatnonzero(below)))
    sorted_reject = np.zeros(n, dtype=bool)
    sorted_reject[: cutoff + 1] = True
    reject = np.zeros(n, dtype=bool)
    reject[order] = sorted_reject
    return reject
