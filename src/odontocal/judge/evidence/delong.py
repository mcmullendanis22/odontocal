from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm


class DeLongResult(NamedTuple):
    auc_a: float
    auc_b: float
    z: float
    p_value: float
    var_diff: float


def _midrank(x: NDArray[np.float64]) -> NDArray[np.float64]:
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    n = x.shape[0]
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        avg = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def _structural_components(
    scores: NDArray[np.float64],
    truth: NDArray[np.bool_],
) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    pos = scores[truth]
    neg = scores[~truth]
    m, n = pos.shape[0], neg.shape[0]
    if m == 0 or n == 0:
        raise ValueError("DeLong requires at least one positive and one negative sample")
    combined = np.concatenate([pos, neg])
    tx = _midrank(combined[:m])
    ty = _midrank(combined[m:])
    tz = _midrank(combined)
    auc = float((tz[:m].sum() - m * (m + 1) / 2.0) / (m * n))
    v10 = (tz[:m] - tx) / float(n)
    v01 = 1.0 - (tz[m:] - ty) / float(m)
    return auc, v10, v01


def delong_paired_auc_test(
    scores_a: NDArray[np.float64],
    scores_b: NDArray[np.float64],
    truth_positive: NDArray[np.bool_],
) -> DeLongResult:
    if scores_a.shape != scores_b.shape or scores_a.shape != truth_positive.shape:
        raise ValueError("scores_a, scores_b, truth_positive must all align")
    auc_a, v10_a, v01_a = _structural_components(scores_a, truth_positive)
    auc_b, v10_b, v01_b = _structural_components(scores_b, truth_positive)
    s10 = np.cov(np.stack([v10_a, v10_b], axis=0), ddof=1)
    s01 = np.cov(np.stack([v01_a, v01_b], axis=0), ddof=1)
    m = int(truth_positive.sum())
    n = int((~truth_positive).sum())
    sigma = s10 / m + s01 / n
    var_diff = float(sigma[0, 0] + sigma[1, 1] - 2.0 * sigma[0, 1])
    if var_diff <= 0.0:
        z = 0.0
        p = 1.0
    else:
        z = float((auc_a - auc_b) / np.sqrt(var_diff))
        p = float(2.0 * (1.0 - norm.cdf(abs(z))))
    return DeLongResult(auc_a=auc_a, auc_b=auc_b, z=z, p_value=p, var_diff=var_diff)
