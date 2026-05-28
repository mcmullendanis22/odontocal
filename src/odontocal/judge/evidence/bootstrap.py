from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray


class BootstrapCI(NamedTuple):
    point: float
    low: float
    high: float


def stratified_bootstrap_ci(
    score_fn: Callable[[NDArray[np.int64]], float],
    n_samples: int,
    strata: NDArray[np.int64] | None = None,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> BootstrapCI:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if n_resamples < 100:
        raise ValueError("n_resamples must be at least 100")

    if strata is None:
        strata = np.zeros(n_samples, dtype=np.int64)
    if strata.shape != (n_samples,):
        raise ValueError("strata must have shape (n_samples,)")

    base_idx = np.arange(n_samples, dtype=np.int64)
    point = score_fn(base_idx)

    rng = np.random.default_rng(seed)
    unique_strata = np.unique(strata)
    samples: list[float] = []
    for _ in range(n_resamples):
        parts: list[NDArray[np.int64]] = []
        for s in unique_strata:
            members = base_idx[strata == s]
            if members.size == 0:
                continue
            parts.append(rng.choice(members, size=members.size, replace=True))
        resampled = np.concatenate(parts) if parts else base_idx
        samples.append(score_fn(resampled))

    samples_arr = np.asarray(samples, dtype=np.float64)
    lo = float(np.quantile(samples_arr, alpha / 2.0))
    hi = float(np.quantile(samples_arr, 1.0 - alpha / 2.0))
    return BootstrapCI(point=float(point), low=lo, high=hi)
