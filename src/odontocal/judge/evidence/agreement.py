from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def cohens_kappa(a: NDArray[np.int64], b: NDArray[np.int64], n_classes: int) -> float:
    if a.shape != b.shape:
        raise ValueError("a and b must have the same shape")
    if n_classes < 2:
        raise ValueError("n_classes must be at least 2")
    n = a.shape[0]
    if n == 0:
        return 0.0
    table = np.zeros((n_classes, n_classes), dtype=np.float64)
    for ai, bi in zip(a.tolist(), b.tolist(), strict=False):
        table[ai, bi] += 1.0
    total = table.sum()
    po = float(np.trace(table) / total)
    row_marg = table.sum(axis=1) / total
    col_marg = table.sum(axis=0) / total
    pe = float((row_marg * col_marg).sum())
    if pe >= 1.0:
        return 1.0 if po >= 1.0 else 0.0
    return float((po - pe) / (1.0 - pe))
