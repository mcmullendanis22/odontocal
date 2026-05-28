from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SplitAssignment:
    train: NDArray[np.int64]
    val: NDArray[np.int64]
    test: NDArray[np.int64]


def build_official_or_stratified_split(
    labels: NDArray[np.int64],
    official: SplitAssignment | None = None,
    fractions: tuple[float, float, float] = (0.7, 0.1, 0.2),
    seed: int = 0,
) -> SplitAssignment:
    if official is not None:
        return official
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("fractions must sum to 1.0")
    if any(f < 0 for f in fractions):
        raise ValueError("fractions must be non-negative")
    n = labels.shape[0]
    if n == 0:
        empty = np.zeros(0, dtype=np.int64)
        return SplitAssignment(train=empty, val=empty, test=empty)

    rng = np.random.default_rng(seed)
    train_parts: list[NDArray[np.int64]] = []
    val_parts: list[NDArray[np.int64]] = []
    test_parts: list[NDArray[np.int64]] = []
    for label in np.unique(labels):
        members = np.flatnonzero(labels == label).astype(np.int64)
        rng.shuffle(members)
        n_train = int(round(members.size * fractions[0]))
        n_val = int(round(members.size * fractions[1]))
        n_train = min(n_train, members.size)
        n_val = min(n_val, members.size - n_train)
        train_parts.append(members[:n_train])
        val_parts.append(members[n_train : n_train + n_val])
        test_parts.append(members[n_train + n_val :])
    return SplitAssignment(
        train=np.concatenate(train_parts).astype(np.int64),
        val=np.concatenate(val_parts).astype(np.int64),
        test=np.concatenate(test_parts).astype(np.int64),
    )
