from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from odontocal.judge.quality.calibration import reliability_bins


class ReliabilityCurve(NamedTuple):
    bin_centres: NDArray[np.float64]
    bin_accuracy: NDArray[np.float64]
    bin_confidence: NDArray[np.float64]
    bin_counts: NDArray[np.int64]
    bin_gap: NDArray[np.float64]


def build_reliability_curve(
    confidence: NDArray[np.float64],
    correct: NDArray[np.bool_],
    n_bins: int = 15,
) -> ReliabilityCurve:
    summary = reliability_bins(confidence, correct, n_bins=n_bins)
    centres = 0.5 * (summary.bin_edges[:-1] + summary.bin_edges[1:])
    gap = summary.bin_accuracy - summary.bin_confidence
    return ReliabilityCurve(
        bin_centres=centres.astype(np.float64),
        bin_accuracy=summary.bin_accuracy,
        bin_confidence=summary.bin_confidence,
        bin_counts=summary.bin_counts,
        bin_gap=gap.astype(np.float64),
    )
