from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class EnsembleOutput:
    prediction: NDArray[np.int64]
    confidence: NDArray[np.float64]


def confidence_weighted_ensemble(
    predictions: NDArray[np.int64],
    calibrated_confidence: NDArray[np.float64],
    n_classes: int,
) -> EnsembleOutput:
    if predictions.ndim != 2 or calibrated_confidence.ndim != 2:
        raise ValueError("predictions and calibrated_confidence must be 2-D (members, samples)")
    if predictions.shape != calibrated_confidence.shape:
        raise ValueError("predictions and calibrated_confidence must have the same shape")
    if n_classes < 2:
        raise ValueError("n_classes must be at least 2")

    n_members, n_samples = predictions.shape
    posterior = np.zeros((n_samples, n_classes), dtype=np.float64)
    weight_totals = np.zeros(n_samples, dtype=np.float64)
    for j in range(n_members):
        for k in range(n_classes):
            mask = predictions[j] == k
            if mask.any():
                posterior[mask, k] += calibrated_confidence[j][mask]
        weight_totals += calibrated_confidence[j]

    safe_totals = np.where(weight_totals > 0.0, weight_totals, 1.0)
    posterior = posterior / safe_totals[:, None]
    ens_prediction = posterior.argmax(axis=1).astype(np.int64)
    ens_confidence = posterior[np.arange(n_samples), ens_prediction].astype(np.float64)
    return EnsembleOutput(prediction=ens_prediction, confidence=ens_confidence)
