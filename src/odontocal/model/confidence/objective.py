from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import NDArray

_EPS: Final[float] = 1e-6


def _verbalised_to_logit(confidence: NDArray[np.float64]) -> NDArray[np.float64]:
    clipped = np.clip(confidence, _EPS, 1.0 - _EPS)
    return np.log(clipped / (1.0 - clipped))


def binary_nll_from_verbalised(
    confidence: NDArray[np.float64],
    correct: NDArray[np.bool_],
    temperature: float,
) -> float:
    if temperature <= 0.0:
        raise ValueError("temperature must be strictly positive")
    if confidence.shape != correct.shape:
        raise ValueError("confidence and correct must align")
    logit = _verbalised_to_logit(confidence)
    p = 1.0 / (1.0 + np.exp(-logit / temperature))
    p = np.clip(p, _EPS, 1.0 - _EPS)
    y = correct.astype(np.float64)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def binary_nll_gradient_log_temperature(
    confidence: NDArray[np.float64],
    correct: NDArray[np.bool_],
    temperature: float,
) -> float:
    if temperature <= 0.0:
        raise ValueError("temperature must be strictly positive")
    if confidence.shape != correct.shape:
        raise ValueError("confidence and correct must align")
    logit = _verbalised_to_logit(confidence)
    z = logit / temperature
    p = 1.0 / (1.0 + np.exp(-z))
    y = correct.astype(np.float64)
    return float(np.mean((y - p) * z))
