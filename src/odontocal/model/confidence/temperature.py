from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar

_LOGIT_EPS: Final[float] = 1e-6
_TEMP_BOUNDS: Final[tuple[float, float]] = (0.05, 20.0)


def _verbalised_to_logit(confidence: NDArray[np.float64]) -> NDArray[np.float64]:
    clipped = np.clip(confidence, _LOGIT_EPS, 1.0 - _LOGIT_EPS)
    return np.log(clipped / (1.0 - clipped))


def _binary_softmax(logit: NDArray[np.float64], temperature: float) -> NDArray[np.float64]:
    scaled = logit / temperature
    return 1.0 / (1.0 + np.exp(-scaled))


def _nll_from_verbalised(
    temperature: float, logit: NDArray[np.float64], correct: NDArray[np.bool_]
) -> float:
    p = _binary_softmax(logit, temperature)
    p = np.clip(p, _LOGIT_EPS, 1.0 - _LOGIT_EPS)
    correct_f = correct.astype(np.float64)
    return float(-np.mean(correct_f * np.log(p) + (1.0 - correct_f) * np.log(1.0 - p)))


def fit_global_temperature(
    confidence: NDArray[np.float64],
    correct: NDArray[np.bool_],
) -> float:
    if confidence.shape != correct.shape:
        raise ValueError("confidence and correct must align")
    if confidence.size == 0:
        return 1.0
    logit = _verbalised_to_logit(confidence)
    result = minimize_scalar(
        _nll_from_verbalised,
        bounds=_TEMP_BOUNDS,
        method="bounded",
        args=(logit, correct),
        options={"xatol": 1e-5},
    )
    return float(result.x)


def fit_modality_temperatures(
    confidence: NDArray[np.float64],
    correct: NDArray[np.bool_],
    modality: NDArray[np.str_],
) -> dict[str, float]:
    if confidence.shape != correct.shape or confidence.shape != modality.shape:
        raise ValueError("confidence, correct, modality must all align")
    out: dict[str, float] = {}
    for tag in np.unique(modality):
        mask = modality == tag
        if not mask.any():
            continue
        out[str(tag)] = fit_global_temperature(confidence[mask], correct[mask])
    return out


def apply_temperature(confidence: NDArray[np.float64], temperature: float) -> NDArray[np.float64]:
    logit = _verbalised_to_logit(confidence)
    return _binary_softmax(logit, temperature)


@dataclass
class GlobalTS:
    temperature: float = 1.0

    def fit(self, confidence: NDArray[np.float64], correct: NDArray[np.bool_]) -> GlobalTS:
        self.temperature = fit_global_temperature(confidence, correct)
        return self

    def transform(self, confidence: NDArray[np.float64]) -> NDArray[np.float64]:
        return apply_temperature(confidence, self.temperature)


@dataclass
class ModalityAwareTS:
    temperatures: dict[str, float] = field(default_factory=dict)

    def fit(
        self,
        confidence: NDArray[np.float64],
        correct: NDArray[np.bool_],
        modality: NDArray[np.str_],
    ) -> ModalityAwareTS:
        self.temperatures = fit_modality_temperatures(confidence, correct, modality)
        return self

    def transform(
        self,
        confidence: NDArray[np.float64],
        modality: NDArray[np.str_],
    ) -> NDArray[np.float64]:
        if confidence.shape != modality.shape:
            raise ValueError("confidence and modality must align")
        out = np.empty_like(confidence, dtype=np.float64)
        for tag, temp in self.temperatures.items():
            mask = modality == tag
            if mask.any():
                out[mask] = apply_temperature(confidence[mask], temp)
        missing = np.array([m not in self.temperatures for m in modality], dtype=bool)
        if missing.any():
            out[missing] = confidence[missing]
        return out
