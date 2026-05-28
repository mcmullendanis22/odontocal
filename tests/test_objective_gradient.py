from __future__ import annotations

import math

import numpy as np

from odontocal.model.confidence.objective import (
    binary_nll_from_verbalised,
    binary_nll_gradient_log_temperature,
)


def _finite_diff(fn, t: float, h: float = 1e-4) -> float:
    return (fn(math.exp(math.log(t) + h)) - fn(math.exp(math.log(t) - h))) / (2.0 * h)


def test_gradient_matches_finite_difference_at_random_inputs(rng: np.random.Generator) -> None:
    confidence = rng.uniform(0.05, 0.95, size=200).astype(np.float64)
    truth = rng.integers(0, 2, size=200).astype(bool)
    for t in (0.4, 0.9, 1.3, 2.5):
        grad = binary_nll_gradient_log_temperature(confidence, truth, t)
        num = _finite_diff(lambda tt: binary_nll_from_verbalised(confidence, truth, tt), t)
        assert math.isfinite(grad)
        assert abs(grad - num) < 5e-3


def test_overconfident_population_drives_log_t_higher(rng: np.random.Generator) -> None:
    confidence = np.full(500, 0.95, dtype=np.float64)
    truth = np.zeros(500, dtype=bool)
    truth[:250] = True
    grad = binary_nll_gradient_log_temperature(confidence, truth, temperature=1.0)
    assert grad < 0.0
