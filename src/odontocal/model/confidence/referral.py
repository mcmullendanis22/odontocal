from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from odontocal.judge.quality.clinical import error_capture_rate, referral_rate


@dataclass(frozen=True)
class ReferralOutcome:
    referred: NDArray[np.bool_]
    accepted: NDArray[np.bool_]
    threshold: float


def decide(calibrated_confidence: NDArray[np.float64], threshold: float) -> ReferralOutcome:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must lie in [0, 1]")
    accepted = calibrated_confidence >= threshold
    return ReferralOutcome(referred=~accepted, accepted=accepted, threshold=float(threshold))


def pick_threshold(
    calibrated_confidence: NDArray[np.float64],
    is_error: NDArray[np.bool_],
    target_capture: float = 0.85,
    referral_cap: float = 0.20,
    candidates: NDArray[np.float64] | None = None,
) -> float:
    if calibrated_confidence.shape != is_error.shape:
        raise ValueError("calibrated_confidence and is_error must align")
    if not 0.0 <= target_capture <= 1.0:
        raise ValueError("target_capture must lie in [0, 1]")
    if not 0.0 <= referral_cap <= 1.0:
        raise ValueError("referral_cap must lie in [0, 1]")

    grid = candidates if candidates is not None else np.linspace(0.05, 0.95, 19, dtype=np.float64)
    best_tau = float(grid[0])
    best_score = -np.inf
    for tau in grid:
        outcome = decide(calibrated_confidence, float(tau))
        rr = referral_rate(outcome.referred)
        ecr = error_capture_rate(outcome.referred, is_error)
        feasible = ecr >= target_capture and rr <= referral_cap
        score = (ecr - rr) + (1e3 if feasible else 0.0)
        if score > best_score:
            best_score = score
            best_tau = float(tau)
    return best_tau
