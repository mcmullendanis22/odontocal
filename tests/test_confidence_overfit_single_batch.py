from __future__ import annotations

import numpy as np

from odontocal.judge.quality.calibration import expected_calibration_error
from odontocal.model.confidence.ensemble import confidence_weighted_ensemble
from odontocal.model.confidence.platt import PlattCalibrator
from odontocal.model.confidence.referral import decide, pick_threshold
from odontocal.model.confidence.reliability import build_reliability_curve
from odontocal.model.confidence.temperature import GlobalTS, ModalityAwareTS, fit_global_temperature


def _overconfident_batch(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    n = 600
    correct = rng.integers(0, 2, size=n).astype(bool)
    confidence = np.where(correct, rng.uniform(0.85, 0.99, size=n), rng.uniform(0.80, 0.98, size=n))
    return confidence.astype(np.float64), correct


def test_global_ts_decreases_ece_on_overconfident_batch(rng: np.random.Generator) -> None:
    confidence, correct = _overconfident_batch(rng)
    base = expected_calibration_error(confidence, correct)
    calibrator = GlobalTS().fit(confidence, correct)
    post = expected_calibration_error(calibrator.transform(confidence), correct)
    assert post + 1e-6 < base
    assert calibrator.temperature > 1.0


def test_mats_per_modality_temperatures_differ(rng: np.random.Generator) -> None:
    n = 400
    modality_a = np.where(np.arange(n) < n // 2, "panoramic", "cephalometric").astype(np.str_)
    correct = rng.integers(0, 2, size=n).astype(bool)
    confidence = np.where(
        modality_a == "panoramic",
        np.where(correct, rng.uniform(0.7, 0.95, size=n), rng.uniform(0.5, 0.8, size=n)),
        np.where(correct, rng.uniform(0.95, 0.99, size=n), rng.uniform(0.9, 0.99, size=n)),
    ).astype(np.float64)
    mats = ModalityAwareTS().fit(confidence, correct, modality_a)
    assert set(mats.temperatures.keys()) == {"panoramic", "cephalometric"}
    assert mats.temperatures["cephalometric"] != mats.temperatures["panoramic"]


def test_pick_threshold_meets_targets_when_feasible() -> None:
    rng = np.random.default_rng(7)
    n = 1000
    correct = rng.uniform(size=n) < 0.7
    confidence = np.where(correct, rng.uniform(0.6, 0.99, size=n), rng.uniform(0.0, 0.4, size=n))
    tau = pick_threshold(confidence, ~correct, target_capture=0.85, referral_cap=0.30)
    outcome = decide(confidence, tau)
    rr = outcome.referred.mean()
    ecr = (outcome.referred & ~correct).sum() / max((~correct).sum(), 1)
    assert ecr >= 0.85 - 1e-6
    assert rr <= 0.30 + 1e-6


def test_confidence_weighted_ensemble_breaks_ties_by_weight(rng: np.random.Generator) -> None:
    predictions = np.array([[0, 1, 2, 0], [1, 1, 0, 2]], dtype=np.int64)
    confidence = np.array([[0.9, 0.1, 0.4, 0.7], [0.1, 0.8, 0.6, 0.2]], dtype=np.float64)
    out = confidence_weighted_ensemble(predictions, confidence, n_classes=3)
    assert out.prediction[0] == 0
    assert out.prediction[1] == 1
    assert out.prediction[2] == 0
    assert out.prediction[3] == 0


def test_platt_calibrator_returns_probabilities_in_unit_interval(rng: np.random.Generator) -> None:
    confidence, correct = _overconfident_batch(rng)
    calibrator = PlattCalibrator().fit(confidence, correct)
    transformed = calibrator.transform(confidence)
    assert transformed.min() >= 0.0 - 1e-9
    assert transformed.max() <= 1.0 + 1e-9


def test_reliability_curve_has_expected_shape(rng: np.random.Generator) -> None:
    confidence, correct = _overconfident_batch(rng)
    curve = build_reliability_curve(confidence, correct, n_bins=15)
    assert curve.bin_centres.shape == (15,)
    assert curve.bin_accuracy.shape == (15,)


def test_fit_global_temperature_returns_one_when_perfectly_calibrated() -> None:
    centres = np.linspace(0.05, 0.95, 19, dtype=np.float64)
    confidence_list: list[float] = []
    correct_list: list[bool] = []
    for c in centres:
        block = 200
        confidence_list.extend([float(c)] * block)
        n_correct = int(round(c * block))
        correct_list.extend([True] * n_correct + [False] * (block - n_correct))
    temp = fit_global_temperature(
        np.array(confidence_list, dtype=np.float64),
        np.array(correct_list, dtype=bool),
    )
    assert abs(temp - 1.0) < 0.15
