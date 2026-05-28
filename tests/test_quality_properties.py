from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from odontocal.judge.quality.calibration import (
    brier_score,
    expected_calibration_error,
    maximum_calibration_error,
    reliability_bins,
)
from odontocal.judge.quality.clinical import error_capture_rate, net_benefit, referral_rate
from odontocal.judge.quality.discrimination import auc_roc, f1_score

_N_STRATEGY = st.integers(min_value=20, max_value=300)


@st.composite
def _aligned_arrays(draw, n=_N_STRATEGY):
    size = draw(n)
    rng = np.random.default_rng(draw(st.integers(min_value=0, max_value=2**31 - 1)))
    confidence = rng.uniform(0.01, 0.99, size=size).astype(np.float64)
    correct = rng.integers(0, 2, size=size).astype(bool)
    return confidence, correct


@given(data=_aligned_arrays())
@settings(max_examples=40, deadline=None)
def test_ece_is_within_unit_interval(data) -> None:
    confidence, correct = data
    value = expected_calibration_error(confidence, correct)
    assert 0.0 <= value <= 1.0


def test_perfectly_calibrated_input_has_zero_ece() -> None:
    edges = np.linspace(0.0, 1.0, 16, dtype=np.float64)
    centres = 0.5 * (edges[:-1] + edges[1:])
    confidence_list: list[float] = []
    correct_list: list[bool] = []
    for c in centres:
        block = 100
        confidence_list.extend([float(c)] * block)
        n_correct = int(round(c * block))
        correct_list.extend([True] * n_correct + [False] * (block - n_correct))
    confidence = np.array(confidence_list, dtype=np.float64)
    correct = np.array(correct_list, dtype=bool)
    assert expected_calibration_error(confidence, correct, n_bins=15) < 0.05


def test_brier_non_negative(rng: np.random.Generator) -> None:
    probabilities = rng.uniform(0.05, 0.95, size=128).astype(np.float64)
    targets = rng.integers(0, 2, size=128).astype(np.int64)
    assert brier_score(probabilities, targets) >= 0.0


def test_maximum_calibration_error_geq_expected(rng: np.random.Generator) -> None:
    confidence = rng.uniform(0.0, 1.0, size=400).astype(np.float64)
    correct = rng.integers(0, 2, size=400).astype(bool)
    ece = expected_calibration_error(confidence, correct)
    mce = maximum_calibration_error(confidence, correct)
    assert mce + 1e-9 >= ece


def test_net_benefit_monotone_in_true_positives() -> None:
    truth = np.array([True, True, True, False, False, False], dtype=bool)
    pred_low = np.array([False, False, True, False, False, False], dtype=bool)
    pred_high = np.array([True, True, True, False, False, False], dtype=bool)
    assert net_benefit(pred_high, truth, 0.5) >= net_benefit(pred_low, truth, 0.5)


def test_referral_rate_and_error_capture_bounds(rng: np.random.Generator) -> None:
    refers = rng.integers(0, 2, size=64).astype(bool)
    errors = rng.integers(0, 2, size=64).astype(bool)
    assert 0.0 <= referral_rate(refers) <= 1.0
    assert 0.0 <= error_capture_rate(refers, errors) <= 1.0


def test_auc_known_separable_case() -> None:
    scores = np.array([0.1, 0.2, 0.7, 0.8, 0.9], dtype=np.float64)
    truth = np.array([False, False, True, True, True], dtype=bool)
    assert auc_roc(scores, truth) == 1.0


def test_f1_zero_when_no_overlap() -> None:
    pred = np.array([True, True, False, False], dtype=bool)
    truth = np.array([False, False, True, True], dtype=bool)
    assert f1_score(pred, truth) == 0.0


def test_reliability_bins_summary_counts_match_input() -> None:
    confidence = np.array([0.05, 0.25, 0.55, 0.85], dtype=np.float64)
    correct = np.array([False, True, False, True], dtype=bool)
    summary = reliability_bins(confidence, correct, n_bins=5)
    assert int(summary.bin_counts.sum()) == 4
