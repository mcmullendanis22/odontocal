from __future__ import annotations

import numpy as np

from odontocal.judge.evidence.agreement import cohens_kappa
from odontocal.judge.evidence.bootstrap import stratified_bootstrap_ci
from odontocal.judge.evidence.corrections import benjamini_hochberg, holm_bonferroni
from odontocal.judge.evidence.delong import delong_paired_auc_test


def test_holm_bonferroni_matches_known_example() -> None:
    p_values = np.array([0.005, 0.011, 0.02, 0.04, 0.13], dtype=np.float64)
    reject = holm_bonferroni(p_values, alpha=0.05)
    assert reject.tolist() == [True, True, False, False, False]


def test_benjamini_hochberg_matches_known_example() -> None:
    p_values = np.array([0.0001, 0.001, 0.01, 0.04, 0.07], dtype=np.float64)
    reject = benjamini_hochberg(p_values, alpha=0.05)
    assert reject.tolist() == [True, True, True, True, False]


def test_holm_bonferroni_more_conservative_than_bh() -> None:
    p_values = np.array([0.001, 0.012, 0.025, 0.04, 0.06, 0.09], dtype=np.float64)
    a = holm_bonferroni(p_values, alpha=0.05)
    b = benjamini_hochberg(p_values, alpha=0.05)
    assert int(a.sum()) <= int(b.sum())


def test_delong_two_identical_models_give_near_zero_z() -> None:
    rng = np.random.default_rng(1)
    truth = rng.integers(0, 2, size=500).astype(bool)
    base = rng.normal(0.0, 1.0, size=500).astype(np.float64)
    scores_a = base.copy() + 0.7 * truth.astype(np.float64)
    scores_b = scores_a.copy()
    result = delong_paired_auc_test(scores_a, scores_b, truth)
    assert abs(result.z) < 1e-6
    assert abs(result.auc_a - result.auc_b) < 1e-9


def test_delong_detects_real_gap() -> None:
    rng = np.random.default_rng(0)
    n = 800
    truth = np.array([True] * (n // 2) + [False] * (n // 2), dtype=bool)
    rng.shuffle(truth)
    scores_a = (truth.astype(np.float64) * 1.5) + rng.normal(0.0, 1.0, size=n)
    scores_b = rng.normal(0.0, 1.0, size=n)
    result = delong_paired_auc_test(scores_a, scores_b, truth)
    assert result.auc_a > result.auc_b
    assert result.p_value < 1e-3


def test_cohens_kappa_recovers_perfect_agreement() -> None:
    a = np.array([0, 1, 2, 1, 0, 2], dtype=np.int64)
    assert abs(cohens_kappa(a, a, n_classes=3) - 1.0) < 1e-9


def test_cohens_kappa_zero_for_independent_predictions() -> None:
    rng = np.random.default_rng(11)
    n = 2000
    a = rng.integers(0, 3, size=n).astype(np.int64)
    b = rng.integers(0, 3, size=n).astype(np.int64)
    assert abs(cohens_kappa(a, b, n_classes=3)) < 0.05


def test_bootstrap_ci_contains_point_estimate(rng: np.random.Generator) -> None:
    arr = rng.normal(0.5, 0.1, size=200).astype(np.float64)

    def fn(idx: np.ndarray) -> float:
        return float(arr[idx].mean())

    ci = stratified_bootstrap_ci(
        fn,
        n_samples=arr.size,
        strata=np.zeros(arr.size, dtype=np.int64),
        n_resamples=400,
        alpha=0.05,
        seed=0,
    )
    assert ci.low <= ci.point <= ci.high
