from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from odontocal.publish.tables.modality import ModalityTableEval
from odontocal.publish.tables.primary import PrimaryTableEval
from odontocal.publish.tables.stratified import StratifiedSubsetEval
from odontocal.publish.tables.subgroup import SubgroupTableEval

_GOLDEN = Path(__file__).resolve().parent / "fixtures" / "evaluation_golden.json"


def _golden_inputs(rng: np.random.Generator) -> dict[str, np.ndarray]:
    n_members = 3
    n_samples = 120
    modality = np.array(
        ["panoramic", "periapical", "intraoral", "cephalometric"] * (n_samples // 4),
        dtype=np.str_,
    )
    dataset_tag = np.array(
        ["mmoral", "dentex", "denpar", "tufts", "intraoral_caries"] * (n_samples // 5),
        dtype=np.str_,
    )
    targets = rng.integers(0, 5, size=n_samples).astype(np.int64)
    predictions = np.stack(
        [
            np.where(
                rng.uniform(size=n_samples) < 0.55, targets, rng.integers(0, 5, size=n_samples)
            ).astype(np.int64)
            for _ in range(n_members)
        ]
    )
    confidence = rng.uniform(0.4, 0.95, size=(n_members, n_samples)).astype(np.float64)
    calibration_mask = np.zeros(n_samples, dtype=bool)
    calibration_mask[: n_samples // 3] = True
    return {
        "predictions": predictions,
        "confidence": confidence,
        "modality": modality,
        "targets": targets,
        "dataset_tag": dataset_tag,
        "calibration_mask": calibration_mask,
    }


@pytest.mark.regression
def test_primary_table_eval_emits_expected_shape() -> None:
    rng = np.random.default_rng(123)
    inputs = _golden_inputs(rng)
    pe = PrimaryTableEval(n_classes=5)
    rows, tau = pe.evaluate(
        inputs["predictions"],
        inputs["confidence"],
        inputs["modality"],
        inputs["targets"],
        inputs["dataset_tag"],
        inputs["calibration_mask"],
    )
    assert {r.dataset for r in rows} == set(np.unique(inputs["dataset_tag"]).tolist())
    assert 0.0 <= tau <= 1.0
    for r in rows:
        assert 0.0 <= r.ece <= 1.0
        assert 0.0 <= r.ecr <= 1.0
        assert 0.0 <= r.rr <= 1.0


@pytest.mark.regression
def test_modality_table_shows_mats_ece_dominates_or_matches_global() -> None:
    rng = np.random.default_rng(2025)
    n = 800
    modality = np.where(np.arange(n) < n // 2, "panoramic", "cephalometric").astype(np.str_)
    correct = rng.uniform(size=n) < np.where(modality == "panoramic", 0.7, 0.4)
    confidence = np.where(
        modality == "panoramic",
        np.where(correct, rng.uniform(0.7, 0.95, size=n), rng.uniform(0.55, 0.85, size=n)),
        np.where(correct, rng.uniform(0.9, 0.99, size=n), rng.uniform(0.85, 0.99, size=n)),
    ).astype(np.float64)
    calibration_mask = np.zeros(n, dtype=bool)
    calibration_mask[: n // 2] = True
    rows = ModalityTableEval().evaluate(confidence, correct, modality, calibration_mask)
    for r in rows:
        assert r.ece_mats <= r.ece_uncalibrated + 1e-6


@pytest.mark.regression
def test_subgroup_eval_produces_one_row_per_subgroup_value(rng: np.random.Generator) -> None:
    n = 200
    confidence = rng.uniform(0.4, 0.95, size=n).astype(np.float64)
    correct = rng.uniform(size=n) < 0.7
    truth_positive = rng.uniform(size=n) < 0.4
    modality = np.array(["panoramic", "periapical"] * (n // 2), dtype=np.str_)
    dataset = np.array(["mmoral", "dentex", "denpar", "tufts"] * (n // 4), dtype=np.str_)
    rows = SubgroupTableEval().evaluate(
        confidence, correct, truth_positive, {"modality": modality, "dataset": dataset}
    )
    n_rows = len({(r.stratification, r.subgroup) for r in rows})
    assert n_rows == 2 + 4


@pytest.mark.regression
def test_stratified_eval_overall_row_aggregates_concordance() -> None:
    rng = np.random.default_rng(99)
    n = 240
    categories = np.array(["caries", "periodontal", "periapical", "opmd"] * (n // 4), dtype=np.str_)
    targets = rng.integers(0, 5, size=n).astype(np.int64)
    predictions = np.where(rng.uniform(size=n) < 0.7, targets, rng.integers(0, 5, size=n)).astype(
        np.int64
    )
    rows, overall = StratifiedSubsetEval(n_classes=5, sample_size=n).evaluate(
        predictions, targets, categories
    )
    assert overall.n == sum(r.n for r in rows)


def test_golden_fixture_exists_when_present() -> None:
    if not _GOLDEN.exists():
        return
    payload = json.loads(_GOLDEN.read_text())
    assert "metric" in payload
