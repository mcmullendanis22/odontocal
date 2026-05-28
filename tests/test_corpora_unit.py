from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from odontocal.observe.corpora import DenPARPeriapical, DentexPanoramic, IntraoralCaries, MMOralVQA
from odontocal.observe.prompting import structured_triage_prompt
from odontocal.observe.routing import MODALITY_OF_DATASET, Modality, route_dataset
from odontocal.observe.splits import build_official_or_stratified_split


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def test_router_maps_each_dataset_to_a_known_modality() -> None:
    assert route_dataset("dentex") is Modality.PANORAMIC
    assert route_dataset("denpar") is Modality.PERIAPICAL
    assert route_dataset("intraoral_caries") is Modality.INTRAORAL
    assert route_dataset("mmoral_cephalometric") is Modality.CEPHALOMETRIC
    assert set(MODALITY_OF_DATASET.values()) == set(Modality)


def test_unknown_dataset_raises() -> None:
    try:
        route_dataset("bogus_dataset")
    except KeyError:
        return
    raise AssertionError("KeyError not raised")


def test_structured_prompt_has_required_fields() -> None:
    prompt = structured_triage_prompt(Modality.PANORAMIC, "history of mild gingivitis")
    assert "panoramic" in prompt.user
    assert "JSON" in prompt.user
    assert "assessment" in prompt.response_schema
    assert "confidence" in prompt.response_schema


def test_stratified_split_is_deterministic() -> None:
    labels = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2], dtype=np.int64)
    a = build_official_or_stratified_split(labels, seed=42)
    b = build_official_or_stratified_split(labels, seed=42)
    assert np.array_equal(a.train, b.train)
    assert np.array_equal(a.val, b.val)
    assert np.array_equal(a.test, b.test)
    total = a.train.size + a.val.size + a.test.size
    assert total == labels.size


def test_mmoral_adapter_schema(tmp_path: Path) -> None:
    root = tmp_path / "mmoral"
    rows = [
        {"image_path": str(tmp_path / "a.png"), "label": 0, "case_id": "c0"},
        {"image_path": str(tmp_path / "b.png"), "label": 2, "case_id": "c1"},
    ]
    _write_manifest(root / "manifests" / "mmoral_panoramic_test.jsonl", rows)
    adapter = MMOralVQA.from_root(root, split="test", modality=Modality.PANORAMIC)
    items = list(adapter)
    assert len(items) == 2
    sample = items[0]
    assert set(sample.keys()) == {"image_path", "label", "modality", "dataset_tag", "case_id"}
    assert sample["modality"] == "panoramic"
    assert sample["dataset_tag"] == "mmoral_panoramic"


def test_dentex_adapter_schema(tmp_path: Path) -> None:
    rows = [
        {"image_path": str(tmp_path / "x.png"), "label": 1, "case_id": "x0"},
    ]
    _write_manifest(tmp_path / "dentex" / "test.jsonl", rows)
    adapter = DentexPanoramic(root=tmp_path, split="test")
    items = adapter.cases()
    assert items[0]["dataset_tag"] == "dentex"
    assert items[0]["modality"] == "panoramic"


def test_denpar_and_intraoral_adapters(tmp_path: Path) -> None:
    _write_manifest(
        tmp_path / "denpar" / "val.jsonl",
        [{"image_path": str(tmp_path / "p.png"), "label": 3, "case_id": "d0"}],
    )
    den = DenPARPeriapical(root=tmp_path, split="val")
    assert next(iter(den))["modality"] == "periapical"

    _write_manifest(
        tmp_path / "intraoral_caries" / "train.jsonl",
        [{"image_path": str(tmp_path / "i.png"), "label": 1, "case_id": "i0"}],
    )
    ic = IntraoralCaries.build(root=tmp_path, split="train")
    assert next(iter(ic))["dataset_tag"] == "intraoral_caries"
