from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from odontocal.base.logging import configure_root_logger, get_logger
from odontocal.base.serialization import read_jsonl, write_atomic
from odontocal.model.confidence.referral import pick_threshold
from odontocal.model.confidence.temperature import ModalityAwareTS


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="odc-fit-calibration", description="fit MATS per modality and pick τ"
    )
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target-capture", type=float, default=0.85)
    parser.add_argument("--referral-cap", type=float, default=0.20)
    return parser


def _index(rows: list[dict[str, object]]) -> dict[tuple[str, str], dict[str, object]]:
    return {(str(r["case_id"]), str(r["model"])): r for r in rows}


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    configure_root_logger()
    log = get_logger("odontocal.entry.fit_calibration")

    pred_rows = list(read_jsonl(args.predictions))
    gt_rows = list(read_jsonl(args.ground_truth))
    gt_by_case = {str(r["case_id"]): r for r in gt_rows}

    grouped: dict[str, list[dict[str, object]]] = {}
    for row in pred_rows:
        grouped.setdefault(str(row["model"]), []).append(row)

    temperatures_per_model: dict[str, dict[str, float]] = {}
    aggregated_confidence: list[float] = []
    aggregated_correct: list[bool] = []
    aggregated_modality: list[str] = []

    for model_name, rows in grouped.items():
        confs = np.array([float(r["confidence"]) for r in rows], dtype=np.float64)
        preds = np.array([int(r["prediction"]) for r in rows], dtype=np.int64)
        modality = np.array([str(r["modality"]) for r in rows], dtype=np.str_)
        targets = np.array(
            [int(gt_by_case[str(r["case_id"])]["label"]) for r in rows], dtype=np.int64
        )
        correct = preds == targets
        scaler = ModalityAwareTS().fit(confs, correct, modality)
        temperatures_per_model[model_name] = scaler.temperatures
        aggregated_confidence.extend(scaler.transform(confs, modality).tolist())
        aggregated_correct.extend(correct.tolist())
        aggregated_modality.extend(modality.tolist())

    confidence_arr = np.array(aggregated_confidence, dtype=np.float64)
    error_arr = ~np.array(aggregated_correct, dtype=bool)
    tau = pick_threshold(
        confidence_arr,
        error_arr,
        target_capture=args.target_capture,
        referral_cap=args.referral_cap,
    )

    payload = {
        "temperatures_per_model": temperatures_per_model,
        "tau": tau,
        "target_capture": args.target_capture,
        "referral_cap": args.referral_cap,
    }
    write_atomic(args.output, json.dumps(payload, indent=2).encode("utf-8"))
    log.info("fitted MATS for %d models · τ=%.4f → %s", len(grouped), tau, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
