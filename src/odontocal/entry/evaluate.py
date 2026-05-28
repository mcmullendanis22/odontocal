from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from odontocal.base.logging import configure_root_logger, get_logger
from odontocal.base.serialization import load_yaml, read_jsonl
from odontocal.judge.quality.calibration import expected_calibration_error
from odontocal.judge.quality.clinical import error_capture_rate, net_benefit, referral_rate
from odontocal.model.confidence.ensemble import confidence_weighted_ensemble
from odontocal.model.confidence.referral import decide
from odontocal.model.confidence.temperature import ModalityAwareTS
from odontocal.publish.tables.modality import ModalityTableEval
from odontocal.publish.tables.primary import PrimaryTableEval
from odontocal.publish.tables.subgroup import SubgroupTableEval


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="odc-evaluate", description="evaluate the calibrated ensemble"
    )
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser


def _load_inputs(
    predictions_path: Path,
    ground_truth_path: Path,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
]:
    pred_rows = list(read_jsonl(predictions_path))
    gt_rows = list(read_jsonl(ground_truth_path))
    gt_by_case = {str(r["case_id"]): r for r in gt_rows}
    cases: list[str] = sorted({str(r["case_id"]) for r in pred_rows})
    models: list[str] = sorted({str(r["model"]) for r in pred_rows})
    case_index = {c: i for i, c in enumerate(cases)}
    n_models = len(models)
    n_cases = len(cases)
    predictions = np.zeros((n_models, n_cases), dtype=np.int64)
    confidence = np.zeros((n_models, n_cases), dtype=np.float64)
    seen = np.zeros((n_models, n_cases), dtype=bool)
    for r in pred_rows:
        j = models.index(str(r["model"]))
        i = case_index[str(r["case_id"])]
        predictions[j, i] = int(r["prediction"])
        confidence[j, i] = float(r["confidence"])
        seen[j, i] = True
    if not seen.all():
        raise ValueError("predictions are not dense across models × cases")
    targets = np.array([int(gt_by_case[c]["label"]) for c in cases], dtype=np.int64)
    modality = np.array([str(gt_by_case[c]["modality"]) for c in cases], dtype=np.str_)
    dataset_tag = np.array([str(gt_by_case[c]["dataset_tag"]) for c in cases], dtype=np.str_)
    split = np.array([str(gt_by_case[c]["split"]) for c in cases], dtype=np.str_)
    calibration_mask = split == "calibration"
    return predictions, confidence, modality, dataset_tag, targets, calibration_mask, models


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    configure_root_logger()
    log = get_logger("odontocal.entry.evaluate")

    experiment = load_yaml(args.experiment)
    n_classes = int(experiment.get("n_classes", 5))
    referral_cap = float(experiment.get("referral_cap", 0.20))
    target_capture = float(experiment.get("target_capture", 0.85))
    tau_override = experiment.get("tau", None)

    predictions, confidence, modality, dataset_tag, targets, calibration_mask, models = (
        _load_inputs(args.predictions, args.ground_truth)
    )
    primary = PrimaryTableEval(
        n_classes=n_classes, target_capture=target_capture, referral_cap=referral_cap
    )
    primary_rows, fitted_tau = primary.evaluate(
        predictions, confidence, modality, targets, dataset_tag, calibration_mask
    )
    tau = float(tau_override) if tau_override is not None else fitted_tau

    calibrated = np.empty_like(confidence)
    scaler = ModalityAwareTS()
    for j in range(predictions.shape[0]):
        correct_member = predictions[j][calibration_mask] == targets[calibration_mask]
        scaler.fit(
            confidence[j][calibration_mask].astype(np.float64),
            correct_member,
            modality[calibration_mask].astype(np.str_),
        )
        calibrated[j] = scaler.transform(confidence[j].astype(np.float64), modality.astype(np.str_))
    ensemble = confidence_weighted_ensemble(predictions, calibrated, n_classes=n_classes)

    is_error = ensemble.prediction != targets
    outcome = decide(ensemble.confidence, tau)
    pooled_ece = expected_calibration_error(ensemble.confidence.astype(np.float64), ~is_error)
    pooled_nb = net_benefit(ensemble.prediction != 0, targets != 0, 0.5)
    pooled_ecr = error_capture_rate(outcome.referred, is_error)
    pooled_rr = referral_rate(outcome.referred)

    modality_eval = ModalityTableEval()
    modality_rows = modality_eval.evaluate(
        ensemble.confidence.astype(np.float64), ~is_error, modality, calibration_mask
    )
    subgroup_eval = SubgroupTableEval(tau=tau)
    subgroup_rows = subgroup_eval.evaluate(
        ensemble.confidence.astype(np.float64),
        ~is_error,
        targets != 0,
        {"modality": modality, "dataset": dataset_tag},
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "tau": tau,
        "models": models,
        "pooled": {
            "ECE": pooled_ece,
            "NB": pooled_nb,
            "ECR": pooled_ecr,
            "RR": pooled_rr,
        },
        "primary": [
            {
                "dataset": r.dataset,
                "AUROC": r.auroc,
                "ECE": r.ece,
                "NB": r.nb,
                "ECR": r.ecr,
                "RR": r.rr,
                "Acc": r.accuracy,
            }
            for r in primary_rows
        ],
        "modality": [
            {
                "modality": r.modality,
                "ECE_uncalibrated": r.ece_uncalibrated,
                "ECE_global_ts": r.ece_global_ts,
                "ECE_mats": r.ece_mats,
            }
            for r in modality_rows
        ],
        "subgroup": [
            {
                "stratification": r.stratification,
                "subgroup": r.subgroup,
                "AUC": r.auc,
                "ECE": r.ece,
                "ECR": r.ecr,
                "RR": r.rr,
            }
            for r in subgroup_rows
        ],
    }
    args.out.write_text(json.dumps(payload, indent=2))
    log.info("evaluation written to %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
