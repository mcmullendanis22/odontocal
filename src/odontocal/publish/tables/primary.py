from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from odontocal.judge.quality.calibration import expected_calibration_error
from odontocal.judge.quality.clinical import error_capture_rate, net_benefit, referral_rate
from odontocal.judge.quality.discrimination import accuracy, auc_roc
from odontocal.model.confidence.ensemble import confidence_weighted_ensemble
from odontocal.model.confidence.referral import decide, pick_threshold
from odontocal.model.confidence.temperature import ModalityAwareTS


@dataclass(frozen=True)
class PrimaryRow:
    dataset: str
    auroc: float
    ece: float
    nb: float
    ecr: float
    rr: float
    accuracy: float


class PrimaryTableEval:
    def __init__(
        self,
        n_classes: int,
        target_capture: float = 0.85,
        referral_cap: float = 0.20,
        n_bins: int = 15,
    ) -> None:
        if n_classes < 2:
            raise ValueError("n_classes must be at least 2")
        self.n_classes = n_classes
        self.target_capture = target_capture
        self.referral_cap = referral_cap
        self.n_bins = n_bins

    def evaluate(
        self,
        member_predictions: NDArray[np.int64],
        member_confidence: NDArray[np.float64],
        modality: NDArray[np.str_],
        targets: NDArray[np.int64],
        dataset_tag: NDArray[np.str_],
        calibration_mask: NDArray[np.bool_],
        decision_threshold_positive: float = 0.5,
    ) -> tuple[list[PrimaryRow], float]:
        if member_predictions.shape != member_confidence.shape:
            raise ValueError("member_predictions and member_confidence must align")
        if member_predictions.shape[1] != targets.shape[0]:
            raise ValueError("sample axis of members must match targets")

        scaler = ModalityAwareTS()
        n_members = member_predictions.shape[0]
        calibrated = np.empty_like(member_confidence, dtype=np.float64)
        for j in range(n_members):
            correct_member = member_predictions[j][calibration_mask] == targets[calibration_mask]
            scaler.fit(
                member_confidence[j][calibration_mask].astype(np.float64),
                correct_member,
                modality[calibration_mask].astype(np.str_),
            )
            calibrated[j] = scaler.transform(
                member_confidence[j].astype(np.float64), modality.astype(np.str_)
            )

        ensemble = confidence_weighted_ensemble(
            member_predictions, calibrated, n_classes=self.n_classes
        )
        ens_correct = ensemble.prediction == targets
        is_error = ~ens_correct

        tau = pick_threshold(
            ensemble.confidence[calibration_mask],
            is_error[calibration_mask],
            target_capture=self.target_capture,
            referral_cap=self.referral_cap,
        )

        rows: list[PrimaryRow] = []
        for tag in np.unique(dataset_tag):
            mask = dataset_tag == tag
            outcome = decide(ensemble.confidence[mask], tau)
            true_positive = targets[mask] != 0
            predicted_positive = ensemble.prediction[mask] != 0
            scores = ensemble.confidence[mask] * np.where(predicted_positive, 1.0, -1.0)
            rows.append(
                PrimaryRow(
                    dataset=str(tag),
                    auroc=auc_roc(scores.astype(np.float64), true_positive),
                    ece=expected_calibration_error(
                        ensemble.confidence[mask].astype(np.float64),
                        ens_correct[mask],
                        n_bins=self.n_bins,
                    ),
                    nb=net_benefit(predicted_positive, true_positive, decision_threshold_positive),
                    ecr=error_capture_rate(outcome.referred, is_error[mask]),
                    rr=referral_rate(outcome.referred),
                    accuracy=accuracy(ensemble.prediction[mask], targets[mask]),
                )
            )
        return rows, float(tau)

    @staticmethod
    def to_dataframe(rows: list[PrimaryRow]) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "dataset": r.dataset,
                    "AUROC": round(r.auroc, 4),
                    "ECE": round(r.ece, 4),
                    "NB": round(r.nb, 4),
                    "ECR": round(r.ecr, 4),
                    "RR": round(r.rr, 4),
                    "Acc": round(r.accuracy, 4),
                }
                for r in rows
            ]
        )
