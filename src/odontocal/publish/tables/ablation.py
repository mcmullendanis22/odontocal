from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from odontocal.judge.quality.calibration import expected_calibration_error
from odontocal.judge.quality.clinical import error_capture_rate, net_benefit, referral_rate
from odontocal.judge.quality.discrimination import auc_roc
from odontocal.model.confidence.ensemble import confidence_weighted_ensemble
from odontocal.model.confidence.referral import decide, pick_threshold
from odontocal.model.confidence.temperature import GlobalTS, ModalityAwareTS

_CONFIGURATIONS = (
    "no_calibration",
    "global_ts_only",
    "ensemble_only",
    "selective_referral_off",
    "full_system",
)


@dataclass(frozen=True)
class AblationRow:
    configuration: str
    auc: float
    ece: float
    nb: float
    ecr: float
    rr: float


class AblationTableEval:
    def __init__(
        self,
        n_classes: int,
        target_capture: float = 0.85,
        referral_cap: float = 0.20,
        n_bins: int = 15,
    ) -> None:
        self.n_classes = n_classes
        self.target_capture = target_capture
        self.referral_cap = referral_cap
        self.n_bins = n_bins

    def _calibrate(
        self,
        mode: str,
        member_predictions: NDArray[np.int64],
        member_confidence: NDArray[np.float64],
        modality: NDArray[np.str_],
        targets: NDArray[np.int64],
        calibration_mask: NDArray[np.bool_],
    ) -> NDArray[np.float64]:
        calibrated = np.empty_like(member_confidence, dtype=np.float64)
        for j in range(member_predictions.shape[0]):
            correct = member_predictions[j][calibration_mask] == targets[calibration_mask]
            if mode == "no_calibration":
                calibrated[j] = member_confidence[j]
            elif mode == "global_ts_only":
                cal = GlobalTS().fit(
                    member_confidence[j][calibration_mask].astype(np.float64), correct
                )
                calibrated[j] = cal.transform(member_confidence[j].astype(np.float64))
            else:
                cal = ModalityAwareTS().fit(
                    member_confidence[j][calibration_mask].astype(np.float64),
                    correct,
                    modality[calibration_mask].astype(np.str_),
                )
                calibrated[j] = cal.transform(
                    member_confidence[j].astype(np.float64), modality.astype(np.str_)
                )
        return calibrated

    def _aggregate(
        self,
        mode: str,
        member_predictions: NDArray[np.int64],
        calibrated: NDArray[np.float64],
    ) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        if mode == "ensemble_only":
            best_member = int(np.argmax(calibrated.mean(axis=1)))
            return member_predictions[best_member], calibrated[best_member]
        ensemble = confidence_weighted_ensemble(
            member_predictions, calibrated, n_classes=self.n_classes
        )
        return ensemble.prediction, ensemble.confidence

    def evaluate(
        self,
        member_predictions: NDArray[np.int64],
        member_confidence: NDArray[np.float64],
        modality: NDArray[np.str_],
        targets: NDArray[np.int64],
        calibration_mask: NDArray[np.bool_],
        decision_threshold_positive: float = 0.5,
    ) -> list[AblationRow]:
        rows: list[AblationRow] = []
        for mode in _CONFIGURATIONS:
            calibrated = self._calibrate(
                mode if mode != "selective_referral_off" else "full_system",
                member_predictions,
                member_confidence,
                modality,
                targets,
                calibration_mask,
            )
            pred, conf = self._aggregate(
                mode if mode != "selective_referral_off" else "full_system",
                member_predictions,
                calibrated,
            )
            correct = pred == targets
            is_error = ~correct
            tau = (
                0.0
                if mode == "selective_referral_off"
                else pick_threshold(
                    conf[calibration_mask],
                    is_error[calibration_mask],
                    target_capture=self.target_capture,
                    referral_cap=self.referral_cap,
                )
            )
            outcome = decide(conf, tau)
            true_positive = targets != 0
            predicted_positive = pred != 0
            rows.append(
                AblationRow(
                    configuration=mode,
                    auc=auc_roc(conf * np.where(predicted_positive, 1.0, -1.0), true_positive),
                    ece=expected_calibration_error(
                        conf.astype(np.float64), correct, n_bins=self.n_bins
                    ),
                    nb=net_benefit(predicted_positive, true_positive, decision_threshold_positive),
                    ecr=error_capture_rate(outcome.referred, is_error),
                    rr=referral_rate(outcome.referred),
                )
            )
        return rows

    @staticmethod
    def to_dataframe(rows: list[AblationRow]) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "configuration": r.configuration,
                    "AUC": round(r.auc, 4),
                    "ECE": round(r.ece, 4),
                    "NB": round(r.nb, 4),
                    "ECR": round(r.ecr, 4),
                    "RR": round(r.rr, 4),
                }
                for r in rows
            ]
        )
