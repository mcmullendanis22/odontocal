from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from odontocal.judge.quality.calibration import expected_calibration_error
from odontocal.judge.quality.clinical import error_capture_rate, referral_rate
from odontocal.judge.quality.discrimination import auc_roc
from odontocal.model.confidence.referral import decide


@dataclass(frozen=True)
class SubgroupRow:
    stratification: str
    subgroup: str
    auc: float
    ece: float
    ecr: float
    rr: float


class SubgroupTableEval:
    def __init__(self, tau: float = 0.55, n_bins: int = 15) -> None:
        self.tau = tau
        self.n_bins = n_bins

    def evaluate(
        self,
        confidence: NDArray[np.float64],
        correct: NDArray[np.bool_],
        true_positive: NDArray[np.bool_],
        stratifications: dict[str, NDArray[np.str_]],
    ) -> list[SubgroupRow]:
        rows: list[SubgroupRow] = []
        is_error = ~correct
        outcome = decide(confidence, self.tau)
        for strat_name, values in stratifications.items():
            for sub in np.unique(values):
                mask = values == sub
                rows.append(
                    SubgroupRow(
                        stratification=strat_name,
                        subgroup=str(sub),
                        auc=auc_roc(confidence[mask].astype(np.float64), true_positive[mask]),
                        ece=expected_calibration_error(
                            confidence[mask].astype(np.float64), correct[mask], n_bins=self.n_bins
                        ),
                        ecr=error_capture_rate(outcome.referred[mask], is_error[mask]),
                        rr=referral_rate(outcome.referred[mask]),
                    )
                )
        return rows

    @staticmethod
    def fairness_gap(rows: list[SubgroupRow], stratification: str) -> float:
        members = [r for r in rows if r.stratification == stratification]
        if len(members) < 2:
            return 0.0
        aucs = np.array([r.auc for r in members], dtype=np.float64)
        return float(aucs.max() - aucs.min())

    @staticmethod
    def to_dataframe(rows: list[SubgroupRow]) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "stratification": r.stratification,
                    "subgroup": r.subgroup,
                    "AUC": round(r.auc, 4),
                    "ECE": round(r.ece, 4),
                    "ECR": round(r.ecr, 4),
                    "RR": round(r.rr, 4),
                }
                for r in rows
            ]
        )
