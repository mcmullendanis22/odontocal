from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from odontocal.judge.quality.calibration import expected_calibration_error
from odontocal.judge.quality.discrimination import auc_roc


@dataclass(frozen=True)
class CrossDatasetRow:
    target_dataset: str
    auc: float
    delta_auc_vs_reference: float
    ece: float


class CrossDatasetTableEval:
    def __init__(self, reference: str = "mmoral", n_bins: int = 15) -> None:
        self.reference = reference
        self.n_bins = n_bins

    def evaluate(
        self,
        confidence: NDArray[np.float64],
        correct: NDArray[np.bool_],
        true_positive: NDArray[np.bool_],
        dataset_tag: NDArray[np.str_],
    ) -> list[CrossDatasetRow]:
        all_aucs: dict[str, float] = {}
        all_eces: dict[str, float] = {}
        for tag in np.unique(dataset_tag):
            mask = dataset_tag == tag
            score = confidence[mask].astype(np.float64)
            all_aucs[str(tag)] = auc_roc(score, true_positive[mask])
            all_eces[str(tag)] = expected_calibration_error(
                score, correct[mask], n_bins=self.n_bins
            )

        ref = next(iter(all_aucs)) if self.reference not in all_aucs else self.reference
        ref_auc = all_aucs[ref]

        rows: list[CrossDatasetRow] = []
        for tag, auc in sorted(all_aucs.items()):
            rows.append(
                CrossDatasetRow(
                    target_dataset=tag,
                    auc=auc,
                    delta_auc_vs_reference=auc - ref_auc,
                    ece=all_eces[tag],
                )
            )
        return rows

    @staticmethod
    def to_dataframe(rows: list[CrossDatasetRow]) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "target_dataset": r.target_dataset,
                    "AUC": round(r.auc, 4),
                    "ΔAUC (pp)": round(100.0 * r.delta_auc_vs_reference, 2),
                    "ECE": round(r.ece, 4),
                }
                for r in rows
            ]
        )
