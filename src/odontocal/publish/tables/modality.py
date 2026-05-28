from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from odontocal.judge.quality.calibration import expected_calibration_error
from odontocal.model.confidence.temperature import GlobalTS, ModalityAwareTS


@dataclass(frozen=True)
class ModalityRow:
    modality: str
    ece_uncalibrated: float
    ece_global_ts: float
    ece_mats: float


class ModalityTableEval:
    def __init__(self, n_bins: int = 15) -> None:
        self.n_bins = n_bins

    def evaluate(
        self,
        confidence: NDArray[np.float64],
        correct: NDArray[np.bool_],
        modality: NDArray[np.str_],
        calibration_mask: NDArray[np.bool_],
    ) -> list[ModalityRow]:
        global_ts = GlobalTS().fit(confidence[calibration_mask], correct[calibration_mask])
        mats = ModalityAwareTS().fit(
            confidence[calibration_mask], correct[calibration_mask], modality[calibration_mask]
        )
        global_calibrated = global_ts.transform(confidence)
        mats_calibrated = mats.transform(confidence, modality)
        out: list[ModalityRow] = []
        for tag in np.unique(modality):
            mask = modality == tag
            out.append(
                ModalityRow(
                    modality=str(tag),
                    ece_uncalibrated=expected_calibration_error(
                        confidence[mask].astype(np.float64), correct[mask], n_bins=self.n_bins
                    ),
                    ece_global_ts=expected_calibration_error(
                        global_calibrated[mask].astype(np.float64),
                        correct[mask],
                        n_bins=self.n_bins,
                    ),
                    ece_mats=expected_calibration_error(
                        mats_calibrated[mask].astype(np.float64), correct[mask], n_bins=self.n_bins
                    ),
                )
            )
        return out

    @staticmethod
    def to_dataframe(rows: list[ModalityRow]) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "modality": r.modality,
                    "Uncalibrated ECE": round(r.ece_uncalibrated, 4),
                    "Global TS ECE": round(r.ece_global_ts, 4),
                    "MATS ECE": round(r.ece_mats, 4),
                    "Δ MATS (%)": round(
                        (
                            -100.0 * (r.ece_uncalibrated - r.ece_mats) / r.ece_uncalibrated
                            if r.ece_uncalibrated > 0
                            else 0.0
                        ),
                        2,
                    ),
                }
                for r in rows
            ]
        )
