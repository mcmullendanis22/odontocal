from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from odontocal.judge.evidence.agreement import cohens_kappa


@dataclass(frozen=True)
class StratifiedRow:
    category: str
    n: int
    concordance: float
    kappa: float
    clinical_baseline: float
    delta_vs_clinical: float


class StratifiedSubsetEval:
    def __init__(self, n_classes: int, sample_size: int = 200, seed: int = 0) -> None:
        if sample_size <= 0:
            raise ValueError("sample_size must be positive")
        if n_classes < 2:
            raise ValueError("n_classes must be at least 2")
        self.n_classes = n_classes
        self.sample_size = sample_size
        self.seed = seed

    def draw_indices(
        self,
        categories: NDArray[np.str_],
        targets: NDArray[np.int64],
    ) -> NDArray[np.int64]:
        if categories.shape != targets.shape:
            raise ValueError("categories and targets must align")
        rng = np.random.default_rng(self.seed)
        unique_cats = np.unique(categories)
        per_cat = max(self.sample_size // max(len(unique_cats), 1), 1)
        chosen: list[NDArray[np.int64]] = []
        for cat in unique_cats:
            members = np.flatnonzero(categories == cat).astype(np.int64)
            if members.size == 0:
                continue
            chosen.append(rng.choice(members, size=min(per_cat, members.size), replace=False))
        return np.concatenate(chosen)

    def evaluate(
        self,
        predictions: NDArray[np.int64],
        targets: NDArray[np.int64],
        categories: NDArray[np.str_],
    ) -> tuple[list[StratifiedRow], StratifiedRow]:
        rows: list[StratifiedRow] = []
        total_n = 0
        total_correct = 0
        total_baseline_correct = 0
        agreed_targets: list[int] = []
        agreed_predictions: list[int] = []

        majority_class = int(np.bincount(targets, minlength=self.n_classes).argmax())
        for cat in np.unique(categories):
            mask = categories == cat
            n = int(mask.sum())
            if n == 0:
                continue
            correct = int((predictions[mask] == targets[mask]).sum())
            baseline = int((targets[mask] == majority_class).sum())
            kappa = cohens_kappa(
                predictions[mask].astype(np.int64),
                targets[mask].astype(np.int64),
                n_classes=self.n_classes,
            )
            concordance = correct / n
            rows.append(
                StratifiedRow(
                    category=str(cat),
                    n=n,
                    concordance=concordance,
                    kappa=kappa,
                    clinical_baseline=baseline / n,
                    delta_vs_clinical=concordance - baseline / n,
                )
            )
            total_n += n
            total_correct += correct
            total_baseline_correct += baseline
            agreed_targets.extend(targets[mask].astype(int).tolist())
            agreed_predictions.extend(predictions[mask].astype(int).tolist())

        if total_n == 0:
            overall_kappa = 0.0
            overall_concordance = 0.0
            overall_baseline = 0.0
        else:
            overall_kappa = cohens_kappa(
                np.asarray(agreed_predictions, dtype=np.int64),
                np.asarray(agreed_targets, dtype=np.int64),
                n_classes=self.n_classes,
            )
            overall_concordance = total_correct / total_n
            overall_baseline = total_baseline_correct / total_n
        overall = StratifiedRow(
            category="overall",
            n=total_n,
            concordance=overall_concordance,
            kappa=overall_kappa,
            clinical_baseline=overall_baseline,
            delta_vs_clinical=overall_concordance - overall_baseline,
        )
        return rows, overall

    @staticmethod
    def to_dataframe(rows: list[StratifiedRow], overall: StratifiedRow) -> pd.DataFrame:
        all_rows = [*rows, overall]
        return pd.DataFrame(
            [
                {
                    "category": r.category,
                    "N": r.n,
                    "Concordance (%)": round(100.0 * r.concordance, 1),
                    "κ": round(r.kappa, 3),
                    "Clinical Baseline (%)": round(100.0 * r.clinical_baseline, 1),
                    "Δ vs Clinical (pp)": round(100.0 * r.delta_vs_clinical, 1),
                }
                for r in all_rows
            ]
        )
