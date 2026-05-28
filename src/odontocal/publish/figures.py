from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from odontocal.model.confidence.reliability import build_reliability_curve


def reliability_diagram(
    confidence: NDArray[np.float64],
    correct: NDArray[np.bool_],
    output_path: Path,
    n_bins: int = 15,
    title: str = "Reliability",
) -> Path:
    curve = build_reliability_curve(confidence, correct, n_bins=n_bins)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", linewidth=1.0, color="grey")
    populated = curve.bin_counts > 0
    ax.bar(
        curve.bin_centres[populated], curve.bin_accuracy[populated], width=1.0 / n_bins, alpha=0.65
    )
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def decision_curve(
    predicted_positive: NDArray[np.bool_],
    truth_positive: NDArray[np.bool_],
    thresholds: NDArray[np.float64],
    output_path: Path,
    title: str = "Decision curve",
) -> Path:
    from odontocal.judge.quality.clinical import net_benefit

    values = np.array(
        [net_benefit(predicted_positive, truth_positive, float(p)) for p in thresholds],
        dtype=np.float64,
    )
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot(thresholds, values, linewidth=1.5)
    ax.axhline(0.0, color="grey", linewidth=0.8)
    ax.set_xlabel("Decision threshold p_t")
    ax.set_ylabel("Net benefit")
    ax.set_title(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
