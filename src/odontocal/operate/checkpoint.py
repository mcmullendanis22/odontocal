from __future__ import annotations

import io
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from odontocal.base.serialization import write_atomic
from odontocal.operate.seeding import RandomStateSnapshot


def save_checkpoint_atomic(
    path: str | os.PathLike[str],
    *,
    model_state: dict[str, torch.Tensor],
    optimizer_state: dict[str, Any],
    scheduler_state: dict[str, Any] | None,
    rng: RandomStateSnapshot,
    epoch: int,
    step: int,
    extra: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "model_state": model_state,
        "optimizer_state": optimizer_state,
        "scheduler_state": scheduler_state,
        "rng": asdict(rng),
        "epoch": epoch,
        "step": step,
        "extra": extra or {},
    }
    buf = io.BytesIO()
    torch.save(payload, buf)
    write_atomic(Path(path), buf.getvalue())


def load_checkpoint(path: str | os.PathLike[str], map_location: str = "cpu") -> dict[str, Any]:
    return torch.load(str(path), map_location=map_location)
