from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class RandomStateSnapshot:
    seed: int
    python_state: tuple[Any, ...]
    numpy_state: dict[str, Any]
    torch_state: torch.Tensor
    cuda_states: list[torch.Tensor]


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def snapshot_random_state(seed: int) -> RandomStateSnapshot:
    cuda_states: list[torch.Tensor] = []
    if torch.cuda.is_available():
        cuda_states = [torch.cuda.get_rng_state(device=i) for i in range(torch.cuda.device_count())]
    return RandomStateSnapshot(
        seed=seed,
        python_state=random.getstate(),
        numpy_state=np.random.get_state(legacy=False),
        torch_state=torch.get_rng_state(),
        cuda_states=cuda_states,
    )
