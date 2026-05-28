from __future__ import annotations

import os

import torch
import torch.distributed as dist


def world_size_or_one() -> int:
    if dist.is_available() and dist.is_initialized():
        return int(dist.get_world_size())
    return int(os.environ.get("WORLD_SIZE", "1"))


def init_distributed(backend: str = "nccl") -> bool:
    if not dist.is_available():
        return False
    if dist.is_initialized():
        return True
    if "WORLD_SIZE" not in os.environ or os.environ.get("WORLD_SIZE") == "1":
        return False
    dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    return True
