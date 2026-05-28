from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import nn


_ALLOWED = {"resnet50", "efficientnet_b4"}


@dataclass(frozen=True)
class CNNBaselineConfig:
    name: str
    n_classes: int
    pretrained: bool = True
    drop_rate: float = 0.0


def build_cnn_baseline(cfg: CNNBaselineConfig) -> nn.Module:
    if cfg.name not in _ALLOWED:
        raise ValueError(f"cnn baseline {cfg.name!r} is not in {_ALLOWED}")
    import timm

    return timm.create_model(
        cfg.name,
        pretrained=cfg.pretrained,
        num_classes=cfg.n_classes,
        drop_rate=cfg.drop_rate,
    )
