from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from odontocal.base.logging import get_logger
from odontocal.operate.checkpoint import save_checkpoint_atomic
from odontocal.operate.seeding import set_seed, snapshot_random_state


@dataclass(frozen=True)
class CNNTrainConfig:
    lr: float
    weight_decay: float
    epochs: int
    batch_size: int
    seed: int
    precision: str = "bfloat16"
    grad_accum: int = 1
    ckpt_dir: Path = Path("checkpoints")
    log_every: int = 50


@dataclass(frozen=True)
class StepResult:
    step: int
    loss: float


class CNNTrainer:
    def __init__(self, model: nn.Module, cfg: CNNTrainConfig, device: torch.device) -> None:
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.optimizer = AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.scheduler: CosineAnnealingLR | None = None
        self.criterion = nn.CrossEntropyLoss()
        self.log = get_logger("odontocal.operate.trainer")
        self._step = 0

    def _autocast_dtype(self) -> torch.dtype:
        return torch.bfloat16 if self.cfg.precision == "bfloat16" else torch.float16

    def step(self, batch: dict[str, torch.Tensor]) -> StepResult:
        self.model.train()
        images = batch["image"].to(self.device, non_blocking=True)
        labels = batch["label"].to(self.device, non_blocking=True)
        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=self.device.type,
            dtype=self._autocast_dtype(),
            enabled=self.device.type == "cuda",
        ):
            logits = self.model(images)
            loss = self.criterion(logits, labels)
        loss.backward()
        self.optimizer.step()
        self._step += 1
        return StepResult(step=self._step, loss=float(loss.detach().cpu().item()))

    def epoch_iterator(self, loader: Iterable[dict[str, torch.Tensor]]) -> Iterator[StepResult]:
        for batch in loader:
            yield self.step(batch)

    def fit(
        self, loader: Iterable[dict[str, torch.Tensor]], steps_per_epoch: int
    ) -> list[StepResult]:
        set_seed(self.cfg.seed)
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=self.cfg.epochs * max(steps_per_epoch, 1)
        )
        history: list[StepResult] = []
        for epoch in range(self.cfg.epochs):
            for result in self.epoch_iterator(loader):
                history.append(result)
                if self.scheduler is not None:
                    self.scheduler.step()
                if result.step % self.cfg.log_every == 0:
                    self.log.info("epoch=%d step=%d loss=%.6f", epoch, result.step, result.loss)
            self._save(epoch=epoch)
        return history

    def _save(self, epoch: int) -> None:
        self.cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)
        save_checkpoint_atomic(
            self.cfg.ckpt_dir / f"epoch_{epoch:03d}.pt",
            model_state={k: v.detach().cpu() for k, v in self.model.state_dict().items()},
            optimizer_state=self.optimizer.state_dict(),
            scheduler_state=None if self.scheduler is None else self.scheduler.state_dict(),
            rng=snapshot_random_state(self.cfg.seed),
            epoch=epoch,
            step=self._step,
            extra={"precision": self.cfg.precision, "batch_size": self.cfg.batch_size},
        )
