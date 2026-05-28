from __future__ import annotations

from pathlib import Path

import pytest
import torch

from odontocal.model.backbones.convnet import CNNBaselineConfig, build_cnn_baseline
from odontocal.operate import CNNTrainConfig, CNNTrainer, load_checkpoint, set_seed


@pytest.mark.integration
def test_two_step_training_smoke(tmp_path: Path) -> None:
    set_seed(7)
    model = build_cnn_baseline(CNNBaselineConfig(name="resnet50", n_classes=5, pretrained=False))
    cfg = CNNTrainConfig(
        lr=1e-3,
        weight_decay=1e-4,
        epochs=1,
        batch_size=4,
        seed=7,
        precision="bfloat16",
        ckpt_dir=tmp_path,
        log_every=1,
    )
    device = torch.device("cpu")
    trainer = CNNTrainer(model, cfg, device)

    torch.manual_seed(7)
    images = torch.randn(4, 3, 64, 64)
    labels = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    batch = {"image": images, "label": labels}

    result_a = trainer.step(batch)
    result_b = trainer.step(batch)
    assert result_b.loss < result_a.loss + 1e-3
    trainer._save(epoch=0)
    ckpt_files = list(tmp_path.glob("epoch_*.pt"))
    assert ckpt_files
    payload = load_checkpoint(ckpt_files[0])
    assert payload["rng"]["seed"] == 7
    assert payload["extra"]["batch_size"] == 4
