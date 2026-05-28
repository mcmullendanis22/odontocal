from __future__ import annotations

import argparse
from pathlib import Path

import torch

from odontocal.base.logging import configure_root_logger, get_logger
from odontocal.base.serialization import load_yaml
from odontocal.model.backbones.convnet import CNNBaselineConfig, build_cnn_baseline
from odontocal.operate import CNNTrainConfig, CNNTrainer


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="odc-train", description="train CNN baselines")
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt-dir", type=Path, default=Path("checkpoints"))
    return parser


def _load_loader(_data_root: Path, _batch_size: int) -> object:
    raise NotImplementedError(
        "image loading is project-deployment specific; wire a torch DataLoader against the prepared manifests"
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    configure_root_logger()
    log = get_logger("odontocal.entry.train")

    exp = load_yaml(args.experiment)
    model_cfg = load_yaml(args.experiment.parent.parent / "model" / f"{exp['model']}.yaml")

    cfg = CNNTrainConfig(
        lr=float(exp["lr"]),
        weight_decay=float(exp["weight_decay"]),
        epochs=int(exp["epochs"]),
        batch_size=int(exp["batch_size"]),
        seed=args.seed,
        precision=str(exp.get("precision", "bfloat16")),
        ckpt_dir=args.ckpt_dir,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_cnn_baseline(
        CNNBaselineConfig(
            name=model_cfg["name"],
            n_classes=int(exp["n_classes"]),
            pretrained=bool(model_cfg.get("pretrained", True)),
        )
    )
    CNNTrainer(model, cfg, device)
    log.info("trainer ready · seed=%d · model=%s", args.seed, model_cfg["name"])
    log.info(
        "data loading is project-specific; replace _load_loader to wire a torch DataLoader from %s",
        args.data,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
