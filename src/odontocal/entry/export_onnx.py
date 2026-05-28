from __future__ import annotations

import argparse
from pathlib import Path

import torch

from odontocal.base.logging import configure_root_logger, get_logger
from odontocal.model.backbones.convnet import CNNBaselineConfig, build_cnn_baseline


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="odc-export-onnx", description="export a CNN baseline to ONNX"
    )
    parser.add_argument("--name", required=True, choices=["resnet50", "efficientnet_b4"])
    parser.add_argument("--n-classes", type=int, required=True)
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--input-size", type=int, default=380)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    configure_root_logger()
    log = get_logger("odontocal.entry.export_onnx")
    model = build_cnn_baseline(
        CNNBaselineConfig(name=args.name, n_classes=args.n_classes, pretrained=False)
    )
    state = torch.load(str(args.ckpt), map_location="cpu")
    model.load_state_dict(state["model_state"])
    model.eval()
    dummy = torch.randn(1, 3, args.input_size, args.input_size)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(args.out),
        input_names=["image"],
        output_names=["logits"],
        opset_version=17,
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
    )
    log.info("ONNX export → %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
