from __future__ import annotations

import argparse
from pathlib import Path

from odontocal.base.logging import configure_root_logger, get_logger
from odontocal.base.serialization import read_jsonl, write_jsonl
from odontocal.model.backbones import build_mllm_wrapper
from odontocal.observe.prompting import structured_triage_prompt
from odontocal.observe.routing import Modality


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="odc-infer", description="run MLLM inference on a manifest"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="one of qwen25_vl, intern_vl, cog_vlm2, llava_med, huatuo_v, medvlm_r1, oralgpt_omni",
    )
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="input manifest jsonl with image_path, modality, case_id",
    )
    parser.add_argument("--output", required=True, type=Path, help="output jsonl path")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    configure_root_logger()
    log = get_logger("odontocal.entry.infer")
    wrapper = build_mllm_wrapper(args.model, dry_run=args.dry_run, device=args.device)
    rows: list[dict[str, object]] = []
    for row in read_jsonl(args.manifest):
        modality = Modality(row["modality"])
        prompt = structured_triage_prompt(
            modality, clinical_context=str(row.get("clinical_context", ""))
        )
        output = wrapper.predict(row["image_path"], prompt.system, prompt.user)
        rows.append(
            {
                "case_id": row["case_id"],
                "modality": modality.value,
                "model": args.model,
                "prediction": int(output.prediction),
                "confidence": float(output.confidence),
                "raw_text": output.raw_text,
            }
        )
    n = write_jsonl(args.output, rows)
    log.info("wrote %d predictions to %s", n, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
