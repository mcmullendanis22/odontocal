from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from odontocal.base.serialization import read_jsonl
from odontocal.base.types import AdapterConfig, Example
from odontocal.observe.routing import Modality

_DENPAR_LABELS = ("stage_i", "stage_ii", "stage_iii", "stage_iv")


def _config_for(root: Path, split: str) -> AdapterConfig:
    return AdapterConfig(
        root=root,
        split=split,
        modality=Modality.PERIAPICAL,
        n_classes=len(_DENPAR_LABELS),
        label_names=_DENPAR_LABELS,
        license="see Nature Scientific Data record",
        access_url="https://zenodo.org/records/13998619",
    )


class DenPARPeriapical:
    def __init__(self, root: str | Path, split: str) -> None:
        self.config = _config_for(Path(root), split)

    def manifest(self) -> Path:
        return self.config.root / "denpar" / f"{self.config.split}.jsonl"

    def __iter__(self) -> Iterator[Example]:
        for row in read_jsonl(self.manifest()):
            yield Example(
                image_path=str(row["image_path"]),
                label=int(row["label"]),
                modality=Modality.PERIAPICAL.value,
                dataset_tag="denpar",
                case_id=str(row["case_id"]),
            )

    def __len__(self) -> int:
        return sum(1 for _ in read_jsonl(self.manifest()))
