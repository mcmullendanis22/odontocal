from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from odontocal.base.serialization import read_jsonl
from odontocal.base.types import AdapterConfig, Example
from odontocal.observe.routing import Modality


@dataclass
class IntraoralCaries:
    config: AdapterConfig

    @classmethod
    def build(cls, root: str | Path, split: str) -> IntraoralCaries:
        labels = ("sound", "caries")
        return cls(
            config=AdapterConfig(
                root=Path(root),
                split=split,
                modality=Modality.INTRAORAL,
                n_classes=len(labels),
                label_names=labels,
                license="see Nature Scientific Data record",
                access_url="https://zenodo.org/records/14827784",
            )
        )

    def _manifest(self) -> Path:
        return self.config.root / "intraoral_caries" / f"{self.config.split}.jsonl"

    def __iter__(self) -> Iterator[Example]:
        for row in read_jsonl(self._manifest()):
            yield Example(
                image_path=str(row["image_path"]),
                label=int(row["label"]),
                modality=Modality.INTRAORAL.value,
                dataset_tag="intraoral_caries",
                case_id=str(row["case_id"]),
            )

    def __len__(self) -> int:
        return sum(1 for _ in read_jsonl(self._manifest()))
