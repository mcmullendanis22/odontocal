from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from odontocal.base.serialization import read_jsonl
from odontocal.base.types import AdapterConfig, Example
from odontocal.observe.routing import Modality

_MMORAL_LABELS = ("caries", "periodontal", "periapical", "opmd", "normal")


@dataclass
class MMOralVQA:
    config: AdapterConfig

    @classmethod
    def from_root(cls, root: str | Path, split: str, modality: Modality) -> MMOralVQA:
        return cls(
            config=AdapterConfig(
                root=Path(root),
                split=split,
                modality=modality,
                n_classes=len(_MMORAL_LABELS),
                label_names=_MMORAL_LABELS,
                license="LICENSE_UNVERIFIED",
                access_url="",
            )
        )

    def _manifest_path(self) -> Path:
        return (
            self.config.root
            / "manifests"
            / f"mmoral_{self.config.modality.value}_{self.config.split}.jsonl"
        )

    def __iter__(self) -> Iterator[Example]:
        for row in read_jsonl(self._manifest_path()):
            yield Example(
                image_path=str(row["image_path"]),
                label=int(row["label"]),
                modality=self.config.modality.value,
                dataset_tag=f"mmoral_{self.config.modality.value}",
                case_id=str(row["case_id"]),
            )

    def __len__(self) -> int:
        return sum(1 for _ in read_jsonl(self._manifest_path()))
