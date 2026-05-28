from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from odontocal.base.serialization import read_jsonl
from odontocal.base.types import AdapterConfig, Example
from odontocal.observe.routing import Modality

_TUFTS_LABELS = ("normal", "abnormal")
_ACCESS_NOTE = (
    "Tufts Dental Database is request-based; obtain access at https://tdd.ece.tufts.edu/ "
    "and place the manifest at <root>/tufts/<split>.jsonl before running."
)


class TuftsPanoramic:
    def __init__(self, root: str | Path, split: str) -> None:
        self.config = AdapterConfig(
            root=Path(root),
            split=split,
            modality=Modality.PANORAMIC,
            n_classes=len(_TUFTS_LABELS),
            label_names=_TUFTS_LABELS,
            license="request-based",
            access_url="https://tdd.ece.tufts.edu/",
        )

    @property
    def manifest_path(self) -> Path:
        return self.config.root / "tufts" / f"{self.config.split}.jsonl"

    def ensure_manifest(self) -> None:
        if not self.manifest_path.exists():
            raise FileNotFoundError(_ACCESS_NOTE)

    def __iter__(self) -> Iterator[Example]:
        self.ensure_manifest()
        for row in read_jsonl(self.manifest_path):
            yield Example(
                image_path=str(row["image_path"]),
                label=int(row["label"]),
                modality=Modality.PANORAMIC.value,
                dataset_tag="tufts",
                case_id=str(row["case_id"]),
            )

    def __len__(self) -> int:
        self.ensure_manifest()
        return sum(1 for _ in read_jsonl(self.manifest_path))
