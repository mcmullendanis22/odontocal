from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from odontocal.base.serialization import read_jsonl
from odontocal.base.types import AdapterConfig, Example
from odontocal.observe.routing import Modality

_DENTEX_LABELS = ("normal", "caries", "deep_caries", "periapical_lesion", "impacted_tooth")


class DentexPanoramic:
    def __init__(self, root: str | Path, split: str) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of train/val/test")
        self.config = AdapterConfig(
            root=Path(root),
            split=split,
            modality=Modality.PANORAMIC,
            n_classes=len(_DENTEX_LABELS),
            label_names=_DENTEX_LABELS,
            license="CC BY-NC-SA",
            access_url="https://dentex.grand-challenge.org/",
        )

    @property
    def manifest_path(self) -> Path:
        return self.config.root / "dentex" / f"{self.config.split}.jsonl"

    def cases(self) -> list[Example]:
        return list(self)

    def __iter__(self) -> Iterator[Example]:
        for row in read_jsonl(self.manifest_path):
            yield Example(
                image_path=str(row["image_path"]),
                label=int(row["label"]),
                modality=Modality.PANORAMIC.value,
                dataset_tag="dentex",
                case_id=str(row["case_id"]),
            )

    def __len__(self) -> int:
        return sum(1 for _ in read_jsonl(self.manifest_path))
