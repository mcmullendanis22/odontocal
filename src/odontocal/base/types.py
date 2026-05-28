from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TypedDict


class Modality(str, Enum):
    PANORAMIC = "panoramic"
    PERIAPICAL = "periapical"
    INTRAORAL = "intraoral"
    CEPHALOMETRIC = "cephalometric"


@dataclass(frozen=True)
class AdapterConfig:
    root: Path
    split: str
    modality: Modality
    n_classes: int
    label_names: tuple[str, ...]
    license: str
    access_url: str


class Example(TypedDict):
    image_path: str
    label: int
    modality: str
    dataset_tag: str
    case_id: str
