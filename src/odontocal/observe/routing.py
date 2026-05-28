from __future__ import annotations

from typing import Final

from odontocal.base.types import Modality

MODALITY_OF_DATASET: Final[dict[str, Modality]] = {
    "mmoral_panoramic": Modality.PANORAMIC,
    "mmoral_periapical": Modality.PERIAPICAL,
    "mmoral_intraoral": Modality.INTRAORAL,
    "mmoral_cephalometric": Modality.CEPHALOMETRIC,
    "dentex": Modality.PANORAMIC,
    "denpar": Modality.PERIAPICAL,
    "intraoral_caries": Modality.INTRAORAL,
    "tufts": Modality.PANORAMIC,
}


def route_dataset(dataset_tag: str) -> Modality:
    if dataset_tag not in MODALITY_OF_DATASET:
        raise KeyError(f"dataset_tag {dataset_tag!r} has no modality assignment")
    return MODALITY_OF_DATASET[dataset_tag]
