from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

from odontocal.base.types import AdapterConfig, Example


class DatasetAdapter(Protocol):
    config: AdapterConfig

    def __iter__(self) -> Iterator[Example]: ...

    def __len__(self) -> int: ...
