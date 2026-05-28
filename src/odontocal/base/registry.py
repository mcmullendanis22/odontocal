from __future__ import annotations

from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._entries: dict[str, T] = {}

    def register(self, name: str, value: T) -> T:
        if name in self._entries:
            raise KeyError(f"{self._kind} {name!r} is already registered")
        self._entries[name] = value
        return value

    def get(self, name: str) -> T:
        if name not in self._entries:
            raise KeyError(
                f"{self._kind} {name!r} not registered; choose from {sorted(self._entries)}"
            )
        return self._entries[name]

    def names(self) -> list[str]:
        return sorted(self._entries)
