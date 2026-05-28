from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable
from pathlib import Path


def file_sha256(path: str | os.PathLike[str], chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def manifest_sha256(entries: Iterable[tuple[str, str]]) -> str:
    digest = hashlib.sha256()
    for name, hex_hash in sorted(entries):
        digest.update(name.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(hex_hash.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()
