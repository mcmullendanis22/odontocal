from __future__ import annotations

import contextlib
import json
import os
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | os.PathLike[str]) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"expected top-level mapping in {path}")
    return data


def write_atomic(path: str | os.PathLike[str], payload: bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=str(target.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, target)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_name)
        raise


def read_jsonl(path: str | os.PathLike[str]) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: str | os.PathLike[str], rows: Iterable[dict[str, Any]]) -> int:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    payload_chunks: list[str] = []
    for row in rows:
        payload_chunks.append(json.dumps(row, sort_keys=True, separators=(",", ":")))
        count += 1
    payload = ("\n".join(payload_chunks) + ("\n" if payload_chunks else "")).encode("utf-8")
    write_atomic(target, payload)
    return count
