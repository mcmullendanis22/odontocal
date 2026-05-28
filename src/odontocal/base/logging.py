from __future__ import annotations

import logging
import sys
from typing import Final

_FMT: Final[str] = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT: Final[str] = "%Y-%m-%dT%H:%M:%S"


def configure_root_logger(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if root.handlers:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(fmt=_FMT, datefmt=_DATEFMT))
    root.addHandler(handler)
    root.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
