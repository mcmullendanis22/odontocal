from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(20260523)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return _ROOT
