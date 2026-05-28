from __future__ import annotations

import subprocess
import sys


def _help(module: str) -> int:
    result = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode


def test_cli_help_returns_zero() -> None:
    assert _help("odontocal.entry.train") == 0
    assert _help("odontocal.entry.fit_calibration") == 0
    assert _help("odontocal.entry.evaluate") == 0
    assert _help("odontocal.entry.infer") == 0
    assert _help("odontocal.entry.export_onnx") == 0
