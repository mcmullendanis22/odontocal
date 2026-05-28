from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple


class MLLMOutput(NamedTuple):
    prediction: int
    confidence: float
    raw_text: str


@dataclass
class MLLMSpec:
    name: str
    hf_repo_id: str
    family: str
    label_aliases: dict[str, int]


class MLLMWrapper(ABC):
    spec: MLLMSpec

    def __init__(self, *, dry_run: bool = False, device: str = "cuda") -> None:
        self.dry_run = dry_run
        self.device = device
        self._inner: Any = None

    @abstractmethod
    def _load(self) -> Any: ...

    def _ensure_loaded(self) -> Any:
        if self.dry_run:
            return None
        if self._inner is None:
            self._inner = self._load()
        return self._inner

    @abstractmethod
    def _generate(self, image_path: Path, system_prompt: str, user_prompt: str) -> str: ...

    def predict(self, image_path: str | Path, system_prompt: str, user_prompt: str) -> MLLMOutput:
        if self.dry_run:
            return _stub_output(image_path, self.spec)
        raw = self._generate(Path(image_path), system_prompt, user_prompt)
        return parse_structured_reply(raw, self.spec.label_aliases)


def _stub_output(image_path: str | Path, spec: MLLMSpec) -> MLLMOutput:
    h = abs(hash((str(image_path), spec.name))) % 997
    label_count = max(len(spec.label_aliases), 1)
    pred = h % label_count
    conf = 0.30 + 0.69 * ((h * 17) % 100) / 100.0
    return MLLMOutput(prediction=int(pred), confidence=float(conf), raw_text="[dry-run]")


def parse_structured_reply(text: str, label_aliases: dict[str, int]) -> MLLMOutput:
    payload: dict[str, Any] = {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match is not None:
            try:
                payload = json.loads(match.group(0))
            except json.JSONDecodeError:
                payload = {}

    assessment = str(payload.get("assessment", "")).strip().lower()
    pred = label_aliases.get(assessment, -1)
    if pred < 0:
        for alias, idx in label_aliases.items():
            if alias in assessment:
                pred = idx
                break
    pred = max(pred, 0)

    conf_raw = payload.get("confidence", 0.5)
    try:
        conf = float(conf_raw)
    except (TypeError, ValueError):
        conf = 0.5
    conf = max(0.0, min(1.0, conf))
    return MLLMOutput(prediction=pred, confidence=conf, raw_text=text)
