from __future__ import annotations

import json

from odontocal.model.backbones import build_mllm_wrapper
from odontocal.model.backbones.base import MLLMOutput, parse_structured_reply

_WRAPPER_NAMES = (
    "qwen25_vl",
    "intern_vl",
    "cog_vlm2",
    "llava_med",
    "huatuo_v",
    "medvlm_r1",
    "oralgpt_omni",
)


def test_every_wrapper_returns_well_shaped_dry_run_output() -> None:
    for name in _WRAPPER_NAMES:
        wrapper = build_mllm_wrapper(name, dry_run=True, device="cpu")
        out = wrapper.predict("/tmp/fake.png", "system", "user")
        assert isinstance(out, MLLMOutput)
        assert isinstance(out.prediction, int)
        assert 0 <= out.prediction < len(wrapper.spec.label_aliases)
        assert 0.0 <= out.confidence <= 1.0
        assert isinstance(out.raw_text, str)


def test_parse_structured_reply_handles_clean_json() -> None:
    payload = {"assessment": "caries", "confidence": 0.83, "refer": False}
    out = parse_structured_reply(
        json.dumps(payload),
        label_aliases={"caries": 0, "periodontal": 1, "periapical": 2, "opmd": 3, "normal": 4},
    )
    assert out.prediction == 0
    assert abs(out.confidence - 0.83) < 1e-9


def test_parse_structured_reply_recovers_from_prose_wrap() -> None:
    text = 'Sure! Here is the result: {"assessment": "opmd", "confidence": 0.42, "refer": true} thanks.'
    out = parse_structured_reply(
        text, label_aliases={"caries": 0, "periodontal": 1, "periapical": 2, "opmd": 3, "normal": 4}
    )
    assert out.prediction == 3
    assert abs(out.confidence - 0.42) < 1e-9


def test_parse_structured_reply_clips_confidence() -> None:
    out = parse_structured_reply(
        json.dumps({"assessment": "normal", "confidence": 1.7}),
        label_aliases={"caries": 0, "periodontal": 1, "periapical": 2, "opmd": 3, "normal": 4},
    )
    assert out.confidence == 1.0
