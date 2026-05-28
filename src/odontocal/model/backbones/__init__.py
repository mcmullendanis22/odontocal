from odontocal.base.registry import Registry
from odontocal.model.backbones.base import MLLMOutput, MLLMSpec, MLLMWrapper, parse_structured_reply
from odontocal.model.backbones.cog_vlm2 import CogVLM2Wrapper
from odontocal.model.backbones.convnet import CNNBaselineConfig, build_cnn_baseline
from odontocal.model.backbones.huatuo_v import HuatuoVisionWrapper
from odontocal.model.backbones.intern_vl import InternVLWrapper
from odontocal.model.backbones.llava_med import LlavaMedWrapper
from odontocal.model.backbones.medvlm_r1 import MedVLMR1Wrapper
from odontocal.model.backbones.oralgpt_omni import OralGPTOmniWrapper
from odontocal.model.backbones.qwen25_vl import Qwen25VLWrapper

_PREDICTORS: Registry[type[MLLMWrapper]] = Registry("mllm wrapper")
_PREDICTORS.register("qwen25_vl", Qwen25VLWrapper)
_PREDICTORS.register("intern_vl", InternVLWrapper)
_PREDICTORS.register("cog_vlm2", CogVLM2Wrapper)
_PREDICTORS.register("llava_med", LlavaMedWrapper)
_PREDICTORS.register("huatuo_v", HuatuoVisionWrapper)
_PREDICTORS.register("medvlm_r1", MedVLMR1Wrapper)
_PREDICTORS.register("oralgpt_omni", OralGPTOmniWrapper)


def build_mllm_wrapper(name: str, *, dry_run: bool = False, device: str = "cuda") -> MLLMWrapper:
    cls = _PREDICTORS.get(name)
    return cls(dry_run=dry_run, device=device)


__all__ = [
    "CNNBaselineConfig",
    "CogVLM2Wrapper",
    "HuatuoVisionWrapper",
    "InternVLWrapper",
    "LlavaMedWrapper",
    "MLLMOutput",
    "MLLMSpec",
    "MLLMWrapper",
    "MedVLMR1Wrapper",
    "OralGPTOmniWrapper",
    "Qwen25VLWrapper",
    "build_cnn_baseline",
    "build_mllm_wrapper",
    "parse_structured_reply",
]
