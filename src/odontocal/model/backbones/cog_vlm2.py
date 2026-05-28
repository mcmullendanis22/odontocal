from __future__ import annotations

from pathlib import Path
from typing import Any

from odontocal.model.backbones.base import MLLMSpec, MLLMWrapper

_LABELS = {"caries": 0, "periodontal": 1, "periapical": 2, "opmd": 3, "normal": 4}


class CogVLM2Wrapper(MLLMWrapper):
    spec = MLLMSpec(
        name="cog_vlm2",
        hf_repo_id="THUDM/cogvlm2-llama3-chat-19B",
        family="general",
        label_aliases=_LABELS,
    )

    def _load(self) -> Any:
        from PIL import Image as _PIL_Image
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.spec.hf_repo_id, trust_remote_code=True)
        model = (
            AutoModelForCausalLM.from_pretrained(
                self.spec.hf_repo_id, torch_dtype="bfloat16", trust_remote_code=True
            )
            .to(self.device)
            .eval()
        )
        return tokenizer, model, _PIL_Image

    def _generate(self, image_path: Path, system_prompt: str, user_prompt: str) -> str:
        tokenizer, model, image_module = self._ensure_loaded()
        image = image_module.open(str(image_path)).convert("RGB")
        prompt = f"{system_prompt}\n{user_prompt}"
        inputs = model.build_conversation_input_ids(
            tokenizer, query=prompt, history=[], images=[image]
        )
        gen_kwargs = {"max_new_tokens": 192, "do_sample": False, "temperature": 0.0}
        outputs = model.generate(
            **{k: v.to(self.device) for k, v in inputs.items() if hasattr(v, "to")}, **gen_kwargs
        )
        return str(tokenizer.decode(outputs[0], skip_special_tokens=True))
