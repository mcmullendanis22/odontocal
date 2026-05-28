from __future__ import annotations

from pathlib import Path
from typing import Any

from odontocal.model.backbones.base import MLLMSpec, MLLMWrapper

_LABEL_ALIASES = {
    "caries": 0,
    "periodontal": 1,
    "periapical": 2,
    "opmd": 3,
    "normal": 4,
}


class Qwen25VLWrapper(MLLMWrapper):
    spec = MLLMSpec(
        name="qwen25_vl",
        hf_repo_id="Qwen/Qwen2.5-VL-7B-Instruct",
        family="general",
        label_aliases=_LABEL_ALIASES,
    )

    def _load(self) -> Any:
        from transformers import AutoModelForCausalLM, AutoProcessor

        processor = AutoProcessor.from_pretrained(self.spec.hf_repo_id, trust_remote_code=True)
        model = (
            AutoModelForCausalLM.from_pretrained(
                self.spec.hf_repo_id, torch_dtype="bfloat16", trust_remote_code=True
            )
            .to(self.device)
            .eval()
        )
        return processor, model

    def _generate(self, image_path: Path, system_prompt: str, user_prompt: str) -> str:
        processor, model = self._ensure_loaded()
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[prompt], images=[str(image_path)], return_tensors="pt").to(
            self.device
        )
        output_ids = model.generate(**inputs, max_new_tokens=192, do_sample=False, temperature=0.0)
        return str(processor.batch_decode(output_ids, skip_special_tokens=True)[0])
