from __future__ import annotations

from pathlib import Path
from typing import Any

from odontocal.model.backbones.base import MLLMSpec, MLLMWrapper

_LABELS = {"caries": 0, "periodontal": 1, "periapical": 2, "opmd": 3, "normal": 4}


class LlavaMedWrapper(MLLMWrapper):
    spec = MLLMSpec(
        name="llava_med",
        hf_repo_id="microsoft/llava-med-v1.5-mistral-7b",
        family="medical",
        label_aliases=_LABELS,
    )

    def _load(self) -> Any:
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        processor = AutoProcessor.from_pretrained(self.spec.hf_repo_id, trust_remote_code=True)
        model = (
            LlavaForConditionalGeneration.from_pretrained(
                self.spec.hf_repo_id, torch_dtype="bfloat16", trust_remote_code=True
            )
            .to(self.device)
            .eval()
        )
        return processor, model

    def _generate(self, image_path: Path, system_prompt: str, user_prompt: str) -> str:
        processor, model = self._ensure_loaded()
        prompt = f"USER: <image>\n{system_prompt}\n{user_prompt}\nASSISTANT:"
        inputs = processor(text=prompt, images=str(image_path), return_tensors="pt").to(self.device)
        output_ids = model.generate(**inputs, max_new_tokens=192, do_sample=False, temperature=0.0)
        return str(processor.decode(output_ids[0], skip_special_tokens=True))
