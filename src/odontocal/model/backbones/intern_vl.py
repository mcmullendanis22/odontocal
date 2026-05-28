from __future__ import annotations

from pathlib import Path
from typing import Any

from odontocal.model.backbones.base import MLLMSpec, MLLMWrapper

_LABELS = {"caries": 0, "periodontal": 1, "periapical": 2, "opmd": 3, "normal": 4}


class InternVLWrapper(MLLMWrapper):
    spec = MLLMSpec(
        name="intern_vl",
        hf_repo_id="OpenGVLab/InternVL3_5-8B",
        family="general",
        label_aliases=_LABELS,
    )

    def _load(self) -> Any:
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.spec.hf_repo_id, trust_remote_code=True)
        model = (
            AutoModel.from_pretrained(
                self.spec.hf_repo_id, torch_dtype="bfloat16", trust_remote_code=True
            )
            .to(self.device)
            .eval()
        )
        return tokenizer, model

    def _generate(self, image_path: Path, system_prompt: str, user_prompt: str) -> str:
        tokenizer, model = self._ensure_loaded()
        question = f"<image>\n{system_prompt}\n\n{user_prompt}"
        generation_config = {"max_new_tokens": 192, "do_sample": False, "temperature": 0.0}
        return str(model.chat(tokenizer, str(image_path), question, generation_config))
