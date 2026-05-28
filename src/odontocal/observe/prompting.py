from __future__ import annotations

from dataclasses import dataclass

from odontocal.observe.routing import Modality


@dataclass(frozen=True)
class PromptTemplate:
    system: str
    user: str
    response_schema: str


_BASE_SYSTEM: str = (
    "You are a dental triage assistant. Given a single dental image and a short clinical "
    "context, return exactly three fields: a categorical diagnostic assessment from "
    "{caries, periodontal, periapical, opmd, normal}; a numerical confidence in [0,1]; "
    "and a binary specialist-referral recommendation."
)


def _user_block(modality: Modality, clinical_context: str) -> str:
    return (
        f"Imaging modality: {modality.value}.\n"
        f"Clinical context: {clinical_context.strip() or 'not provided'}.\n"
        "Respond in JSON with keys assessment, confidence, refer."
    )


def structured_triage_prompt(modality: Modality, clinical_context: str = "") -> PromptTemplate:
    return PromptTemplate(
        system=_BASE_SYSTEM,
        user=_user_block(modality, clinical_context),
        response_schema='{"assessment": str, "confidence": float, "refer": bool}',
    )
