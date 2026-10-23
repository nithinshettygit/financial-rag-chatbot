# app/guardrails_config/actions.py
"""NeMo Guardrails custom actions. Thin wrappers over app.utils.guard_rules."""
from typing import Optional

from nemoguardrails.actions import action

from app.config import settings
from app.utils import guard_rules as rules


@action(name="mask_pii")
async def mask_pii(text: str = "") -> str:
    return rules.mask_pii(text or "")


@action(name="check_input_heuristics")
async def check_input_heuristics(text: str = "") -> str:
    return rules.detect_injection(text or "")


@action(name="check_prompt_leak")
async def check_prompt_leak(context: Optional[dict] = None) -> bool:
    """True == safe."""
    return not rules.leaks_prompt((context or {}).get("bot_message", ""))


@action(name="check_numeric_grounding")
async def check_numeric_grounding(context: Optional[dict] = None) -> bool:
    """True == grounded. Majority of material figures must be traceable to the retrieved context."""
    if not settings.GUARDRAILS_NUMERIC_CHECK:
        return True
    ctx = context or {}
    answer = ctx.get("bot_message") or ""
    evidence = ctx.get("relevant_chunks") or ""
    if isinstance(evidence, (list, tuple)):
        evidence = "\n".join(map(str, evidence))
    if not evidence:
        return True  # nothing to verify against (e.g. non-RAG reply)
    ratio, _ = rules.ungrounded_ratio(answer, evidence)
    return ratio <= 0.5