"""
LLM-as-judge layer.
Determines verdict (improvement/regression/neutral) + reason per diff.
Escalates to larger Groq model when confidence is low.
"""
from __future__ import annotations
import json
import re
from diffprompt.models import Verdict, DiffResult, TestCase
from diffprompt.models.cascade import call_cascade, call_groq_only


JUDGE_PROMPT = """You are evaluating whether a prompt change improved or worsened an output.

Input: {input}

Output V1:
{v1}

Output V2:
{v2}

Which output better serves this input? Consider: accuracy, completeness, tone, and appropriateness.
Be specific. One sentence reason.

Respond ONLY with valid JSON (no markdown fences):
{{"verdict": "improvement|regression|neutral", "reason": "one sentence explaining the key difference", "confidence": 0.0}}

confidence should reflect how clear-cut the verdict is (1.0 = obvious, 0.5 = ambiguous)."""

CONFIDENCE_THRESHOLD = 0.65  # escalate to larger model below this


async def judge_single(
    test_case: TestCase,
    v1_output: str,
    v2_output: str,
    similarity: float,
    local_only: bool = False,
) -> tuple[Verdict, str, float]:
    """
    Returns (verdict, reason, confidence).
    Auto-escalates to Groq 70B if confidence is low.
    """
    # Fast path: very high similarity → neutral, skip judge entirely
    if similarity > 0.95:
        return Verdict.NEUTRAL, "outputs are semantically identical", 1.0

    prompt = JUDGE_PROMPT.format(
        input=test_case.input,
        v1=v1_output[:800],   # truncate to avoid token waste
        v2=v2_output[:800],
    )

    raw, _ = await call_cascade(prompt, local_only=local_only)
    verdict, reason, confidence = _parse_judge_response(raw)

    # Escalate if confidence is low and we're not local-only.
    # BUG FIX: previously called call_cascade() again which would re-try
    # Ollama first — defeating the purpose of escalation. Now calls
    # call_groq_only() directly to guarantee a larger model.
    if confidence < CONFIDENCE_THRESHOLD and not local_only:
        escalated = await call_groq_only(
            prompt,
            groq_model="llama-3.3-70b-versatile",
        )
        if escalated:
            verdict, reason, confidence = _parse_judge_response(escalated)

    return verdict, reason, confidence


def _parse_judge_response(raw: str) -> tuple[Verdict, str, float]:
    """Parse judge JSON response with fallback."""
    try:
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        # Handle responses where the model adds text before/after JSON
        match = re.search(r'\{.*\}', clean, re.DOTALL)
        if match:
            clean = match.group(0)
        data = json.loads(clean)
        verdict_str = data.get("verdict", "neutral").lower().strip()
        # Normalize common variants
        if verdict_str in ("improve", "improved", "better"):
            verdict_str = "improvement"
        elif verdict_str in ("regress", "regressed", "worse"):
            verdict_str = "regression"
        verdict = Verdict(verdict_str) if verdict_str in Verdict._value2member_map_ else Verdict.NEUTRAL
        reason = str(data.get("reason", "no reason provided"))[:300]
        confidence = float(data.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))
        return verdict, reason, confidence
    except Exception:
        return Verdict.NEUTRAL, "judge parse error — raw response was not valid JSON", 0.0