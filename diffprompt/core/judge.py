"""
LLM-as-judge layer.
Determines verdict (improvement/regression/neutral) + reason per diff.
Escalates to larger model when confidence is low.
"""
from __future__ import annotations
import json
import re
from diffprompt.models import Verdict, DiffResult, TestCase
from diffprompt.models.cascade import call_cascade


JUDGE_PROMPT = """You are evaluating whether a prompt change improved or worsened an output.

Input: {input}

Output V1: {v1}

Output V2: {v2}

Which output is better for this input, and why?
Be concise. Focus on factual accuracy, completeness, and appropriateness.

Respond ONLY with valid JSON:
{{"verdict": "improvement|regression|neutral", "reason": "one sentence", "confidence": 0.0-1.0}}"""

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
    # Fast path: very high similarity → neutral, skip judge
    if similarity > 0.95:
        return Verdict.NEUTRAL, "outputs are semantically identical", 1.0

    prompt = JUDGE_PROMPT.format(
        input=test_case.input,
        v1=v1_output,
        v2=v2_output,
    )

    raw, model_used = await call_cascade(prompt, local_only=local_only)
    verdict, reason, confidence = _parse_judge_response(raw)

    # Escalate if confidence is low and we're not local-only
    if confidence < CONFIDENCE_THRESHOLD and not local_only:
        raw, _ = await call_cascade(
            prompt,
            groq_model="llama-3.3-70b-versatile",
            local_only=False,
        )
        verdict, reason, confidence = _parse_judge_response(raw)

    return verdict, reason, confidence


def _parse_judge_response(raw: str) -> tuple[Verdict, str, float]:
    """Parse judge JSON response with fallback."""
    try:
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        data = json.loads(clean)
        verdict = Verdict(data.get("verdict", "neutral"))
        reason = data.get("reason", "no reason provided")
        confidence = float(data.get("confidence", 0.5))
        return verdict, reason, confidence
    except Exception:
        return Verdict.NEUTRAL, "judge parse error", 0.0
