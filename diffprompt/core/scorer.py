"""
Regression scoring and key example selection.
"""
from __future__ import annotations
import numpy as np
from diffprompt.models import DiffResult, KeyExample, Verdict
from diffprompt.models.cascade import call_cascade


WHY_IT_MATTERS_PROMPT = """
Input: {input}
V1 output: {v1}
V2 output: {v2}
Verdict: {verdict}

In ONE sentence, explain what this reveals about the prompt change.
Focus on the mechanism — why did v2 behave differently here?
"""


def regression_score(diffs: list[DiffResult]) -> float:
    """
    Compute overall regression score 0-100.
    100 = v2 is perfect, 0 = v2 regresses everywhere.

    Formula: weighted average where improvements add, regressions subtract.
    Weighted by divergence so big changes matter more than small ones.
    """
    if not diffs:
        return 50.0

    total_weight = sum(d.divergence for d in diffs)
    if total_weight == 0:
        return 100.0

    weighted_sum = 0.0
    for d in diffs:
        if d.verdict == Verdict.IMPROVEMENT:
            weighted_sum += d.divergence * 1.0
        elif d.verdict == Verdict.REGRESSION:
            weighted_sum += d.divergence * -1.0
        # neutral = 0 contribution

    # Normalize to 0-100 (0.5 = 50 baseline)
    normalized = (weighted_sum / total_weight + 1) / 2
    return float(round(normalized * 100, 1))


def importance_score(diff: DiffResult) -> float:
    """
    Rank test cases by how informative they are.
    importance = 0.4*divergence + 0.3*centrality + 0.3*surprise
    surprise = big change on a simple (short) input
    """
    divergence = diff.divergence
    centrality = diff.cluster_centrality
    input_length = len(diff.test_case.input.split())
    simplicity = max(0.0, 1.0 - input_length / 50)  # short inputs = high simplicity
    surprise = divergence * simplicity

    return 0.4 * divergence + 0.3 * centrality + 0.3 * surprise


async def select_key_examples(
    diffs: list[DiffResult],
    local_only: bool = False,
) -> list[KeyExample]:
    """
    Select 3 key examples: most important, best improvement, most surprising.
    Each gets a 'why it matters' sentence from the judge.
    """
    if not diffs:
        return []

    # Score all diffs
    for d in diffs:
        d.importance_score = importance_score(d)

    # Slot 1: Most Important (highest importance score)
    most_important = max(diffs, key=lambda d: d.importance_score)

    # Slot 2: Best Improvement (highest divergence among improvements)
    improvements = [d for d in diffs if d.verdict == Verdict.IMPROVEMENT]
    best_improvement = max(improvements, key=lambda d: d.divergence) if improvements else None

    # Slot 3: Most Surprising (high divergence + simple input, excluding slot 1)
    remaining = [d for d in diffs if d.test_case.id != most_important.test_case.id]
    most_surprising = max(
        remaining,
        key=lambda d: d.divergence * max(0, 1 - len(d.test_case.input.split()) / 50)
    ) if remaining else None

    examples = []
    slots = [
        ("most_important", most_important),
        ("best_improvement", best_improvement),
        ("most_surprising", most_surprising),
    ]

    for slot_name, diff in slots:
        if diff is None:
            continue
        why, _ = await call_cascade(
            WHY_IT_MATTERS_PROMPT.format(
                input=diff.test_case.input,
                v1=diff.v1_output,
                v2=diff.v2_output,
                verdict=diff.verdict.value,
            ),
            local_only=local_only,
        )
        examples.append(KeyExample(
            slot=slot_name,
            diff=diff,
            why_it_matters=why.strip(),
        ))

    return examples
