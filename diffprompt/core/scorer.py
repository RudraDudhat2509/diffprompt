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
    if not diffs:
        return 50.0
    total_weight = sum(d.divergence for d in diffs)
    if total_weight == 0:
        return 100.0
    weighted_sum = sum(
        d.divergence * (1.0 if d.verdict == Verdict.IMPROVEMENT else -1.0 if d.verdict == Verdict.REGRESSION else 0.0)
        for d in diffs
    )
    return float(round(((weighted_sum / total_weight) + 1) / 2 * 100, 1))


def importance_score(diff: DiffResult) -> float:
    divergence   = diff.divergence
    centrality   = diff.cluster_centrality
    input_length = len(diff.test_case.input.split())
    simplicity   = max(0.0, 1.0 - input_length / 50)
    surprise     = divergence * simplicity
    return 0.4 * divergence + 0.3 * centrality + 0.3 * surprise


async def select_key_examples(
    diffs: list[DiffResult],
    top_n: int = 3,
    local_only: bool = False,
) -> list[KeyExample]:
    if not diffs:
        return []

    for d in diffs:
        d.importance_score = importance_score(d)

    selected: list[tuple[str, DiffResult]] = []
    used_ids: set[str] = set()

    def _pick(slot: str, candidate) -> None:
        if candidate and candidate.test_case.id not in used_ids:
            selected.append((slot, candidate))
            used_ids.add(candidate.test_case.id)

    _pick("most_important", max(diffs, key=lambda d: d.importance_score))

    improvements = [d for d in diffs if d.verdict == Verdict.IMPROVEMENT and d.test_case.id not in used_ids]
    if improvements:
        _pick("best_improvement", max(improvements, key=lambda d: d.divergence))

    remaining = [d for d in diffs if d.test_case.id not in used_ids]
    if remaining:
        _pick("most_surprising", max(
            remaining,
            key=lambda d: d.divergence * max(0, 1 - len(d.test_case.input.split()) / 50),
        ))

    if top_n > 3:
        regressions = sorted(
            [d for d in diffs if d.verdict == Verdict.REGRESSION and d.test_case.id not in used_ids],
            key=lambda d: d.divergence, reverse=True,
        )
        for d in regressions[:top_n - 3]:
            _pick(f"regression_{len(selected)}", d)

    selected = selected[:top_n]

    examples = []
    for slot_name, diff in selected:
        why, _ = await call_cascade(
            WHY_IT_MATTERS_PROMPT.format(
                input=diff.test_case.input,
                v1=diff.v1_output[:600],
                v2=diff.v2_output[:600],
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