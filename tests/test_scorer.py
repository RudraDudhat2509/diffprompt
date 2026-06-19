"""
Tests for the scorer module.
Run with: pytest tests/test_scorer.py -v
"""
from unittest.mock import AsyncMock, patch

from diffprompt.core.scorer import regression_score, select_key_examples
from diffprompt.models import DiffResult, TestCase, TestCategory, Verdict


def make_diff(text: str, sim: float, verdict: Verdict) -> DiffResult:
    tc = TestCase(
        id=text.replace(" ", "_"),  # full text -> unique id per diff
        input=text,
        category=TestCategory.TYPICAL,
        tags={},
    )
    return DiffResult(
        test_case=tc,
        v1_output="v1 output",
        v2_output="v2 output",
        similarity=sim,
        divergence=1 - sim,
        verdict=verdict,
        reason="test reason",
        judge_confidence=0.9,
    )


def test_regression_score_all_improvements_is_high():
    diffs = [make_diff(f"input {i}", 0.3, Verdict.IMPROVEMENT) for i in range(5)]
    assert regression_score(diffs) == 100.0


def test_regression_score_all_regressions_is_low():
    diffs = [make_diff(f"input {i}", 0.3, Verdict.REGRESSION) for i in range(5)]
    assert regression_score(diffs) == 0.0


def test_regression_score_empty_is_neutral():
    assert regression_score([]) == 50.0


async def test_select_key_examples_respects_top_n():
    """
    Regression for #2: --top-n must reach select_key_examples. Previously the
    CLI never forwarded it, so the report was always capped at 3 examples.
    """
    diffs = [make_diff(f"input number {i}", 0.2, Verdict.REGRESSION) for i in range(8)]
    diffs.append(make_diff("a clear improvement here", 0.3, Verdict.IMPROVEMENT))

    mock = AsyncMock(return_value=("because reasons", "local/x"))
    with patch("diffprompt.core.scorer.call_cascade", new=mock):
        examples = await select_key_examples(diffs, top_n=5)

    assert len(examples) == 5


async def test_select_key_examples_default_is_three():
    diffs = [make_diff(f"input number {i}", 0.2, Verdict.REGRESSION) for i in range(8)]
    diffs.append(make_diff("a clear improvement here", 0.3, Verdict.IMPROVEMENT))

    mock = AsyncMock(return_value=("because reasons", "local/x"))
    with patch("diffprompt.core.scorer.call_cascade", new=mock):
        examples = await select_key_examples(diffs)

    assert len(examples) == 3
