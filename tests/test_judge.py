"""
Tests for the judge module.
Run with: pytest tests/test_judge.py -v
"""
from unittest.mock import AsyncMock, patch

from diffprompt.core.judge import judge_all
from diffprompt.models import TestCase, TestCategory, Verdict


def _tc(i: int) -> TestCase:
    return TestCase(id=f"t{i}", input=f"input {i}", category=TestCategory.TYPICAL)


async def test_judge_all_preserves_order_and_count():
    """
    judge_all runs concurrently but must return results in input order, one
    per test case. We echo each test_case.id back through the reason slot and
    assert the order is intact.
    """
    cases = [_tc(i) for i in range(6)]
    v1 = [f"a{i}" for i in range(6)]
    v2 = [f"b{i}" for i in range(6)]
    sims = [0.5] * 6

    async def fake_single(test_case, v1o, v2o, sim, local_only=False):
        return Verdict.NEUTRAL, test_case.id, 0.9

    with patch("diffprompt.core.judge.judge_single", new=AsyncMock(side_effect=fake_single)):
        results = await judge_all(cases, v1, v2, sims)

    assert len(results) == 6
    assert [reason for _, reason, _ in results] == [f"t{i}" for i in range(6)]
