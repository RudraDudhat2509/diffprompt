"""
Tests for the fitness module.
Run with: pytest tests/test_fitness.py -v
"""
from unittest.mock import AsyncMock, patch

from diffprompt.core.fitness import fitness, score_task
from diffprompt.models import Check, CheckType, GoldenTask


def test_score_task_checks_only():
    task = GoldenTask(
        input="capital of France?",
        checks=[Check(type=CheckType.KEYWORD, keywords=["Paris"], weight=1.0)],
    )
    assert score_task(task, "Paris is the capital.") == 1.0
    assert score_task(task, "Berlin is the capital.") == 0.0


def test_score_task_blends_check_and_embedding_by_weight():
    task = GoldenTask(
        input="capital of France?",
        golden_answer="Paris is the capital of France.",
        golden_answer_weight=1.0,
        checks=[Check(type=CheckType.KEYWORD, keywords=["nonexistent"], weight=1.0)],
    )
    # keyword check fails (0.0), embedding vs a near-identical golden_answer is high
    score = score_task(task, "Paris is the capital of France.")
    assert 0.3 < score < 0.7  # roughly the average of 0.0 and a high embedding score


def test_score_task_weight_zero_total_is_zero():
    task = GoldenTask(input="x", golden_answer="y", golden_answer_weight=0.0)
    # only golden_answer with weight 0 -> total weight 0
    assert score_task(task, "anything") == 0.0


async def test_fitness_weights_tasks_and_uses_run_prompt_on_tasks():
    tasks = [
        GoldenTask(id="a", input="q1", weight=1.0,
                   checks=[Check(type=CheckType.KEYWORD, keywords=["yes"])]),
        GoldenTask(id="b", input="q2", weight=3.0,
                   checks=[Check(type=CheckType.KEYWORD, keywords=["yes"])]),
    ]
    mock = AsyncMock(return_value={"a": "no", "b": "yes indeed"})
    with patch("diffprompt.core.fitness.run_prompt_on_tasks", new=mock):
        score = await fitness("some prompt", tasks)

    # task a fails (weight 1), task b passes (weight 3) -> 3/4
    assert abs(score - 0.75) < 1e-9


async def test_fitness_empty_tasks_is_zero():
    assert await fitness("prompt", []) == 0.0
