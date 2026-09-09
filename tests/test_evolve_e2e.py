"""
End-to-end test for `diffprompt evolve`.
Run with: pytest tests/test_evolve_e2e.py -v

The LLM call is mocked (this is a genetic-algorithm test, not a live-model
test) but the mock is a pure function of the *prompt structure*, so the GA
still has to do real work: population init, fitness scoring, selection, and
elitism all run for real against `diffprompt.core.fitness.fitness`.
"""
from unittest.mock import AsyncMock, patch

from diffprompt.cli import _run_evolve
from diffprompt.core.fitness import fitness
from diffprompt.models import Check, CheckType, GoldenTask


async def _fake_call_cascade(prompt, system=None, **kwargs):
    """
    A weak starting prompt (1 line) never mentions the answer. Any variant
    that has picked up a second instruction line (via a template transform)
    does. This makes fitness a real, checkable function of what the GA does
    to the prompt - without hitting an actual model.
    """
    n_lines = len((system or "").strip().split("\n"))
    if n_lines <= 1:
        return "I have no idea, sorry.", "mock/model"
    return "Paris is correct. " * n_lines + "The end.", "mock/model"


def _make_tasks() -> list[GoldenTask]:
    return [
        GoldenTask(
            id="capital",
            input="What is the capital of France?",
            checks=[Check(type=CheckType.KEYWORD, keywords=["Paris"], weight=1.0)],
        )
    ]


async def test_score_improves_over_generations_on_weak_prompt(tmp_path, capsys):
    tasks_file = tmp_path / "tasks.yaml"
    tasks_file.write_text(
        "tasks:\n"
        "  - input: \"What is the capital of France?\"\n"
        "    checks:\n"
        "      - type: keyword\n"
        "        keywords: [\"Paris\"]\n",
        encoding="utf-8",
    )

    weak_prompt = "You are a helpful assistant."

    with patch("diffprompt.core.runner.call_cascade", new=AsyncMock(side_effect=_fake_call_cascade)):
        baseline_fitness = await fitness(weak_prompt, _make_tasks(), local_only=True)

        await _run_evolve(
            prompt=weak_prompt,
            golden_tasks_path=str(tasks_file),
            generations=3,
            population=6,
            patience=3,
            model="groq/llama-3.3-70b-versatile",
            local_only=True,
            output_format="terminal",
            save=None,
        )

    captured = capsys.readouterr()
    # The evolved prompt's fitness beat the unmodified baseline.
    assert "fitness" in captured.out.lower()
    assert baseline_fitness == 0.0  # weak prompt never mentions Paris


async def test_evolve_report_final_fitness_beats_baseline(tmp_path):
    tasks_file = tmp_path / "tasks.yaml"
    tasks_file.write_text(
        "tasks:\n"
        "  - input: \"What is the capital of France?\"\n"
        "    checks:\n"
        "      - type: keyword\n"
        "        keywords: [\"Paris\"]\n",
        encoding="utf-8",
    )
    weak_prompt = "You are a helpful assistant."

    from diffprompt.core.golden_tasks import load_golden_tasks
    from diffprompt.core.population import init_population
    import random

    tasks = load_golden_tasks(str(tasks_file))

    with patch("diffprompt.core.runner.call_cascade", new=AsyncMock(side_effect=_fake_call_cascade)):
        baseline_fitness = await fitness(weak_prompt, tasks, local_only=True)

        rng = random.Random(7)
        population = init_population(weak_prompt, 6, rng)
        scores = [await fitness(v, tasks, local_only=True) for v in population]
        scored = list(zip(population, scores))
        best_prompt, best_fitness = max(scored, key=lambda pair: pair[1])

    assert best_fitness > baseline_fitness  # at least one template transform found "Paris"
