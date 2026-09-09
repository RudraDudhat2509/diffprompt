"""
Fitness scoring for `diffprompt evolve`.

Deterministic only — regex/keyword/json_schema/numeric checks, plus embedding
similarity (local all-MiniLM-L6-v2, no generative model) against a golden
answer. `run_prompt_on_tasks` calls an LLM to produce the output text being
scored — that's generation, not judging, and is the only LLM call in this path.
"""
from __future__ import annotations

from diffprompt.core import embedder
from diffprompt.core.checks import score_check
from diffprompt.core.runner import run_prompt_on_tasks
from diffprompt.models import GoldenTask


def score_task(task: GoldenTask, output: str) -> float:
    parts = [(c.weight, score_check(c, output)) for c in task.checks]
    if task.golden_answer:
        parts.append((task.golden_answer_weight, embedder.similarity(output, task.golden_answer)))

    total_weight = sum(w for w, _ in parts)
    if total_weight == 0:
        return 0.0
    return sum(w * s for w, s in parts) / total_weight


async def fitness(
    prompt: str,
    tasks: list[GoldenTask],
    model: str = "groq/llama-3.3-70b-versatile",
    local_only: bool = False,
    concurrency: int = 5,
) -> float:
    if not tasks:
        return 0.0

    outputs = await run_prompt_on_tasks(
        tasks, prompt, model=model, local_only=local_only, concurrency=concurrency,
    )

    total_task_weight = sum(t.weight for t in tasks)
    if total_task_weight == 0:
        return 0.0

    weighted_sum = sum(t.weight * score_task(t, outputs[t.id]) for t in tasks)
    return weighted_sum / total_task_weight
