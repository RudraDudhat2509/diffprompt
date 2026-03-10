"""
Runs both prompts on all test cases.
Returns RunResult for each (test_case, prompt_version) pair.
"""
from __future__ import annotations
import asyncio
import time
from diffprompt.models import TestCase, RunResult
from diffprompt.models.cascade import call_cascade


async def run_single(
    test_case: TestCase,
    prompt: str,
    version: str,
    model: str,
    local_only: bool = False,
) -> RunResult:
    start = time.monotonic()
    output, model_used = await call_cascade(
        test_case.input,
        system=prompt,
        local_only=local_only,
    )
    latency_ms = (time.monotonic() - start) * 1000

    return RunResult(
        test_id=test_case.id,
        prompt_version=version,
        output=output,
        model_used=model_used,
        latency_ms=latency_ms,
    )


async def run_both(
    test_cases: list[TestCase],
    prompt_v1: str,
    prompt_v2: str,
    model: str = "groq/llama-3.3-70b-versatile",
    local_only: bool = False,
    concurrency: int = 5,
) -> tuple[dict[str, RunResult], dict[str, RunResult]]:
    """
    Run both prompts on all test cases concurrently.
    Returns (v1_results, v2_results) as dicts keyed by test_id.
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def run_with_sem(tc, prompt, version):
        async with semaphore:
            return await run_single(tc, prompt, version, model, local_only)

    # Fire all tasks concurrently (bounded by semaphore)
    v1_tasks = [run_with_sem(tc, prompt_v1, "v1") for tc in test_cases]
    v2_tasks = [run_with_sem(tc, prompt_v2, "v2") for tc in test_cases]

    v1_raw, v2_raw = await asyncio.gather(
        asyncio.gather(*v1_tasks),
        asyncio.gather(*v2_tasks),
    )

    v1_results = {r.test_id: r for r in v1_raw}
    v2_results = {r.test_id: r for r in v2_raw}
    return v1_results, v2_results
