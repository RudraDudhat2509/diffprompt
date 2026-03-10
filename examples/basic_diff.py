"""
Basic example — diff two prompts with auto-generated test cases.

Run with:
  python examples/basic_diff.py

Requirements:
  - Ollama running locally with qwen2.5:7b pulled
  - OR GROQ_API_KEY set in your environment
"""
import asyncio
from diffprompt.core.ontology import Ontology
from diffprompt.core.generator import generate_test_cases, diversity_score
from diffprompt.core.runner import run_both
from diffprompt.core.embedder import batch_similarity
from diffprompt.core.judge import judge_single
from diffprompt.core.clusterer import cluster_diffs
from diffprompt.core.slicer import compute_slices
from diffprompt.core.scorer import regression_score, select_key_examples
from diffprompt.models import DiffResult, Verdict


PROMPT_V1 = "You are a helpful assistant. Answer clearly and completely."
PROMPT_V2 = "You are a concise assistant. Be brief."


async def main():
    print("diffprompt basic example")
    print("=" * 40)

    # Step 1 — Build ontology
    print("\n[1/5] Inferring ontology...")
    ontology = Ontology()
    await ontology.infer(PROMPT_V1)
    await ontology.build_anchors(PROMPT_V1)
    print(f"      Dimensions: {list(ontology.dimensions.keys())}")

    # Step 2 — Generate test cases
    print("\n[2/5] Generating test cases...")
    test_cases = await generate_test_cases(PROMPT_V1, n=10, ontology=ontology)
    div = diversity_score(test_cases, ontology.embedder)
    print(f"      Generated {len(test_cases)} test cases  diversity={div:.2f}")

    # Step 3 — Run both prompts
    print("\n[3/5] Running both prompts...")
    v1_results, v2_results = await run_both(test_cases, PROMPT_V1, PROMPT_V2)
    print(f"      Completed {len(test_cases) * 2} completions")

    # Step 4 — Semantic diff
    print("\n[4/5] Computing semantic diff...")
    pairs = [(v1_results[tc.id].output, v2_results[tc.id].output) for tc in test_cases]
    similarities = batch_similarity(pairs)

    diffs = []
    for i, tc in enumerate(test_cases):
        sim = similarities[i]
        v1_out = v1_results[tc.id].output
        v2_out = v2_results[tc.id].output
        verdict, reason, confidence = await judge_single(tc, v1_out, v2_out, sim)
        diffs.append(DiffResult(
            test_case=tc,
            v1_output=v1_out,
            v2_output=v2_out,
            similarity=sim,
            divergence=1 - sim,
            verdict=verdict,
            reason=reason,
            judge_confidence=confidence,
        ))

    # Step 5 — Results
    print("\n[5/5] Results")
    print("-" * 40)

    score = regression_score(diffs)
    n_improved  = sum(1 for d in diffs if d.verdict == Verdict.IMPROVEMENT)
    n_regressed = sum(1 for d in diffs if d.verdict == Verdict.REGRESSION)
    n_neutral   = sum(1 for d in diffs if d.verdict == Verdict.NEUTRAL)

    print(f"Regression score: {score}/100")
    print(f"Improved:  {n_improved}")
    print(f"Regressed: {n_regressed}")
    print(f"Neutral:   {n_neutral}")

    print("\nTop regressions:")
    regressions = sorted(
        [d for d in diffs if d.verdict == Verdict.REGRESSION],
        key=lambda d: d.divergence,
        reverse=True
    )
    for d in regressions[:3]:
        print(f"  [{d.similarity:.2f}] {d.test_case.input[:60]}")
        print(f"         reason: {d.reason}")


if __name__ == "__main__":
    asyncio.run(main())