"""
Test case generation using taxonomy-based prompting.
Four buckets: typical, boundary, adversarial, format.
"""
from __future__ import annotations
import asyncio
import json
import re
import uuid
from typing import Optional

from diffprompt.models import TestCase, TestCategory
from diffprompt.models.cascade import call_cascade
from diffprompt.core.ontology import Ontology


TAXONOMY_PROMPTS = {
    TestCategory.TYPICAL: """Generate {n} realistic inputs a real user would send to this prompt.
These should reflect actual everyday usage — diverse topics, normal phrasing.
Prompt: {prompt}
Return ONLY a JSON array of strings. No explanation.""",

    TestCategory.BOUNDARY: """Generate {n} inputs at the edges of what this prompt handles.
Too long, too short, tangentially related, slightly out of scope.
The prompt should handle these but might struggle.
Prompt: {prompt}
Return ONLY a JSON array of strings. No explanation.""",

    TestCategory.ADVERSARIAL: """Generate {n} inputs designed to expose inconsistencies or failures.
Use ambiguous phrasing, contradictory requirements, edge cases, trick questions.
Make the prompt work hard.
Prompt: {prompt}
Return ONLY a JSON array of strings. No explanation.""",

    TestCategory.FORMAT: """Generate {n} inputs with unusual formatting.
Try: ALL CAPS, no punctuation, mixed languages, extremely long sentences,
bullet points as input, emojis, very short (1-2 words), JSON-formatted input.
Prompt: {prompt}
Return ONLY a JSON array of strings. No explanation.""",
}

# Distribution: 45% typical, 35% adversarial, 10% boundary, 10% format
DISTRIBUTION = {
    TestCategory.TYPICAL:     0.45,
    TestCategory.ADVERSARIAL: 0.35,
    TestCategory.BOUNDARY:    0.10,
    TestCategory.FORMAT:      0.10,
}


async def generate_test_cases(
    prompt: str,
    n: int = 40,
    ontology: Optional[Ontology] = None,
    local_only: bool = False,
) -> list[TestCase]:
    """Generate n test cases across all taxonomy buckets, all buckets in parallel."""
    categories = list(DISTRIBUTION.items())
    counts = [max(2, round(n * fraction)) for _, fraction in categories]

    # Fire all four bucket generations concurrently instead of one at a time.
    bucket_inputs = await asyncio.gather(*[
        _generate_bucket(prompt, category, count, local_only)
        for (category, _), count in zip(categories, counts)
    ])

    test_cases: list[TestCase] = []
    for (category, _), count, inputs in zip(categories, counts, bucket_inputs):
        for inp in inputs[:count]:
            test_cases.append(TestCase(
                id=str(uuid.uuid4()),
                input=inp,
                category=category,
            ))

    return test_cases


async def _generate_bucket(
    prompt: str,
    category: TestCategory,
    count: int,
    local_only: bool,
) -> list[str]:
    """Generate one bucket of test cases with up to 2 retries on parse failure."""
    for attempt in range(2):
        raw, _ = await call_cascade(
            TAXONOMY_PROMPTS[category].format(n=count, prompt=prompt),
            local_only=local_only,
        )
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        match = re.search(r'\[.*\]', clean, re.DOTALL)
        if match:
            clean = match.group(0)
        try:
            parsed = json.loads(clean)
            if isinstance(parsed, list):
                return [str(x) for x in parsed if x]
        except json.JSONDecodeError:
            if attempt == 1:
                return []  # give up after 2 attempts

    return []


def diversity_score(test_cases: list[TestCase]) -> float:
    """
    Compute diversity score for the test suite.
    1 - mean pairwise similarity. Higher = more diverse.
    Uses the shared embedder from diffprompt.core.embedder to avoid
    loading a second model instance.
    """
    from diffprompt.core.embedder import embed
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    if len(test_cases) < 2:
        return 1.0

    inputs = [tc.input for tc in test_cases]
    embs = embed(inputs)
    sim_matrix = cosine_similarity(embs)
    np.fill_diagonal(sim_matrix, 0)
    n = len(inputs)
    mean_sim = sim_matrix.sum() / (n * (n - 1))
    return float(1 - mean_sim)