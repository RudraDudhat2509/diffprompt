"""
Test case generation using taxonomy-based prompting.
Four buckets: typical, boundary, adversarial, format.
"""
from __future__ import annotations
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
    """Generate n test cases across all taxonomy buckets."""
    test_cases: list[TestCase] = []

    for category, fraction in DISTRIBUTION.items():
        count = max(2, round(n * fraction))
        
        # Try up to 2 times if JSON parsing fails
        inputs = None
        for attempt in range(2):
            raw, _ = await call_cascade(
                TAXONOMY_PROMPTS[category].format(n=count, prompt=prompt),
                local_only=local_only,
            )
            clean = re.sub(r"```(?:json)?|```", "", raw).strip()
            # Extract just the JSON array if there's surrounding text
            match = re.search(r'\[.*\]', clean, re.DOTALL)
            if match:
                clean = match.group(0)
            try:
                inputs = json.loads(clean)
                break
            except json.JSONDecodeError:
                if attempt == 1:
                    inputs = []  # give up, skip this bucket
    
        for inp in (inputs or [])[:count]:  # In case LLM returns more than requested
            test_cases.append(TestCase(
                id=str(uuid.uuid4()),
                input=inp,
                category=category,
            ))
        return test_cases


def diversity_score(test_cases: list[TestCase], embedder) -> float:
    """
    Compute diversity score for the test suite.
    1 - mean pairwise similarity. Higher = more diverse.
    """
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    inputs = [tc.input for tc in test_cases]
    embs = embedder.encode(inputs)
    sim_matrix = cosine_similarity(embs)
    np.fill_diagonal(sim_matrix, 0)
    mean_sim = sim_matrix.sum() / (len(inputs) * (len(inputs) - 1))
    return float(1 - mean_sim)
