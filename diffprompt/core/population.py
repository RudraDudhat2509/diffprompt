"""
Population init, crossover, and selection for `diffprompt evolve`.
Rule-based, no LLM involved anywhere in this module.
"""
from __future__ import annotations
import random

from diffprompt.core.mutate import TRANSFORMS, apply_transform, mutate, to_lines

_MUTATION_RATE = 0.3


def init_population(base_prompt: str, n: int, rng: random.Random) -> list[str]:
    """
    Variant 0 is always the literal, unmodified starting prompt — evolution
    can never end up worse than the baseline. The rest apply one distinct
    template transform each, cycling through the menu if n exceeds its size.
    """
    if n <= 1:
        return [base_prompt]

    names = list(TRANSFORMS)
    variants = [base_prompt]
    i = 0
    while len(variants) < n:
        variants.append(apply_transform(base_prompt, names[i % len(names)], rng))
        i += 1
    return variants


def crossover(parent_a: str, parent_b: str, rng: random.Random) -> str:
    """Single-point splice: head of A's instructions + tail of B's."""
    lines_a = to_lines(parent_a)
    lines_b = to_lines(parent_b)
    if not lines_a:
        return parent_b
    if not lines_b:
        return parent_a

    ratio = rng.random()
    cut_a = max(1, min(len(lines_a), round(len(lines_a) * ratio))) if len(lines_a) > 1 else len(lines_a)
    cut_b = min(len(lines_b), round(len(lines_b) * ratio))

    child_lines = lines_a[:cut_a] + lines_b[cut_b:]
    return "\n".join(child_lines) if child_lines else parent_a


def select(scored: list[tuple[str, float]], k: int) -> list[tuple[str, float]]:
    """Keep the top-k (prompt, fitness) pairs by fitness, descending."""
    return sorted(scored, key=lambda pair: pair[1], reverse=True)[:max(1, k)]


def breed_next_generation(
    survivors: list[tuple[str, float]],
    elite_prompt: str,
    population_size: int,
    rng: random.Random,
    mutation_rate: float = _MUTATION_RATE,
) -> list[str]:
    """
    Elitism: elite_prompt is always carried forward unmodified as slot 0.
    The rest are bred from survivors via crossover, then mutated with
    probability `mutation_rate`.
    """
    next_gen = [elite_prompt]
    pool = [p for p, _ in survivors] or [elite_prompt]

    while len(next_gen) < population_size:
        a, b = rng.choice(pool), rng.choice(pool)
        child = crossover(a, b, rng)
        if rng.random() < mutation_rate:
            child = mutate(child, rng)
        next_gen.append(child)

    return next_gen[:population_size]
