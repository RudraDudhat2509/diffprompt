"""
Tests for the population module.
Run with: pytest tests/test_population.py -v
"""
import random

from diffprompt.core.population import breed_next_generation, crossover, init_population, select


def test_init_population_first_variant_is_unmodified():
    rng = random.Random(1)
    base = "You are a helpful assistant.\nBe concise."
    pop = init_population(base, 5, rng)
    assert pop[0] == base
    assert len(pop) == 5


def test_init_population_produces_distinct_variants():
    rng = random.Random(1)
    base = "You are a helpful assistant.\nBe concise.\nCite sources."
    pop = init_population(base, 5, rng)
    assert len(set(pop)) > 1  # not all identical


def test_init_population_n_one_returns_base_only():
    rng = random.Random(1)
    assert init_population("hello", 1, rng) == ["hello"]


def test_init_population_cycles_menu_when_n_exceeds_it():
    rng = random.Random(1)
    base = "You are a helpful assistant."
    pop = init_population(base, 12, rng)
    assert len(pop) == 12


def test_crossover_splices_lines_from_both_parents():
    rng = random.Random(2)
    a = "Line A1\nLine A2\nLine A3"
    b = "Line B1\nLine B2\nLine B3"
    child = crossover(a, b, rng)
    child_lines = set(child.split("\n"))
    assert child_lines & set(a.split("\n")) or child_lines & set(b.split("\n"))


def test_crossover_handles_empty_parent():
    rng = random.Random(1)
    assert crossover("", "hello", rng) == "hello"
    assert crossover("hello", "", rng) == "hello"


def test_select_keeps_top_k_by_fitness():
    scored = [("a", 0.1), ("b", 0.9), ("c", 0.5), ("d", 0.3)]
    top2 = select(scored, 2)
    assert top2 == [("b", 0.9), ("c", 0.5)]


def test_breed_next_generation_keeps_elite_unmodified_at_slot_zero():
    rng = random.Random(1)
    survivors = [("prompt A", 0.9), ("prompt B", 0.5)]
    next_gen = breed_next_generation(survivors, elite_prompt="prompt A", population_size=5, rng=rng)
    assert next_gen[0] == "prompt A"
    assert len(next_gen) == 5


def test_breed_next_generation_respects_population_size():
    rng = random.Random(1)
    survivors = [("You are helpful.\nBe concise.", 0.9)]
    next_gen = breed_next_generation(survivors, elite_prompt="You are helpful.\nBe concise.",
                                      population_size=3, rng=rng)
    assert len(next_gen) == 3
