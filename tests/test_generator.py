"""
Tests for the generator module.
Run with: pytest tests/test_generator.py -v
"""
import pytest
from diffprompt.models import TestCase, TestCategory
from diffprompt.core.generator import diversity_score


def make_test_cases(inputs: list[str], category=TestCategory.TYPICAL) -> list[TestCase]:
    """Helper to quickly create TestCase objects for testing."""
    return [
        TestCase(id=str(i), input=inp, category=category, tags={})
        for i, inp in enumerate(inputs)
    ]


def test_diversity_score_identical_inputs():
    """Identical inputs should have very low diversity."""
    from diffprompt.core.embedder import get_embedder
    cases = make_test_cases([
        "What is the capital of France?",
        "What is the capital of France?",
        "What is the capital of France?",
    ])
    score = diversity_score(cases, get_embedder())
    assert score < 0.1


def test_diversity_score_different_inputs():
    """Very different inputs should have high diversity."""
    from diffprompt.core.embedder import get_embedder
    cases = make_test_cases([
        "What is the capital of France?",
        "I've been feeling really anxious lately and need help",
        "Explain the quicksort algorithm in Python",
        "Write a haiku about autumn leaves",
        "What are the main causes of World War 1?",
    ])
    score = diversity_score(cases, get_embedder())
    assert score > 0.5


def test_diversity_score_returns_float_between_0_and_1():
    """diversity_score() should always return a float between 0 and 1."""
    from diffprompt.core.embedder import get_embedder
    cases = make_test_cases(["hello", "world", "foo"])
    score = diversity_score(cases, get_embedder())
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_test_case_creation():
    """TestCase should be created correctly with all fields."""
    tc = TestCase(
        id="abc123",
        input="What is Python?",
        category=TestCategory.TYPICAL,
        tags={"tone": "neutral", "complexity": "simple"}
    )
    assert tc.id == "abc123"
    assert tc.input == "What is Python?"
    assert tc.category == TestCategory.TYPICAL
    assert tc.tags["tone"] == "neutral"


def test_test_case_default_tags():
    """TestCase should default to empty tags if none provided."""
    tc = TestCase(id="x", input="hello", category=TestCategory.FORMAT)
    assert tc.tags == {}


def test_test_category_values():
    """TestCategory enum should have exactly four values."""
    categories = list(TestCategory)
    assert len(categories) == 4
    assert TestCategory.TYPICAL in categories
    assert TestCategory.ADVERSARIAL in categories
    assert TestCategory.BOUNDARY in categories
    assert TestCategory.FORMAT in categories