"""
Tests for the embedder module.
Run with: pytest tests/test_embedder.py -v
"""
import pytest
from diffprompt.core.embedder import embed, similarity, batch_similarity


def test_embed_returns_correct_shape():
    """embed() should return one vector per input text."""
    texts = ["hello world", "goodbye world", "foo bar"]
    result = embed(texts)
    assert result.shape[0] == 3       # one row per text
    assert result.shape[1] == 384     # all-MiniLM-L6-v2 produces 384-dim vectors


def test_similarity_identical_texts():
    """Two identical texts should have similarity close to 1.0."""
    score = similarity("The cat sat on the mat", "The cat sat on the mat")
    assert score > 0.99


def test_similarity_unrelated_texts():
    """Very different texts should have low similarity."""
    score = similarity(
        "I love eating pizza on weekends",
        "The quarterly earnings report exceeded expectations"
    )
    assert score < 0.5


def test_similarity_related_texts():
    """Semantically similar texts should have high similarity."""
    score = similarity(
        "Paris is the capital of France",
        "France's capital city is Paris"
    )
    assert score > 0.85


def test_similarity_returns_float_between_0_and_1():
    """similarity() should always return a float between 0 and 1."""
    score = similarity("hello", "world")
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_batch_similarity_matches_individual():
    """batch_similarity() should give same results as calling similarity() individually."""
    pairs = [
        ("Paris is the capital of France", "France's capital is Paris"),
        ("I love pizza", "The stock market crashed today"),
        ("Hello world", "Hello world"),
    ]

    batch_scores = batch_similarity(pairs)
    individual_scores = [similarity(a, b) for a, b in pairs]

    assert len(batch_scores) == 3
    for batch, individual in zip(batch_scores, individual_scores):
        assert abs(batch - individual) < 0.001  # should be nearly identical


def test_batch_similarity_correct_length():
    """batch_similarity() should return one score per pair."""
    pairs = [("a", "b"), ("c", "d"), ("e", "f"), ("g", "h")]
    scores = batch_similarity(pairs)
    assert len(scores) == 4


def test_embed_caches_model():
    """Calling embed() multiple times should use the same cached model."""
    from diffprompt.core.embedder import get_embedder
    model_1 = get_embedder()
    model_2 = get_embedder()
    assert model_1 is model_2  # exact same object, not a copy