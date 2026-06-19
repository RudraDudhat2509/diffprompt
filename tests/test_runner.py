"""
Tests for the runner module.
Run with: pytest tests/test_runner.py -v
"""
from unittest.mock import AsyncMock, patch

from diffprompt.core.runner import _split_model, run_single
from diffprompt.models import TestCase, TestCategory


def test_split_model_groq():
    """A groq/ prefix maps to the groq_model kwarg."""
    assert _split_model("groq/llama-3.3-70b-versatile") == {
        "groq_model": "llama-3.3-70b-versatile"
    }


def test_split_model_local():
    """Both local/ and ollama/ prefixes map to the local_model kwarg."""
    assert _split_model("local/qwen2.5:7b") == {"local_model": "qwen2.5:7b"}
    assert _split_model("ollama/qwen2.5:7b") == {"local_model": "qwen2.5:7b"}


def test_split_model_bare_or_unknown_falls_back():
    """Bare names and unknown providers return no override (cascade defaults)."""
    assert _split_model("llama-3.3-70b") == {}
    assert _split_model("") == {}
    assert _split_model("azure/gpt-4o") == {}


async def test_run_single_forwards_model_to_cascade():
    """
    Regression for #2: --model must reach call_cascade. Previously run_single
    received the model arg and dropped it, so call_cascade always ran its
    hardcoded default (llama-3.1-8b-instant).
    """
    tc = TestCase(id="t1", input="hi", category=TestCategory.TYPICAL)
    mock = AsyncMock(return_value=("out", "groq/llama-3.3-70b-versatile"))
    with patch("diffprompt.core.runner.call_cascade", new=mock):
        await run_single(tc, "system prompt", "v1", "groq/llama-3.3-70b-versatile")

    _, kwargs = mock.call_args
    assert kwargs.get("groq_model") == "llama-3.3-70b-versatile"
