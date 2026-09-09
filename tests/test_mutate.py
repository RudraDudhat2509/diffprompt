"""
Tests for the mutate module.
Run with: pytest tests/test_mutate.py -v
"""
import random

from diffprompt.core.mutate import TRANSFORMS, apply_transform, mutate, to_lines


def test_to_lines_strips_blank_lines():
    assert to_lines("a\n\n  b  \n\nc") == ["a", "b", "c"]


def test_reorder_keeps_first_line_fixed():
    rng = random.Random(1)
    lines = ["You are a helpful assistant.", "Be concise.", "Cite sources.", "Be polite."]
    result = apply_transform("\n".join(lines), "reorder", rng)
    result_lines = to_lines(result)
    assert result_lines[0] == lines[0]
    assert sorted(result_lines) == sorted(lines)


def test_reorder_noop_on_short_prompt():
    rng = random.Random(1)
    result = apply_transform("Only one line.", "reorder", rng)
    assert result == "Only one line."


def test_tighten_word_limit_shrinks_existing_number():
    rng = random.Random(1)
    result = apply_transform("Respond in under 100 words.", "tighten_word_limit", rng)
    assert "70 words" in result


def test_tighten_word_limit_appends_when_absent():
    rng = random.Random(1)
    result = apply_transform("Be helpful.", "tighten_word_limit", rng)
    assert "words" in result.lower()
    assert "Be helpful." in result


def test_explicit_constraint_converts_soft_language():
    rng = random.Random(1)
    result = apply_transform("Be concise in your answers.", "explicit_constraint", rng)
    assert "3 sentences or fewer" in result


def test_explicit_constraint_fallback_when_no_soft_language():
    rng = random.Random(1)
    result = apply_transform("Answer the question.", "explicit_constraint", rng)
    assert "Follow every constraint above exactly as written." in result


def test_add_format_appends_a_directive():
    rng = random.Random(1)
    result = apply_transform("Answer the question.", "add_format", rng)
    assert result != "Answer the question."


def test_remove_format_strips_matching_lines():
    rng = random.Random(1)
    result = apply_transform("Be helpful.\nRespond in valid JSON.", "remove_format", rng)
    assert "json" not in result.lower()
    assert "Be helpful." in result


def test_remove_format_noop_when_nothing_matches():
    rng = random.Random(1)
    result = apply_transform("Be helpful.", "remove_format", rng)
    assert result == "Be helpful."


def test_mutate_picks_from_fixed_menu_only():
    rng = random.Random(1)
    prompt = "You are a helpful assistant.\nBe concise."
    for _ in range(20):
        result = mutate(prompt, rng)
        assert isinstance(result, str)
        assert result.strip() != ""


def test_transforms_menu_is_fixed_set():
    assert set(TRANSFORMS) == {
        "reorder", "tighten_word_limit", "explicit_constraint", "add_format", "remove_format",
    }
