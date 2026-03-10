"""
Tests for the slicer module.
Run with: pytest tests/test_slicer.py -v
"""
import pytest
from diffprompt.models import DiffResult, TestCase, TestCategory, Verdict
from diffprompt.core.slicer import compute_slices, _compute_confidence


def make_diff(input_text: str, similarity: float, verdict: Verdict, tags: dict) -> DiffResult:
    """Helper to create a DiffResult for testing."""
    tc = TestCase(
        id=input_text[:6].replace(" ", "_"),
        input=input_text,
        category=TestCategory.TYPICAL,
        tags=tags
    )
    return DiffResult(
        test_case=tc,
        v1_output="v1 output",
        v2_output="v2 output",
        similarity=similarity,
        divergence=1 - similarity,
        verdict=verdict,
        reason="test reason",
        judge_confidence=0.9,
    )


def test_compute_slices_groups_by_tag():
    """compute_slices() should group diffs by tag dimension correctly."""
    diffs = [
        make_diff("formal input 1", 0.9, Verdict.IMPROVEMENT, {"tone": "formal"}),
        make_diff("formal input 2", 0.88, Verdict.IMPROVEMENT, {"tone": "formal"}),
        make_diff("emotional input 1", 0.2, Verdict.REGRESSION, {"tone": "emotional"}),
        make_diff("emotional input 2", 0.18, Verdict.REGRESSION, {"tone": "emotional"}),
        make_diff("emotional input 3", 0.22, Verdict.REGRESSION, {"tone": "emotional"}),
    ]
    slices = compute_slices(diffs)
    labels = [s.label for s in slices]
    assert "tone:formal" in labels
    assert "tone:emotional" in labels


def test_compute_slices_worst_first():
    """compute_slices() should sort slices worst first (lowest mean_similarity)."""
    diffs = [
        make_diff("formal 1", 0.9, Verdict.IMPROVEMENT, {"tone": "formal"}),
        make_diff("formal 2", 0.88, Verdict.IMPROVEMENT, {"tone": "formal"}),
        make_diff("emotional 1", 0.2, Verdict.REGRESSION, {"tone": "emotional"}),
        make_diff("emotional 2", 0.18, Verdict.REGRESSION, {"tone": "emotional"}),
        make_diff("emotional 3", 0.22, Verdict.REGRESSION, {"tone": "emotional"}),
    ]
    slices = compute_slices(diffs)
    assert slices[0].mean_similarity < slices[-1].mean_similarity


def test_compute_slices_empty_diffs():
    """compute_slices() should return empty list for empty input."""
    result = compute_slices([])
    assert result == []


def test_compute_slices_no_tags():
    """compute_slices() should return empty list if diffs have no tags."""
    diffs = [
        make_diff("input 1", 0.5, Verdict.NEUTRAL, {}),
        make_diff("input 2", 0.6, Verdict.NEUTRAL, {}),
    ]
    result = compute_slices(diffs)
    assert result == []


def test_slice_confidence_high_variance_is_low():
    """High variance should result in low confidence."""
    from diffprompt.core.slicer import _compute_confidence
    low_conf = _compute_confidence(n=10, variance=0.3, typical_ratio=0.8)
    high_conf = _compute_confidence(n=10, variance=0.02, typical_ratio=0.8)
    assert low_conf < high_conf


def test_slice_confidence_low_n_is_low():
    """Low n should result in lower confidence than high n."""
    low_n = _compute_confidence(n=2, variance=0.05, typical_ratio=0.8)
    high_n = _compute_confidence(n=20, variance=0.05, typical_ratio=0.8)
    assert low_n < high_n


def test_slice_confidence_zero_typical_ratio():
    """Zero typical ratio (all adversarial) should reduce confidence."""
    all_adversarial = _compute_confidence(n=10, variance=0.05, typical_ratio=0.0)
    all_typical = _compute_confidence(n=10, variance=0.05, typical_ratio=1.0)
    assert all_adversarial < all_typical