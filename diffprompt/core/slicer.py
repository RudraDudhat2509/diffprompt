"""
Behavioral slicing.
Groups diffs by input tags and computes per-slice performance.
Recursive splitting for high-variance slices (max depth 3).
"""
from __future__ import annotations
import numpy as np
from diffprompt.models import DiffResult, SliceResult, Verdict, TestCategory


MIN_SLICE_SIZE = 5
MAX_DEPTH = 3
VARIANCE_THRESHOLD = 0.1   # stop splitting if variance is already low
HIGH_VARIANCE_THRESHOLD = 0.15  # trigger recursive split above this


def compute_slices(diffs: list[DiffResult]) -> list[SliceResult]:
    """Compute behavioral slices across all tag dimensions."""
    if not diffs or not diffs[0].test_case.tags:
        return []

    dimensions = list(diffs[0].test_case.tags.keys())
    slices = []

    for dimension in dimensions:
        # Group by tag value
        groups: dict[str, list[DiffResult]] = {}
        for d in diffs:
            val = d.test_case.tags.get(dimension, "unknown")
            groups.setdefault(val, []).append(d)

        for value, group_diffs in groups.items():
            slice_result = _compute_slice(
                dimension=dimension,
                value=value,
                diffs=group_diffs,
                depth=1,
            )
            if slice_result:
                slices.append(slice_result)

            # Recursive split if high variance + enough data + not too deep
            if (
                slice_result
                and slice_result.variance > HIGH_VARIANCE_THRESHOLD
                and len(group_diffs) >= MIN_SLICE_SIZE * 2
                and slice_result.depth < MAX_DEPTH
            ):
                sub_slices = _recursive_split(group_diffs, dimension, depth=2)
                slices.extend(sub_slices)

    # Sort by mean_similarity ascending (worst slices first)
    slices.sort(key=lambda s: s.mean_similarity)
    return slices


def _compute_slice(
    dimension: str,
    value: str,
    diffs: list[DiffResult],
    depth: int,
) -> SliceResult | None:
    if len(diffs) < 2:
        return None

    sims = [d.similarity for d in diffs]
    mean_sim = float(np.mean(sims))
    variance = float(np.var(sims))

    typical_ratio = sum(
        1 for d in diffs if d.test_case.category == TestCategory.TYPICAL
    ) / len(diffs)

    confidence = _compute_confidence(
        n=len(diffs),
        variance=variance,
        typical_ratio=typical_ratio,
    )

    verdicts = [d.verdict for d in diffs]
    verdict = _aggregate_verdict(verdicts)

    return SliceResult(
        dimension=dimension,
        value=value,
        label=f"{dimension}:{value}",
        n=len(diffs),
        mean_similarity=mean_sim,
        variance=variance,
        typical_ratio=typical_ratio,
        confidence=confidence,
        verdict=verdict,
        depth=depth,
    )


def _recursive_split(
    diffs: list[DiffResult],
    parent_dimension: str,
    depth: int,
) -> list[SliceResult]:
    """Try splitting a high-variance slice by other dimensions."""
    if depth > MAX_DEPTH or not diffs:
        return []

    other_dimensions = [
        k for k in diffs[0].test_case.tags.keys()
        if k != parent_dimension
    ]

    results = []
    for dimension in other_dimensions:
        groups: dict[str, list[DiffResult]] = {}
        for d in diffs:
            val = d.test_case.tags.get(dimension, "unknown")
            groups.setdefault(val, []).append(d)

        for value, group in groups.items():
            if len(group) < MIN_SLICE_SIZE:
                continue
            slice_result = _compute_slice(
                dimension=f"{parent_dimension}+{dimension}",
                value=value,
                diffs=group,
                depth=depth,
            )
            if slice_result:
                results.append(slice_result)

    return results


def _compute_confidence(n: int, variance: float, typical_ratio: float) -> float:
    """
    Confidence = f(variance, typical_ratio, n).
    Variance is checked first — high variance = impure slice = low confidence.
    """
    # Variance penalty (primary signal)
    variance_score = max(0.0, 1.0 - variance * 5)

    # Typical ratio bonus (real-world signal)
    typical_score = typical_ratio

    # N score (statistical power)
    n_score = min(1.0, n / 20)

    return float(0.5 * variance_score + 0.3 * typical_score + 0.2 * n_score)


def _aggregate_verdict(verdicts: list[Verdict]) -> Verdict:
    counts = {v: verdicts.count(v) for v in Verdict}
    return max(counts, key=counts.get)
