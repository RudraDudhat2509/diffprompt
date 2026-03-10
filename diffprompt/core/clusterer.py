"""
Clusters diff results by embedding judge reasons.
Uses HDBSCAN + UMAP. Returns named failure modes.
"""
from __future__ import annotations
import numpy as np
from collections import defaultdict
from diffprompt.models import DiffResult, Cluster, Verdict
from diffprompt.core.embedder import embed


def cluster_diffs(diffs: list[DiffResult]) -> tuple[list[Cluster], list[DiffResult]]:
    """
    Cluster diffs by their judge reasons.
    Returns (clusters, unclustered) where unclustered = label -1 noise points.
    """
    try:
        import hdbscan
        import umap
    except ImportError:
        raise ImportError("Run: pip install hdbscan umap-learn")

    if len(diffs) < 4:
        return [], diffs

    reasons = [d.reason for d in diffs]
    embs = embed(reasons)

    # Reduce dimensions before clustering (HDBSCAN works better in low-dim)
    reducer = umap.UMAP(n_components=min(5, len(diffs) - 2), random_state=42)
    reduced = reducer.fit_transform(embs)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1)
    labels = clusterer.fit_predict(reduced)

    # Group by label
    groups: dict[int, list[int]] = defaultdict(list)
    for i, label in enumerate(labels):
        groups[label].append(i)

    clusters = []
    unclustered = []

    for label, indices in groups.items():
        group_diffs = [diffs[i] for i in indices]

        if label == -1:
            unclustered.extend(group_diffs)
            continue

        name = _name_cluster(label, group_diffs)
        mean_sim = float(np.mean([d.similarity for d in group_diffs]))

        cluster = Cluster(
            label=label,
            name=name,
            description=_describe_cluster(group_diffs),
            n=len(group_diffs),
            mean_similarity=mean_sim,
            test_ids=[d.test_case.id for d in group_diffs],
        )
        clusters.append(cluster)

        # Update cluster_label on each diff
        for d in group_diffs:
            d.cluster_label = label

    clusters.sort(key=lambda c: c.n, reverse=True)
    return clusters, unclustered


def _name_cluster(label: int, diffs: list[DiffResult]) -> str:
    """Generate a short name for a cluster based on dominant reason keywords."""
    reasons = " ".join(d.reason.lower() for d in diffs)

    # Simple keyword matching → named failure modes
    if any(w in reasons for w in ["brief", "short", "concise", "terse"]):
        return "BREVITY_GAIN" if _is_mostly_improvements(diffs) else "BREVITY_LOSS"
    if any(w in reasons for w in ["context", "nuance", "detail", "omit", "missing"]):
        return "CONTEXT_LOSS"
    if any(w in reasons for w in ["refus", "declin", "won't", "cannot"]):
        return "REFUSAL_SHIFT"
    if any(w in reasons for w in ["tone", "empathy", "warm", "cold", "formal"]):
        return "TONE_SHIFT"
    if any(w in reasons for w in ["accura", "wrong", "incorrect", "error", "fact"]):
        return "ACCURACY_CHANGE"
    if any(w in reasons for w in ["verbos", "long", "padded", "unnecessar"]):
        return "VERBOSITY_GAIN"
    return f"CLUSTER_{label}"


def _describe_cluster(diffs: list[DiffResult]) -> str:
    """One-line description from most common reason words."""
    all_reasons = ". ".join(d.reason for d in diffs[:3])
    return all_reasons[:120]


def _is_mostly_improvements(diffs: list[DiffResult]) -> bool:
    improvements = sum(1 for d in diffs if d.verdict == Verdict.IMPROVEMENT)
    return improvements > len(diffs) / 2
