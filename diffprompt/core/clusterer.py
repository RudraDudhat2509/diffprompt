"""
Clusters diff results by embedding judge reasons.
Uses HDBSCAN + UMAP. Returns named failure modes.
"""
from __future__ import annotations
import numpy as np
from collections import defaultdict
from diffprompt.models import DiffResult, Cluster, Verdict
from diffprompt.core.embedder import embed

# Below this count, clustering is meaningless — return everything as unclustered
_MIN_CLUSTER_INPUT = 10


def cluster_diffs(diffs: list[DiffResult]) -> tuple[list[Cluster], list[DiffResult]]:
    """
    Cluster diffs by their judge reasons.
    Returns (clusters, unclustered) where unclustered = HDBSCAN noise (label -1).
    """
    try:
        import hdbscan
        import umap
    except ImportError:
        raise ImportError("Run: pip install hdbscan umap-learn")

    if len(diffs) < _MIN_CLUSTER_INPUT:
        return [], diffs

    reasons = [d.reason for d in diffs]
    embs = embed(reasons)

    # UMAP: reduce to low-dim before HDBSCAN (better cluster quality)
    n_components = min(5, len(diffs) - 2)
    reducer = umap.UMAP(n_components=n_components, random_state=42, verbose=False)
    reduced = reducer.fit_transform(embs)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1)
    labels = clusterer.fit_predict(reduced)

    # Compute centrality per point within its cluster
    centrality_map = _compute_centrality(embs, labels)

    groups: dict[int, list[int]] = defaultdict(list)
    for i, label in enumerate(labels):
        groups[label].append(i)

    clusters = []
    unclustered = []

    for label, indices in groups.items():
        group_diffs = [diffs[i] for i in indices]

        # Update cluster metadata on each diff
        for i_local, i_global in enumerate(indices):
            diffs[i_global].cluster_label = label
            diffs[i_global].cluster_centrality = centrality_map.get(i_global, 0.0)

        if label == -1:
            unclustered.extend(group_diffs)
            continue

        cluster = Cluster(
            label=label,
            name=_name_cluster(label, group_diffs),
            description=_describe_cluster(group_diffs),
            n=len(group_diffs),
            mean_similarity=float(np.mean([d.similarity for d in group_diffs])),
            test_ids=[d.test_case.id for d in group_diffs],
        )
        clusters.append(cluster)

    clusters.sort(key=lambda c: c.n, reverse=True)
    return clusters, unclustered


def _compute_centrality(embs: np.ndarray, labels: np.ndarray) -> dict[int, float]:
    """
    For each point, compute its mean cosine similarity to others in its cluster.
    Returns a dict: index → centrality score (0-1).
    """
    from sklearn.metrics.pairwise import cosine_similarity

    centrality = {}
    unique_labels = set(labels)

    for label in unique_labels:
        if label == -1:
            continue
        indices = [i for i, l in enumerate(labels) if l == label]
        if len(indices) < 2:
            for i in indices:
                centrality[i] = 1.0
            continue
        cluster_embs = embs[indices]
        sim_matrix = cosine_similarity(cluster_embs)
        np.fill_diagonal(sim_matrix, 0)
        mean_sims = sim_matrix.mean(axis=1)
        for i_local, i_global in enumerate(indices):
            centrality[i_global] = float(mean_sims[i_local])

    return centrality


def _name_cluster(label: int, diffs: list[DiffResult]) -> str:
    """Generate a named failure mode from dominant reason keywords."""
    reasons = " ".join(d.reason.lower() for d in diffs)

    if any(w in reasons for w in ["brief", "short", "concise", "terse", "succinct"]):
        return "BREVITY_GAIN" if _is_mostly_improvements(diffs) else "BREVITY_LOSS"
    if any(w in reasons for w in ["context", "nuance", "detail", "omit", "missing", "incomplete"]):
        return "CONTEXT_LOSS"
    if any(w in reasons for w in ["refus", "declin", "won't", "cannot", "avoid"]):
        return "REFUSAL_SHIFT"
    if any(w in reasons for w in ["tone", "empathy", "warm", "cold", "formal", "harsh"]):
        return "TONE_SHIFT"
    if any(w in reasons for w in ["accura", "wrong", "incorrect", "error", "fact", "hallucin"]):
        return "ACCURACY_CHANGE"
    if any(w in reasons for w in ["verbos", "long", "padded", "unnecessar", "redundant"]):
        return "VERBOSITY_GAIN"
    if any(w in reasons for w in ["format", "structur", "bullet", "list", "markdown"]):
        return "FORMAT_CHANGE"
    return f"CLUSTER_{label}"


def _describe_cluster(diffs: list[DiffResult]) -> str:
    return ". ".join(d.reason for d in diffs[:3])[:120]


def _is_mostly_improvements(diffs: list[DiffResult]) -> bool:
    return sum(1 for d in diffs if d.verdict == Verdict.IMPROVEMENT) > len(diffs) / 2