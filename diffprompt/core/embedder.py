"""
Embedding + similarity layer.
All local, all free. No API calls.
"""
from __future__ import annotations
import numpy as np
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine


MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    """Lazy-load embedder. Cached so it only loads once per session."""
    return SentenceTransformer(MODEL_NAME)


def embed(texts: list[str]) -> np.ndarray:
    return get_embedder().encode(texts, show_progress_bar=False)


def similarity(text_a: str, text_b: str) -> float:
    """Cosine similarity between two texts. Returns float 0-1."""
    embs = embed([text_a, text_b])
    score = sk_cosine([embs[0]], [embs[1]])[0][0]
    return float(np.clip(score, 0, 1))


def batch_similarity(pairs: list[tuple[str, str]]) -> list[float]:
    """
    Efficient batch similarity for many pairs.
    Embeds all texts in one pass instead of N passes.
    """
    all_texts = [t for pair in pairs for t in pair]
    all_embs = embed(all_texts)

    scores = []
    for i in range(0, len(all_embs), 2):
        score = sk_cosine([all_embs[i]], [all_embs[i+1]])[0][0]
        scores.append(float(np.clip(score, 0, 1)))
    return scores
