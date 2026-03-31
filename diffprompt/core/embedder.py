"""
Embedding + similarity layer.
All local, all free. No API calls.
"""
from __future__ import annotations
import logging
import os
import warnings
import numpy as np
from functools import lru_cache

# Suppress noisy warnings from HuggingFace / sentence-transformers / UMAP
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*n_jobs value.*overridden.*")
warnings.filterwarnings("ignore", message=".*unauthenticated.*")
warnings.filterwarnings("ignore", message=".*UNEXPECTED.*")
warnings.filterwarnings("ignore", category=UserWarning, module="umap")

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine


MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    """Lazy-load embedder. Cached so it only loads once per session."""
    # Silence the load report printed to stdout by newer sentence-transformers
    import io, contextlib
    f = io.StringIO()
    with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        model = SentenceTransformer(MODEL_NAME)
    return model


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
        score = sk_cosine([all_embs[i]], [all_embs[i + 1]])[0][0]
        scores.append(float(np.clip(score, 0, 1)))
    return scores