"""
Ontology inference and anchor-based tagging.

Two steps:
1. Infer dimensions + tags from the prompt (one LLM call)
2. Generate anchor sentences per tag (one LLM call per tag)
3. Tag test inputs using embedding similarity to anchors (no LLM)
"""
from __future__ import annotations
import json
import re
from typing import Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from diffprompt.models.cascade import call_cascade


INFER_PROMPT = """You are designing a test suite for an LLM prompt.

Prompt: {prompt}

Identify 3-4 dimensions that would reveal BEHAVIORAL differences in how this prompt responds.
Focus on: user intent, emotional state, topic complexity, and request type.
Avoid surface-level dimensions like length, format, or punctuation style.

Return ONLY valid JSON. Example:
{{
  "tone": ["formal", "casual", "emotional", "urgent"],
  "complexity": ["simple", "multi-part", "ambiguous"],
  "intent": ["lookup", "reasoning", "emotional-support"]
}}

Each dimension must predict a meaningfully different response from this specific prompt."""


ANCHOR_PROMPT = """For the tag "{tag}" in dimension "{dimension}", write 3 short example sentences
that real users would send to this prompt: {prompt}

Each sentence must clearly represent "{tag}" and be distinct from each other.
Return ONLY a JSON array of 3 strings. No explanation."""



class Ontology:
    def __init__(self):
        self.dimensions: dict[str, list[str]] = {}
        self.anchors: dict[str, dict[str, str]] = {}      # dimension → tag → anchor sentence
        self.anchor_embeddings: dict[str, np.ndarray] = {} # dimension → matrix of embeddings
        self._embedder: Optional[SentenceTransformer] = None

    @property
    def embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
        return self._embedder

    async def infer(self, prompt: str, local_only: bool = False) -> None:
        """Infer dimensions and tags from the prompt. One LLM call."""
        raw, _ = await call_cascade(
            INFER_PROMPT.format(prompt=prompt),
            local_only=local_only
        )
        # Strip markdown fences if present
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        self.dimensions = json.loads(clean)

    async def build_anchors(self, prompt: str, local_only: bool = False) -> None:
        """Generate 3 anchor sentences per tag. One LLM call per tag."""
        for dimension, tags in self.dimensions.items():
            self.anchors[dimension] = {}
            for tag in tags:
                raw, _ = await call_cascade(
                    ANCHOR_PROMPT.format(tag=tag, dimension=dimension, prompt=prompt),
                    local_only=local_only
                )
                clean = re.sub(r"```(?:json)?|```", "", raw).strip()
                try:
                    sentences = json.loads(clean)
                    if isinstance(sentences, list):
                        self.anchors[dimension][tag] = sentences
                    else:
                        self.anchors[dimension][tag] = [sentences]
                except Exception:
                    self.anchors[dimension][tag] = [raw.strip()]
        for dimension, tag_anchors in self.anchors.items():
            self.anchor_embeddings[dimension] = {}
            for tag, sentences in tag_anchors.items():
                self.anchor_embeddings[dimension][tag] = self.embedder.encode(sentences)


    def tag(self, input_text: str) -> dict[str, str]:
        """
        Tag an input using max similarity across multiple anchors per tag.
        No LLM call — pure embeddings. Deterministic and fast.
        """
        input_emb = self.embedder.encode([input_text])
        result = {}

        for dimension, tags in self.dimensions.items():
            best_tag = None
            best_score = -1.0

            for tag in tags:
                anchor_embs = self.anchor_embeddings[dimension][tag]
                sims = cosine_similarity(input_emb, anchor_embs)[0]
                score = float(np.max(sims))  # best match across all anchors for this tag
                if score > best_score:
                    best_score = score
                    best_tag = tag

            result[dimension] = best_tag

        return result

    def to_dict(self) -> dict:
        return {"dimensions": self.dimensions, "anchors": self.anchors}

    @classmethod
    def from_dict(cls, data: dict) -> "Ontology":
        o = cls()
        o.dimensions = data["dimensions"]
        o.anchors = data["anchors"]
        # Rebuild anchor embeddings
        for dimension, tag_anchors in o.anchors.items():
            o.anchor_embeddings[dimension] = {}
            for tag, sentences in tag_anchors.items():
                o.anchor_embeddings[dimension][tag] = o.embedder.encode(sentences)
        return o
