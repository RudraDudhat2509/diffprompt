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


INFER_PROMPT = """You are analyzing a prompt to determine what input dimensions matter for testing it.

Prompt: {prompt}

Identify 3-5 dimensions along which inputs to this prompt could meaningfully vary.
For each dimension, provide 3-5 discrete values.

Return ONLY valid JSON. Example:
{{
  "tone": ["formal", "casual", "emotional", "urgent"],
  "complexity": ["simple", "multi-part", "open-ended"],
  "intent": ["lookup", "reasoning", "generation"]
}}

Base dimensions entirely on what THIS prompt handles."""


ANCHOR_PROMPT = """For the tag "{tag}" in dimension "{dimension}", write one short example sentence
that perfectly represents this tag in the context of this prompt: {prompt}

Return just the sentence, nothing else. No quotes, no explanation."""


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
        """Generate anchor sentences for each tag. One LLM call per tag."""
        for dimension, tags in self.dimensions.items():
            self.anchors[dimension] = {}
            for tag in tags:
                anchor, _ = await call_cascade(
                    ANCHOR_PROMPT.format(tag=tag, dimension=dimension, prompt=prompt),
                    local_only=local_only
                )
                self.anchors[dimension][tag] = anchor.strip()

        # Pre-compute anchor embeddings per dimension
        for dimension, tag_anchors in self.anchors.items():
            sentences = list(tag_anchors.values())
            self.anchor_embeddings[dimension] = self.embedder.encode(sentences)

    def tag(self, input_text: str) -> dict[str, str]:
        """
        Tag an input using embedding similarity to anchors.
        No LLM call — pure embeddings. Deterministic and fast.
        """
        input_emb = self.embedder.encode([input_text])
        result = {}

        for dimension, tags in self.dimensions.items():
            anchor_embs = self.anchor_embeddings[dimension]
            sims = cosine_similarity(input_emb, anchor_embs)[0]
            best_idx = int(np.argmax(sims))
            result[dimension] = tags[best_idx]

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
            sentences = list(tag_anchors.values())
            o.anchor_embeddings[dimension] = o.embedder.encode(sentences)
        return o
