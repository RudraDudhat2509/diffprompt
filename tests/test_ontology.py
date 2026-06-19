"""
Tests for the ontology module.
Run with: pytest tests/test_ontology.py -v
"""
from unittest.mock import AsyncMock, patch

import numpy as np

from diffprompt.core.ontology import Ontology


class _FakeEmbedder:
    """Stand-in for SentenceTransformer so tests don't load the real model."""

    def encode(self, sentences):
        return np.zeros((len(sentences), 4))


def test_parse_anchors_json_array():
    assert Ontology._parse_anchors('["a", "b", "c"]') == ["a", "b", "c"]


def test_parse_anchors_fallback_on_garbage():
    out = Ontology._parse_anchors("not json at all")
    assert out == ["not json at all"]


async def test_build_anchors_covers_every_tag():
    """
    build_anchors fans out one call per tag concurrently and must populate
    anchors + embeddings for all of them.
    """
    o = Ontology()
    o._embedder = _FakeEmbedder()
    o.dimensions = {"tone": ["formal", "casual"], "intent": ["lookup"]}

    fake = AsyncMock(return_value=('["s1", "s2", "s3"]', "local/x"))
    with patch("diffprompt.core.ontology.call_cascade", new=fake):
        await o.build_anchors("some prompt")

    assert set(o.anchors.keys()) == {"tone", "intent"}
    assert o.anchors["tone"]["formal"] == ["s1", "s2", "s3"]
    assert "formal" in o.anchor_embeddings["tone"]
    assert fake.call_count == 3  # 2 tone tags + 1 intent tag
