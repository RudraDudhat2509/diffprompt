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


def test_coerce_tag_handles_dicts_and_strings():
    assert Ontology._coerce_tag("formal") == "formal"
    # {tag: description} -> the key is the tag
    assert Ontology._coerce_tag({"formal": "a description"}) == "formal"
    # {"name": tag, "description": ...} -> the name field's value is the tag
    assert Ontology._coerce_tag({"name": "formal", "description": "x"}) == "formal"
    assert Ontology._coerce_tag(None) == ""


async def test_infer_coerces_nonstring_tags():
    """
    Regression for #10: the model sometimes returns tags as dicts. infer must
    coerce them to hashable strings instead of crashing build_anchors later.
    """
    raw = '{"tone": [{"formal": "x"}, "casual"], "intent": ["lookup"]}'
    o = Ontology()
    with patch("diffprompt.core.ontology.call_cascade", new=AsyncMock(return_value=(raw, "groq/x"))):
        await o.infer("a prompt")

    assert o.dimensions == {"tone": ["formal", "casual"], "intent": ["lookup"]}
    # every tag must be a hashable str so build_anchors can key on it
    for tags in o.dimensions.values():
        assert all(isinstance(t, str) for t in tags)


async def test_infer_falls_back_when_unusable():
    """Garbage / non-dict JSON falls back to default dimensions, never empty."""
    o = Ontology()
    with patch("diffprompt.core.ontology.call_cascade", new=AsyncMock(return_value=("[1, 2, 3]", "groq/x"))):
        await o.infer("a prompt")
    assert o.dimensions  # non-empty defaults


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
