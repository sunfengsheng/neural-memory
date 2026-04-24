"""Tests for neural_memory.core.neuron — Neuron data model and factory."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

import pytest

from neural_memory.core.neuron import MemoryLayer, Neuron, NeuronFactory, NeuronType


# ── NeuronType enum ──────────────────────────────────────────────

class TestNeuronType:
    def test_values(self):
        assert NeuronType.EPISODIC.value == "episodic"
        assert NeuronType.SEMANTIC.value == "semantic"
        assert NeuronType.PROCEDURAL.value == "procedural"
        assert NeuronType.SCHEMA.value == "schema"

    def test_from_string(self):
        assert NeuronType("semantic") is NeuronType.SEMANTIC

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            NeuronType("invalid_type")


# ── MemoryLayer enum ─────────────────────────────────────────────

class TestMemoryLayer:
    def test_values(self):
        assert MemoryLayer.WORKING.value == "working"
        assert MemoryLayer.SHORT_TERM.value == "short_term"
        assert MemoryLayer.LONG_TERM.value == "long_term"

    def test_from_string(self):
        assert MemoryLayer("long_term") is MemoryLayer.LONG_TERM


# ── Neuron dataclass ─────────────────────────────────────────────

class TestNeuron:
    def _make_neuron(self, **overrides):
        defaults = dict(
            id=str(uuid.uuid4()),
            content="Test memory content",
            neuron_type=NeuronType.SEMANTIC,
            layer=MemoryLayer.WORKING,
            strength=1.0,
            stability=1.0,
            importance=0.5,
        )
        defaults.update(overrides)
        return Neuron(**defaults)

    def test_defaults(self):
        n = self._make_neuron()
        assert n.emotional_valence == 0.0
        assert n.emotional_arousal == 0.0
        assert n.access_count == 0
        assert n.embedding is None
        assert n.tags == []
        assert n.source == "manual"

    def test_tags_json(self):
        n = self._make_neuron(tags=["python", "testing"])
        parsed = json.loads(n.tags_json)
        assert parsed == ["python", "testing"]

    def test_tags_json_empty(self):
        n = self._make_neuron(tags=[])
        assert n.tags_json == "[]"

    def test_is_emotional_false(self):
        n = self._make_neuron(emotional_valence=0.0, emotional_arousal=0.0)
        assert n.is_emotional is False

    def test_is_emotional_by_valence(self):
        n = self._make_neuron(emotional_valence=0.5, emotional_arousal=0.0)
        assert n.is_emotional is True

    def test_is_emotional_by_arousal(self):
        n = self._make_neuron(emotional_valence=0.0, emotional_arousal=0.5)
        assert n.is_emotional is True

    def test_is_emotional_negative_valence(self):
        n = self._make_neuron(emotional_valence=-0.5, emotional_arousal=0.0)
        assert n.is_emotional is True

    def test_is_emotional_borderline(self):
        """Exactly at threshold = not emotional."""
        n = self._make_neuron(emotional_valence=0.3, emotional_arousal=0.3)
        assert n.is_emotional is False

    def test_to_dict(self):
        n = self._make_neuron(
            tags=["tag1"],
            importance=0.7,
            source="test",
        )
        d = n.to_dict()
        assert d["id"] == n.id
        assert d["content"] == "Test memory content"
        assert d["neuron_type"] == "semantic"
        assert d["layer"] == "working"
        assert d["importance"] == 0.7
        assert d["tags"] == ["tag1"]
        assert d["source"] == "test"
        assert "created_at" in d
        assert "last_accessed" in d

    def test_to_dict_strength_rounding(self):
        n = self._make_neuron(strength=0.123456789)
        d = n.to_dict()
        assert d["strength"] == 0.1235

    def test_to_dict_stability_rounding(self):
        n = self._make_neuron(stability=2.999999)
        d = n.to_dict()
        assert d["stability"] == 3.0


# ── NeuronFactory ────────────────────────────────────────────────

class TestNeuronFactory:
    def test_create_defaults(self):
        n = NeuronFactory.create(content="Hello world")
        assert n.content == "Hello world"
        assert n.neuron_type == NeuronType.SEMANTIC
        assert n.layer == MemoryLayer.WORKING
        assert n.strength == 1.0
        assert n.stability == 1.0
        assert n.importance == 0.5
        assert n.tags == []
        assert n.source == "manual"

    def test_create_unique_id(self):
        n1 = NeuronFactory.create(content="A")
        n2 = NeuronFactory.create(content="A")
        assert n1.id != n2.id

    def test_create_valid_uuid(self):
        n = NeuronFactory.create(content="Test")
        uuid.UUID(n.id)  # Should not raise

    def test_create_timestamps_are_utc(self):
        n = NeuronFactory.create(content="Test")
        assert n.created_at.tzinfo is not None
        assert n.last_accessed.tzinfo is not None
        assert n.last_decayed.tzinfo is not None

    def test_create_importance_clamp_high(self):
        n = NeuronFactory.create(content="Test", importance=5.0)
        assert n.importance == 1.0

    def test_create_importance_clamp_low(self):
        n = NeuronFactory.create(content="Test", importance=-1.0)
        assert n.importance == 0.0

    def test_create_emotional_valence_clamp(self):
        n = NeuronFactory.create(content="Test", emotional_valence=2.0)
        assert n.emotional_valence == 1.0
        n2 = NeuronFactory.create(content="Test", emotional_valence=-3.0)
        assert n2.emotional_valence == -1.0

    def test_create_emotional_arousal_clamp(self):
        n = NeuronFactory.create(content="Test", emotional_arousal=5.0)
        assert n.emotional_arousal == 1.0
        n2 = NeuronFactory.create(content="Test", emotional_arousal=-1.0)
        assert n2.emotional_arousal == 0.0

    def test_create_auto_summary_long_content(self):
        long_text = "A" * 300
        n = NeuronFactory.create(content=long_text)
        assert n.summary is not None
        assert n.summary.endswith("...")
        assert len(n.summary) == 200

    def test_create_no_auto_summary_short_content(self):
        n = NeuronFactory.create(content="Short text")
        assert n.summary is None

    def test_create_explicit_summary(self):
        n = NeuronFactory.create(content="A" * 300, summary="Custom summary")
        assert n.summary == "Custom summary"

    def test_create_with_tags(self):
        n = NeuronFactory.create(content="Test", tags=["a", "b"])
        assert n.tags == ["a", "b"]

    def test_create_with_file_metadata(self):
        n = NeuronFactory.create(
            content="code snippet",
            file_path="code/python/abc123.py",
            file_type="code/python",
            file_hash="sha256hash",
        )
        assert n.file_path == "code/python/abc123.py"
        assert n.file_type == "code/python"
        assert n.file_hash == "sha256hash"

    def test_create_all_neuron_types(self):
        for nt in NeuronType:
            n = NeuronFactory.create(content="Test", neuron_type=nt)
            assert n.neuron_type == nt
