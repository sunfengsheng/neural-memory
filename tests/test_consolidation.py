"""Tests for neural_memory.core.consolidation — Clustering & schema generation."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import numpy as np
import pytest

from neural_memory.config import ConsolidationConfig
from neural_memory.core.consolidation import ConsolidationEngine
from neural_memory.core.neuron import MemoryLayer, Neuron, NeuronFactory, NeuronType
from neural_memory.core.synapse import SynapseType


def _make_neuron(content: str = "Test", **overrides) -> Neuron:
    return NeuronFactory.create(content=content, **overrides)


def _random_unit_vec(dim: int = 384, seed: int | None = None) -> np.ndarray:
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _similar_vec(base: np.ndarray, noise: float = 0.05, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    noisy = base + rng.randn(*base.shape).astype(np.float32) * noise
    return noisy / np.linalg.norm(noisy)


class TestFindClusters:
    def test_too_few_neurons_returns_empty(self):
        cfg = ConsolidationConfig(min_cluster_size=3)
        engine = ConsolidationEngine(cfg)
        neurons = [_make_neuron(f"n{i}") for i in range(2)]
        embeddings = {n.id: _random_unit_vec(seed=i) for i, n in enumerate(neurons)}
        assert engine.find_clusters(neurons, embeddings) == []

    def test_no_embeddings_returns_empty(self):
        engine = ConsolidationEngine()
        neurons = [_make_neuron(f"n{i}") for i in range(5)]
        assert engine.find_clusters(neurons, {}) == []

    def test_similar_neurons_form_cluster(self):
        cfg = ConsolidationConfig(
            similarity_threshold=0.8,
            min_cluster_size=3,
            max_cluster_size=10,
        )
        engine = ConsolidationEngine(cfg)

        base = _random_unit_vec(seed=0)
        neurons = [_make_neuron(f"similar_{i}") for i in range(5)]
        embeddings = {}
        for i, n in enumerate(neurons):
            embeddings[n.id] = _similar_vec(base, noise=0.03, seed=i)

        clusters = engine.find_clusters(neurons, embeddings)
        assert len(clusters) >= 1
        assert len(clusters[0]) >= 3

    def test_dissimilar_neurons_no_cluster(self):
        cfg = ConsolidationConfig(
            similarity_threshold=0.95,
            min_cluster_size=3,
        )
        engine = ConsolidationEngine(cfg)

        neurons = [_make_neuron(f"different_{i}") for i in range(5)]
        embeddings = {n.id: _random_unit_vec(seed=i * 1000) for i, n in enumerate(neurons)}

        clusters = engine.find_clusters(neurons, embeddings)
        assert len(clusters) == 0

    def test_max_cluster_size_respected(self):
        cfg = ConsolidationConfig(
            similarity_threshold=0.5,
            min_cluster_size=2,
            max_cluster_size=3,
        )
        engine = ConsolidationEngine(cfg)

        base = _random_unit_vec(seed=0)
        neurons = [_make_neuron(f"n{i}") for i in range(10)]
        embeddings = {n.id: _similar_vec(base, noise=0.01, seed=i) for i, n in enumerate(neurons)}

        clusters = engine.find_clusters(neurons, embeddings)
        for c in clusters:
            assert len(c) <= 3

    def test_partial_embeddings(self):
        """Neurons without embeddings should be skipped, not crash."""
        cfg = ConsolidationConfig(min_cluster_size=2)
        engine = ConsolidationEngine(cfg)

        base = _random_unit_vec(seed=0)
        neurons = [_make_neuron(f"n{i}") for i in range(5)]
        # Only provide embeddings for 3 of 5
        embeddings = {
            neurons[0].id: _similar_vec(base, noise=0.01, seed=0),
            neurons[1].id: _similar_vec(base, noise=0.01, seed=1),
            neurons[2].id: _similar_vec(base, noise=0.01, seed=2),
        }
        clusters = engine.find_clusters(neurons, embeddings)
        # Should work without error; cluster may or may not form
        assert isinstance(clusters, list)


class TestCreateSchemaNeuron:
    def _make_cluster(self, size: int = 3):
        neurons = []
        embeddings = []
        base = _random_unit_vec(seed=42)
        for i in range(size):
            n = _make_neuron(
                content=f"Memory about topic {i}",
                importance=0.5 + i * 0.1,
                emotional_arousal=0.1 * i,
                tags=[f"tag{i}", "shared_tag"],
            )
            neurons.append(n)
            embeddings.append(_similar_vec(base, noise=0.02, seed=i))
        return neurons, embeddings

    def test_returns_neuron_embedding_synapses(self):
        engine = ConsolidationEngine()
        neurons, embeddings = self._make_cluster(3)
        schema, schema_emb, synapses = engine.create_schema_neuron(neurons, embeddings)

        assert isinstance(schema, Neuron)
        assert isinstance(schema_emb, np.ndarray)
        assert len(synapses) == 3

    def test_schema_neuron_properties(self):
        engine = ConsolidationEngine()
        neurons, embeddings = self._make_cluster(3)
        schema, _, _ = engine.create_schema_neuron(neurons, embeddings)

        assert schema.neuron_type == NeuronType.SCHEMA
        assert schema.layer == MemoryLayer.LONG_TERM
        assert schema.stability == 3.0
        assert schema.source == "consolidation"
        assert "[Schema:" in schema.content

    def test_schema_collects_all_tags(self):
        engine = ConsolidationEngine()
        neurons, embeddings = self._make_cluster(3)
        schema, _, _ = engine.create_schema_neuron(neurons, embeddings)

        assert "shared_tag" in schema.tags
        assert "tag0" in schema.tags
        assert "tag1" in schema.tags

    def test_schema_importance_boosted(self):
        engine = ConsolidationEngine()
        neurons, embeddings = self._make_cluster(3)
        avg_imp = np.mean([n.importance for n in neurons])
        schema, _, _ = engine.create_schema_neuron(neurons, embeddings)

        assert schema.importance >= avg_imp

    def test_schema_embedding_is_centroid(self):
        engine = ConsolidationEngine()
        neurons, embeddings = self._make_cluster(3)
        _, schema_emb, _ = engine.create_schema_neuron(neurons, embeddings)

        centroid = np.mean(np.stack(embeddings), axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        np.testing.assert_allclose(schema_emb, centroid, atol=1e-5)

    def test_synapses_are_hierarchical(self):
        engine = ConsolidationEngine()
        neurons, embeddings = self._make_cluster(3)
        schema, _, synapses = engine.create_schema_neuron(neurons, embeddings)

        for syn in synapses:
            assert syn.synapse_type == SynapseType.HIERARCHICAL
            assert syn.pre_neuron_id == schema.id
            assert syn.post_neuron_id in [n.id for n in neurons]
            assert syn.weight == 0.8

    def test_schema_content_contains_snippets(self):
        engine = ConsolidationEngine()
        neurons, embeddings = self._make_cluster(3)
        schema, _, _ = engine.create_schema_neuron(neurons, embeddings)

        for n in neurons:
            # Content or truncated version should appear
            assert n.content[:50] in schema.content or "..." in schema.content
