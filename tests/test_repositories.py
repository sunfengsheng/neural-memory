"""Tests for neural_memory.db.repositories — NeuronRepository & SynapseRepository."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from neural_memory.core.neuron import MemoryLayer, Neuron, NeuronFactory, NeuronType
from neural_memory.core.synapse import Synapse, SynapseType
from neural_memory.db.repositories import NeuronRepository, SynapseRepository


# ── NeuronRepository ─────────────────────────────────────────────

class TestNeuronRepository:
    async def test_insert_and_get_by_id(self, neuron_repo, sample_neuron):
        await neuron_repo.insert(sample_neuron)
        fetched = await neuron_repo.get_by_id(sample_neuron.id)
        assert fetched is not None
        assert fetched.id == sample_neuron.id
        assert fetched.content == sample_neuron.content
        assert fetched.neuron_type == sample_neuron.neuron_type
        assert fetched.layer == sample_neuron.layer

    async def test_get_by_id_nonexistent(self, neuron_repo):
        result = await neuron_repo.get_by_id("nonexistent-id")
        assert result is None

    async def test_insert_preserves_tags(self, neuron_repo, sample_neuron):
        sample_neuron.tags = ["python", "testing", "async"]
        await neuron_repo.insert(sample_neuron)
        fetched = await neuron_repo.get_by_id(sample_neuron.id)
        assert fetched.tags == ["python", "testing", "async"]

    async def test_insert_preserves_embedding(self, neuron_repo, sample_neuron):
        vec = np.random.randn(384).astype(np.float32)
        sample_neuron.embedding = vec.tobytes()
        sample_neuron.embedding_model = "test-model"
        await neuron_repo.insert(sample_neuron)

        fetched = await neuron_repo.get_by_id(sample_neuron.id)
        assert fetched.embedding_model == "test-model"
        restored = np.frombuffer(fetched.embedding, dtype=np.float32)
        np.testing.assert_allclose(restored, vec)

    async def test_update_strength(self, neuron_repo, sample_neuron):
        await neuron_repo.insert(sample_neuron)
        await neuron_repo.update_strength(sample_neuron.id, 0.42)
        fetched = await neuron_repo.get_by_id(sample_neuron.id)
        assert abs(fetched.strength - 0.42) < 1e-6

    async def test_update_strength_with_stability(self, neuron_repo, sample_neuron):
        await neuron_repo.insert(sample_neuron)
        await neuron_repo.update_strength(sample_neuron.id, 0.9, stability=2.5)
        fetched = await neuron_repo.get_by_id(sample_neuron.id)
        assert abs(fetched.strength - 0.9) < 1e-6
        assert abs(fetched.stability - 2.5) < 1e-6
        assert fetched.access_count == 1

    async def test_update_layer(self, neuron_repo, sample_neuron):
        await neuron_repo.insert(sample_neuron)
        await neuron_repo.update_layer(sample_neuron.id, MemoryLayer.LONG_TERM)
        fetched = await neuron_repo.get_by_id(sample_neuron.id)
        assert fetched.layer == MemoryLayer.LONG_TERM

    async def test_update_decay_batch(self, neuron_repo):
        neurons = []
        for i in range(3):
            n = NeuronFactory.create(content=f"Neuron {i}")
            await neuron_repo.insert(n)
            neurons.append(n)

        now_iso = datetime.now(timezone.utc).isoformat()
        updates = [(0.5 + i * 0.1, now_iso, n.id) for i, n in enumerate(neurons)]
        await neuron_repo.update_decay_batch(updates)

        for i, n in enumerate(neurons):
            fetched = await neuron_repo.get_by_id(n.id)
            assert abs(fetched.strength - (0.5 + i * 0.1)) < 1e-6

    async def test_get_all_for_decay(self, neuron_repo):
        n1 = NeuronFactory.create(content="Active")
        n1.strength = 0.5
        n2 = NeuronFactory.create(content="Dead")
        n2.strength = 0.0
        await neuron_repo.insert(n1)
        await neuron_repo.insert(n2)

        result = await neuron_repo.get_all_for_decay()
        ids = [n.id for n in result]
        assert n1.id in ids
        assert n2.id not in ids

    async def test_get_by_layer(self, neuron_repo):
        n1 = NeuronFactory.create(content="Working memory item")
        n2 = NeuronFactory.create(content="Long term item")
        n2.layer = MemoryLayer.LONG_TERM
        await neuron_repo.insert(n1)
        await neuron_repo.insert(n2)

        working = await neuron_repo.get_by_layer(MemoryLayer.WORKING)
        assert any(n.id == n1.id for n in working)
        assert not any(n.id == n2.id for n in working)

    async def test_get_temporal_neighbors(self, neuron_repo):
        now = datetime.now(timezone.utc)
        n1 = NeuronFactory.create(content="Recent 1")
        n1.created_at = now
        n2 = NeuronFactory.create(content="Recent 2")
        n2.created_at = now - timedelta(minutes=10)
        n3 = NeuronFactory.create(content="Old")
        n3.created_at = now - timedelta(hours=2)
        for n in [n1, n2, n3]:
            await neuron_repo.insert(n)

        neighbors = await neuron_repo.get_temporal_neighbors(now, window_seconds=1800)
        ids = [n.id for n in neighbors]
        assert n1.id in ids
        assert n2.id in ids
        assert n3.id not in ids

    async def test_get_embedding_batch(self, neuron_repo):
        n1 = NeuronFactory.create(content="With embedding")
        n1.embedding = np.zeros(384, dtype=np.float32).tobytes()
        n2 = NeuronFactory.create(content="Without embedding")
        await neuron_repo.insert(n1)
        await neuron_repo.insert(n2)

        pairs = await neuron_repo.get_embedding_batch()
        ids = [p[0] for p in pairs]
        assert n1.id in ids
        assert n2.id not in ids

    async def test_fts_search(self, neuron_repo):
        n1 = NeuronFactory.create(content="Python is a great programming language")
        n2 = NeuronFactory.create(content="The weather is nice today")
        await neuron_repo.insert(n1)
        await neuron_repo.insert(n2)

        results = await neuron_repo.fts_search("Python programming")
        ids = [n.id for n in results]
        assert n1.id in ids

    async def test_delete(self, neuron_repo, sample_neuron):
        await neuron_repo.insert(sample_neuron)
        deleted = await neuron_repo.delete(sample_neuron.id)
        assert deleted is True
        assert await neuron_repo.get_by_id(sample_neuron.id) is None

    async def test_delete_nonexistent(self, neuron_repo):
        deleted = await neuron_repo.delete("nonexistent")
        assert deleted is False

    async def test_count_by_layer(self, neuron_repo):
        n1 = NeuronFactory.create(content="W1")
        n2 = NeuronFactory.create(content="W2")
        n3 = NeuronFactory.create(content="LT")
        n3.layer = MemoryLayer.LONG_TERM
        for n in [n1, n2, n3]:
            await neuron_repo.insert(n)

        counts = await neuron_repo.count_by_layer()
        assert counts.get("working", 0) == 2
        assert counts.get("long_term", 0) == 1

    async def test_get_by_type_and_layer(self, neuron_repo):
        n1 = NeuronFactory.create(content="Semantic working", neuron_type=NeuronType.SEMANTIC)
        n2 = NeuronFactory.create(content="Episodic working", neuron_type=NeuronType.EPISODIC)
        await neuron_repo.insert(n1)
        await neuron_repo.insert(n2)

        results = await neuron_repo.get_by_type_and_layer(
            NeuronType.SEMANTIC, MemoryLayer.WORKING
        )
        ids = [n.id for n in results]
        assert n1.id in ids
        assert n2.id not in ids

    async def test_get_pruning_candidates(self, neuron_repo):
        weak = NeuronFactory.create(content="Weak", importance=0.1, emotional_arousal=0.0)
        weak.strength = 0.01
        strong = NeuronFactory.create(content="Strong", importance=0.1)
        strong.strength = 0.5
        important = NeuronFactory.create(content="Important", importance=0.5)
        important.strength = 0.01

        for n in [weak, strong, important]:
            await neuron_repo.insert(n)

        candidates = await neuron_repo.get_pruning_candidates()
        ids = [n.id for n in candidates]
        assert weak.id in ids
        assert strong.id not in ids
        assert important.id not in ids


# ── SynapseRepository ────────────────────────────────────────────

class TestSynapseRepository:
    async def _insert_two_neurons(self, neuron_repo):
        n1 = NeuronFactory.create(content="Neuron A")
        n2 = NeuronFactory.create(content="Neuron B")
        await neuron_repo.insert(n1)
        await neuron_repo.insert(n2)
        return n1, n2

    async def test_upsert_and_get_connections(self, neuron_repo, synapse_repo):
        n1, n2 = await self._insert_two_neurons(neuron_repo)
        now = datetime.now(timezone.utc)
        syn = Synapse(
            id=str(uuid.uuid4()),
            pre_neuron_id=n1.id,
            post_neuron_id=n2.id,
            synapse_type=SynapseType.SEMANTIC,
            weight=0.7,
            created_at=now,
            last_activated=now,
        )
        await synapse_repo.upsert(syn)

        conns = await synapse_repo.get_connections(n1.id)
        assert len(conns) == 1
        assert conns[0].id == syn.id
        assert abs(conns[0].weight - 0.7) < 1e-6

    async def test_upsert_updates_existing(self, neuron_repo, synapse_repo):
        n1, n2 = await self._insert_two_neurons(neuron_repo)
        now = datetime.now(timezone.utc)
        syn_id = str(uuid.uuid4())

        syn1 = Synapse(
            id=syn_id, pre_neuron_id=n1.id, post_neuron_id=n2.id,
            synapse_type=SynapseType.SEMANTIC, weight=0.5,
            created_at=now, last_activated=now,
        )
        await synapse_repo.upsert(syn1)

        syn2 = Synapse(
            id=syn_id, pre_neuron_id=n1.id, post_neuron_id=n2.id,
            synapse_type=SynapseType.SEMANTIC, weight=0.9,
            created_at=now, last_activated=now,
        )
        await synapse_repo.upsert(syn2)

        conns = await synapse_repo.get_connections(n1.id)
        assert len(conns) == 1
        assert abs(conns[0].weight - 0.9) < 1e-6

    async def test_get_connections_both_directions(self, neuron_repo, synapse_repo):
        n1, n2 = await self._insert_two_neurons(neuron_repo)
        now = datetime.now(timezone.utc)
        syn = Synapse(
            id=str(uuid.uuid4()), pre_neuron_id=n1.id, post_neuron_id=n2.id,
            synapse_type=SynapseType.TEMPORAL, weight=0.6,
            created_at=now, last_activated=now,
        )
        await synapse_repo.upsert(syn)

        # Should appear when querying from either direction
        from_n1 = await synapse_repo.get_connections(n1.id)
        from_n2 = await synapse_repo.get_connections(n2.id)
        assert len(from_n1) == 1
        assert len(from_n2) == 1

    async def test_get_connections_min_weight_filter(self, neuron_repo, synapse_repo):
        n1, n2 = await self._insert_two_neurons(neuron_repo)
        now = datetime.now(timezone.utc)
        syn = Synapse(
            id=str(uuid.uuid4()), pre_neuron_id=n1.id, post_neuron_id=n2.id,
            synapse_type=SynapseType.SEMANTIC, weight=0.3,
            created_at=now, last_activated=now,
        )
        await synapse_repo.upsert(syn)

        high_min = await synapse_repo.get_connections(n1.id, min_weight=0.5)
        assert len(high_min) == 0

        low_min = await synapse_repo.get_connections(n1.id, min_weight=0.2)
        assert len(low_min) == 1

    async def test_get_outgoing(self, neuron_repo, synapse_repo):
        n1, n2 = await self._insert_two_neurons(neuron_repo)
        now = datetime.now(timezone.utc)
        syn = Synapse(
            id=str(uuid.uuid4()), pre_neuron_id=n1.id, post_neuron_id=n2.id,
            synapse_type=SynapseType.SEMANTIC, weight=0.5,
            created_at=now, last_activated=now,
        )
        await synapse_repo.upsert(syn)

        outgoing = await synapse_repo.get_outgoing(n1.id)
        assert len(outgoing) == 1

        outgoing_n2 = await synapse_repo.get_outgoing(n2.id)
        assert len(outgoing_n2) == 0

    async def test_get_all_connections_for_neuron(self, neuron_repo, synapse_repo):
        n1 = NeuronFactory.create(content="Center")
        n2 = NeuronFactory.create(content="Left")
        n3 = NeuronFactory.create(content="Right")
        for n in [n1, n2, n3]:
            await neuron_repo.insert(n)

        now = datetime.now(timezone.utc)
        # n1 -> n2, n3 -> n1
        s1 = Synapse(
            id=str(uuid.uuid4()), pre_neuron_id=n1.id, post_neuron_id=n2.id,
            synapse_type=SynapseType.SEMANTIC, weight=0.5,
            created_at=now, last_activated=now,
        )
        s2 = Synapse(
            id=str(uuid.uuid4()), pre_neuron_id=n3.id, post_neuron_id=n1.id,
            synapse_type=SynapseType.TEMPORAL, weight=0.4,
            created_at=now, last_activated=now,
        )
        await synapse_repo.upsert(s1)
        await synapse_repo.upsert(s2)

        all_conns = await synapse_repo.get_all_connections_for_neuron(n1.id)
        assert len(all_conns) == 2

    async def test_delete_for_neuron(self, neuron_repo, synapse_repo):
        n1, n2 = await self._insert_two_neurons(neuron_repo)
        now = datetime.now(timezone.utc)
        syn = Synapse(
            id=str(uuid.uuid4()), pre_neuron_id=n1.id, post_neuron_id=n2.id,
            synapse_type=SynapseType.SEMANTIC, weight=0.5,
            created_at=now, last_activated=now,
        )
        await synapse_repo.upsert(syn)

        count = await synapse_repo.delete_for_neuron(n1.id)
        assert count == 1

        conns = await synapse_repo.get_connections(n1.id)
        assert len(conns) == 0

    async def test_decay_all_weights(self, neuron_repo, synapse_repo):
        n1, n2 = await self._insert_two_neurons(neuron_repo)
        now = datetime.now(timezone.utc)
        syn = Synapse(
            id=str(uuid.uuid4()), pre_neuron_id=n1.id, post_neuron_id=n2.id,
            synapse_type=SynapseType.SEMANTIC, weight=0.5,
            created_at=now, last_activated=now,
        )
        await synapse_repo.upsert(syn)

        await synapse_repo.decay_all_weights(factor=0.9)

        conns = await synapse_repo.get_connections(n1.id)
        assert len(conns) == 1
        assert abs(conns[0].weight - 0.45) < 1e-6

    async def test_decay_cleans_dead_synapses(self, neuron_repo, synapse_repo):
        n1, n2 = await self._insert_two_neurons(neuron_repo)
        now = datetime.now(timezone.utc)
        syn = Synapse(
            id=str(uuid.uuid4()), pre_neuron_id=n1.id, post_neuron_id=n2.id,
            synapse_type=SynapseType.SEMANTIC, weight=0.005,
            created_at=now, last_activated=now,
        )
        await synapse_repo.upsert(syn)

        await synapse_repo.decay_all_weights(factor=0.5)

        conns = await synapse_repo.get_connections(n1.id, min_weight=0.0)
        assert len(conns) == 0
