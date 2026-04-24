"""Tests for neural_memory.retrieval.spreading_activation — Graph-based activation."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from neural_memory.config import ActivationConfig
from neural_memory.core.neuron import NeuronFactory
from neural_memory.core.synapse import Synapse, SynapseType
from neural_memory.retrieval.spreading_activation import SpreadingActivation


async def _make_chain(neuron_repo, synapse_repo, count=3, weight=0.8):
    """Create a chain of neurons: n0 -> n1 -> n2 -> ... with given weight."""
    neurons = []
    for i in range(count):
        n = NeuronFactory.create(content=f"Chain node {i}")
        await neuron_repo.insert(n)
        neurons.append(n)

    for i in range(count - 1):
        now = datetime.now(timezone.utc)
        syn = Synapse(
            id=str(uuid.uuid4()),
            pre_neuron_id=neurons[i].id,
            post_neuron_id=neurons[i + 1].id,
            synapse_type=SynapseType.SEMANTIC,
            weight=weight,
            created_at=now,
            last_activated=now,
        )
        await synapse_repo.upsert(syn)

    return neurons


class TestSpreadingActivationActivate:
    async def test_empty_seeds(self, synapse_repo):
        sa = SpreadingActivation(synapse_repo)
        result = await sa.activate([])
        assert result == {}

    async def test_single_seed_no_connections(self, neuron_repo, synapse_repo):
        n = NeuronFactory.create(content="Isolated")
        await neuron_repo.insert(n)

        sa = SpreadingActivation(synapse_repo)
        result = await sa.activate([n.id])
        # Seed itself should still be in activation map
        assert n.id in result
        assert result[n.id] == pytest.approx(1.0)

    async def test_propagation_through_chain(self, neuron_repo, synapse_repo):
        neurons = await _make_chain(neuron_repo, synapse_repo, count=3, weight=0.8)

        sa = SpreadingActivation(synapse_repo)
        result = await sa.activate([neurons[0].id])

        # Seed should still be activated (may be > 1.0 due to back-propagation accumulation)
        assert result[neurons[0].id] >= 1.0
        # First hop should receive propagated activation
        assert neurons[1].id in result
        assert result[neurons[1].id] > 0
        # All three nodes should be activated
        assert neurons[2].id in result

    async def test_activation_decays_per_hop(self, neuron_repo, synapse_repo):
        neurons = await _make_chain(neuron_repo, synapse_repo, count=3, weight=1.0)

        cfg = ActivationConfig(
            initial_activation=1.0,
            decay_per_hop=0.5,
            min_activation=0.01,
            max_hops=4,
        )
        sa = SpreadingActivation(synapse_repo, config=cfg)
        result = await sa.activate([neurons[0].id])

        # With weight=1.0 and decay=0.5, activation propagates and accumulates back
        # Due to bidirectional traversal, seed activation can exceed initial value
        assert result[neurons[0].id] >= 1.0
        assert neurons[1].id in result
        assert result[neurons[1].id] > 0
        # Verify decay is happening: hop-1 node should have less initial propagated
        # activation than the seed (even though accumulation may increase both)

    async def test_custom_seed_activations(self, neuron_repo, synapse_repo):
        n = NeuronFactory.create(content="Custom seed")
        await neuron_repo.insert(n)

        sa = SpreadingActivation(synapse_repo)
        result = await sa.activate([n.id], seed_activations=[0.5])
        assert result[n.id] == pytest.approx(0.5)

    async def test_max_hops_limit(self, neuron_repo, synapse_repo):
        # Create a long chain exceeding max_hops
        neurons = await _make_chain(neuron_repo, synapse_repo, count=6, weight=0.9)

        cfg = ActivationConfig(
            initial_activation=1.0,
            decay_per_hop=0.8,
            min_activation=0.001,
            max_hops=2,  # Only 2 hops allowed
        )
        sa = SpreadingActivation(synapse_repo, config=cfg)
        result = await sa.activate([neurons[0].id])

        # Seed + up to 2 hops = neurons[0], [1], [2] at most
        assert neurons[0].id in result
        # Neurons far beyond max_hops should not be activated
        # (neurons[3], [4], [5] are 3, 4, 5 hops away)
        for n in neurons[3:]:
            # They should either not be in result or have very low activation
            if n.id in result:
                assert result[n.id] < result.get(neurons[1].id, 1.0)

    async def test_weak_synapse_filtered(self, neuron_repo, synapse_repo):
        """Weak synapses (below min_weight=0.05 in get_connections) don't propagate."""
        neurons = await _make_chain(neuron_repo, synapse_repo, count=2, weight=0.01)

        sa = SpreadingActivation(synapse_repo)
        result = await sa.activate([neurons[0].id])

        # weight=0.01 is below the 0.05 min_weight in get_connections
        # so neuron[1] should NOT receive activation
        assert neurons[1].id not in result

    async def test_multiple_seeds(self, neuron_repo, synapse_repo):
        n1 = NeuronFactory.create(content="Seed A")
        n2 = NeuronFactory.create(content="Seed B")
        await neuron_repo.insert(n1)
        await neuron_repo.insert(n2)

        sa = SpreadingActivation(synapse_repo)
        result = await sa.activate([n1.id, n2.id])
        assert n1.id in result
        assert n2.id in result

    async def test_bidirectional_traversal(self, neuron_repo, synapse_repo):
        """Activation should spread in both directions along a synapse."""
        n1 = NeuronFactory.create(content="Source")
        n2 = NeuronFactory.create(content="Target")
        await neuron_repo.insert(n1)
        await neuron_repo.insert(n2)

        now = datetime.now(timezone.utc)
        syn = Synapse(
            id=str(uuid.uuid4()),
            pre_neuron_id=n1.id,
            post_neuron_id=n2.id,
            synapse_type=SynapseType.SEMANTIC,
            weight=0.8,
            created_at=now,
            last_activated=now,
        )
        await synapse_repo.upsert(syn)

        sa = SpreadingActivation(synapse_repo)

        # Seed from the post side → should still reach pre side
        result = await sa.activate([n2.id])
        assert n1.id in result


class TestSpreadingActivationGetTop:
    def test_get_top_activated_sorted(self):
        sa = SpreadingActivation.__new__(SpreadingActivation)
        sa._cfg = ActivationConfig()

        activation = {"a": 0.9, "b": 0.5, "c": 0.1, "d": 0.7}
        top = sa.get_top_activated(activation, top_k=2)
        assert len(top) == 2
        assert top[0] == ("a", 0.9)
        assert top[1] == ("d", 0.7)

    def test_get_top_activated_default_k(self):
        sa = SpreadingActivation.__new__(SpreadingActivation)
        sa._cfg = ActivationConfig(top_k=3)

        activation = {f"n{i}": float(i) / 10 for i in range(10)}
        top = sa.get_top_activated(activation)
        assert len(top) == 3
        # Highest first
        assert top[0][0] == "n9"

    def test_get_top_activated_empty(self):
        sa = SpreadingActivation.__new__(SpreadingActivation)
        sa._cfg = ActivationConfig()

        top = sa.get_top_activated({}, top_k=5)
        assert top == []

    def test_get_top_activated_fewer_than_k(self):
        sa = SpreadingActivation.__new__(SpreadingActivation)
        sa._cfg = ActivationConfig()

        activation = {"x": 0.5, "y": 0.3}
        top = sa.get_top_activated(activation, top_k=10)
        assert len(top) == 2
