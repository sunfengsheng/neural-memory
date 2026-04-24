"""Tests for neural_memory.core.synapse — Synapse data model."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from neural_memory.core.synapse import Synapse, SynapseType


class TestSynapseType:
    def test_values(self):
        assert SynapseType.SEMANTIC.value == "semantic"
        assert SynapseType.TEMPORAL.value == "temporal"
        assert SynapseType.CAUSAL.value == "causal"
        assert SynapseType.HIERARCHICAL.value == "hierarchical"
        assert SynapseType.REFERENCE.value == "reference"

    def test_from_string(self):
        assert SynapseType("temporal") is SynapseType.TEMPORAL

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            SynapseType("invalid")


class TestSynapse:
    def _make_synapse(self, **overrides):
        now = datetime.now(timezone.utc)
        defaults = dict(
            id=str(uuid.uuid4()),
            pre_neuron_id="pre-id",
            post_neuron_id="post-id",
            synapse_type=SynapseType.SEMANTIC,
            weight=0.5,
            activation_count=0,
            created_at=now,
            last_activated=now,
        )
        defaults.update(overrides)
        return Synapse(**defaults)

    def test_defaults(self):
        s = self._make_synapse()
        assert s.weight == 0.5
        assert s.activation_count == 0

    def test_hebbian_update_increases_weight(self):
        s = self._make_synapse(weight=0.5)
        s.hebbian_update(learning_rate=0.05)
        assert s.weight > 0.5

    def test_hebbian_update_increments_activation_count(self):
        s = self._make_synapse(weight=0.5, activation_count=3)
        s.hebbian_update()
        assert s.activation_count == 4

    def test_hebbian_update_updates_last_activated(self):
        old_time = datetime(2020, 1, 1, tzinfo=timezone.utc)
        s = self._make_synapse(weight=0.5, last_activated=old_time)
        s.hebbian_update()
        assert s.last_activated > old_time

    def test_hebbian_update_respects_max_weight(self):
        s = self._make_synapse(weight=0.99)
        for _ in range(100):
            s.hebbian_update(learning_rate=0.1, max_weight=1.0)
        assert s.weight <= 1.0

    def test_hebbian_update_asymptotic(self):
        """Weight approaches max_weight asymptotically."""
        s = self._make_synapse(weight=0.0)
        for _ in range(50):
            s.hebbian_update(learning_rate=0.1, max_weight=1.0)
        assert s.weight > 0.99

    def test_hebbian_update_formula(self):
        """Verify: w += lr * (max - w)."""
        s = self._make_synapse(weight=0.6)
        lr = 0.05
        max_w = 1.0
        expected = 0.6 + lr * (max_w - 0.6)
        s.hebbian_update(learning_rate=lr, max_weight=max_w)
        assert abs(s.weight - expected) < 1e-10

    def test_decay_weight(self):
        s = self._make_synapse(weight=0.5)
        s.decay_weight(factor=0.9)
        assert abs(s.weight - 0.45) < 1e-10

    def test_decay_weight_clamps_to_zero(self):
        s = self._make_synapse(weight=0.005)
        s.decay_weight(factor=0.5)
        assert s.weight == 0.0

    def test_decay_weight_default_factor(self):
        s = self._make_synapse(weight=1.0)
        s.decay_weight()
        assert abs(s.weight - 0.995) < 1e-10

    def test_repeated_decay(self):
        s = self._make_synapse(weight=1.0)
        for _ in range(1000):
            s.decay_weight(factor=0.99)
        assert s.weight == 0.0
