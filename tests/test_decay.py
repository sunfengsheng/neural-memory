"""Tests for neural_memory.core.decay — Ebbinghaus forgetting curve engine."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from neural_memory.config import DecayConfig
from neural_memory.core.decay import DecayEngine
from neural_memory.core.neuron import MemoryLayer, Neuron, NeuronFactory, NeuronType


def _make_neuron(**overrides) -> Neuron:
    defaults = dict(content="Test memory", neuron_type=NeuronType.SEMANTIC)
    defaults.update(overrides)
    return NeuronFactory.create(**defaults)


class TestComputeDecayedStrength:
    def test_no_time_elapsed(self):
        engine = DecayEngine()
        n = _make_neuron()
        now = n.last_decayed
        assert engine.compute_decayed_strength(n, now) == n.strength

    def test_strength_decreases_over_time(self):
        engine = DecayEngine()
        n = _make_neuron()
        future = n.last_decayed + timedelta(hours=24)
        decayed = engine.compute_decayed_strength(n, future)
        assert 0 < decayed < n.strength

    def test_more_time_more_decay(self):
        engine = DecayEngine()
        n = _make_neuron()
        t1 = n.last_decayed + timedelta(hours=1)
        t2 = n.last_decayed + timedelta(hours=10)
        s1 = engine.compute_decayed_strength(n, t1)
        s2 = engine.compute_decayed_strength(n, t2)
        assert s2 < s1

    def test_high_stability_slows_decay(self):
        engine = DecayEngine()
        n1 = _make_neuron()
        n1.stability = 1.0
        n2 = _make_neuron()
        n2.stability = 5.0
        n2.last_decayed = n1.last_decayed

        future = n1.last_decayed + timedelta(hours=24)
        s1 = engine.compute_decayed_strength(n1, future)
        s2 = engine.compute_decayed_strength(n2, future)
        assert s2 > s1

    def test_importance_shield(self):
        cfg = DecayConfig(importance_shield=0.3)
        engine = DecayEngine(cfg)
        n_low = _make_neuron(importance=0.2)
        n_high = _make_neuron(importance=0.5)
        n_high.last_decayed = n_low.last_decayed

        future = n_low.last_decayed + timedelta(hours=24)
        s_low = engine.compute_decayed_strength(n_low, future)
        s_high = engine.compute_decayed_strength(n_high, future)
        assert s_high > s_low

    def test_emotional_arousal_shield(self):
        cfg = DecayConfig(emotional_arousal_shield=0.5)
        engine = DecayEngine(cfg)
        n_calm = _make_neuron(emotional_arousal=0.1)
        n_emotional = _make_neuron(emotional_arousal=0.8)
        n_emotional.last_decayed = n_calm.last_decayed

        future = n_calm.last_decayed + timedelta(hours=24)
        s_calm = engine.compute_decayed_strength(n_calm, future)
        s_emo = engine.compute_decayed_strength(n_emotional, future)
        assert s_emo > s_calm

    def test_strength_never_negative(self):
        engine = DecayEngine()
        n = _make_neuron()
        far_future = n.last_decayed + timedelta(days=365)
        decayed = engine.compute_decayed_strength(n, far_future)
        assert decayed >= 0.0

    def test_uses_current_time_when_none(self):
        engine = DecayEngine()
        n = _make_neuron()
        n.last_decayed = datetime.now(timezone.utc) - timedelta(hours=1)
        s = engine.compute_decayed_strength(n, now=None)
        assert s < 1.0


class TestShouldPrune:
    def test_strong_neuron_not_pruned(self):
        engine = DecayEngine()
        n = _make_neuron()
        n.strength = 0.5
        assert engine.should_prune(n) is False

    def test_weak_unimportant_pruned(self):
        engine = DecayEngine()
        n = _make_neuron(importance=0.1, emotional_arousal=0.0)
        n.strength = 0.01
        assert engine.should_prune(n) is True

    def test_weak_but_important_not_pruned(self):
        engine = DecayEngine()
        n = _make_neuron(importance=0.5)
        n.strength = 0.01
        assert engine.should_prune(n) is False

    def test_weak_but_emotional_not_pruned(self):
        engine = DecayEngine()
        n = _make_neuron(importance=0.1, emotional_arousal=0.8)
        n.strength = 0.01
        assert engine.should_prune(n) is False

    def test_at_threshold_not_pruned(self):
        cfg = DecayConfig(prune_threshold=0.05)
        engine = DecayEngine(cfg)
        n = _make_neuron(importance=0.1, emotional_arousal=0.0)
        n.strength = 0.05
        assert engine.should_prune(n) is False


class TestReinforce:
    def test_restores_strength(self):
        engine = DecayEngine()
        n = _make_neuron()
        n.strength = 0.3
        engine.reinforce(n)
        assert n.strength == 1.0

    def test_increases_stability(self):
        cfg = DecayConfig(stability_growth_on_review=0.3)
        engine = DecayEngine(cfg)
        n = _make_neuron()
        n.stability = 1.0
        engine.reinforce(n)
        assert n.stability == 1.3

    def test_stability_capped(self):
        cfg = DecayConfig(max_stability=10.0, stability_growth_on_review=0.3)
        engine = DecayEngine(cfg)
        n = _make_neuron()
        n.stability = 9.9
        engine.reinforce(n)
        assert n.stability <= 10.0

    def test_increments_access_count(self):
        engine = DecayEngine()
        n = _make_neuron()
        assert n.access_count == 0
        engine.reinforce(n)
        assert n.access_count == 1

    def test_updates_timestamps(self):
        engine = DecayEngine()
        n = _make_neuron()
        old_accessed = n.last_accessed
        old_decayed = n.last_decayed
        engine.reinforce(n)
        assert n.last_accessed >= old_accessed
        assert n.last_decayed >= old_decayed


class TestBatchDecay:
    def test_returns_updates_and_prune_ids(self):
        engine = DecayEngine()
        n1 = _make_neuron(importance=0.1, emotional_arousal=0.0)
        n1.strength = 0.5
        n2 = _make_neuron(importance=0.1, emotional_arousal=0.0)
        n2.strength = 0.01
        n2.last_decayed = n1.last_decayed

        future = n1.last_decayed + timedelta(hours=24)
        updates, prune_ids = engine.batch_decay([n1, n2], future)

        # n1 should have an update, n2 should be pruned (very weak)
        assert len(updates) + len(prune_ids) == 2

    def test_empty_list(self):
        engine = DecayEngine()
        updates, prune_ids = engine.batch_decay([])
        assert updates == []
        assert prune_ids == []

    def test_update_format(self):
        engine = DecayEngine()
        n = _make_neuron(importance=0.5)
        n.strength = 0.8
        future = n.last_decayed + timedelta(hours=1)
        updates, _ = engine.batch_decay([n], future)
        assert len(updates) == 1
        new_strength, last_decayed_iso, neuron_id = updates[0]
        assert isinstance(new_strength, float)
        assert isinstance(last_decayed_iso, str)
        assert neuron_id == n.id
