"""Tests for neural_memory.core.working_memory — Limited-capacity active buffer."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from neural_memory.config import WorkingMemoryConfig
from neural_memory.core.neuron import MemoryLayer, NeuronFactory
from neural_memory.core.working_memory import WorkingMemoryManager


class TestWorkingMemoryAdd:
    async def test_add_neuron(self, db_conn, neuron_repo):
        wm = WorkingMemoryManager(db_conn)
        n = NeuronFactory.create(content="Test")
        await neuron_repo.insert(n)
        await wm.add(n.id, priority=0.5)

        ids = await wm.get_active_ids()
        assert n.id in ids

    async def test_add_duplicate_updates_priority(self, db_conn, neuron_repo):
        wm = WorkingMemoryManager(db_conn)
        n = NeuronFactory.create(content="Test")
        await neuron_repo.insert(n)
        await wm.add(n.id, priority=0.3)
        await wm.add(n.id, priority=0.9)

        ids = await wm.get_active_ids()
        assert ids.count(n.id) == 1

    async def test_add_evicts_lowest_when_full(self, db_conn, neuron_repo):
        cfg = WorkingMemoryConfig(capacity=2)
        wm = WorkingMemoryManager(db_conn, cfg)

        neurons = []
        for i in range(3):
            n = NeuronFactory.create(content=f"Neuron {i}")
            await neuron_repo.insert(n)
            neurons.append(n)

        await wm.add(neurons[0].id, priority=0.5)
        await wm.add(neurons[1].id, priority=0.3)
        # This should evict neurons[1] (lowest priority)
        await wm.add(neurons[2].id, priority=0.7)

        ids = await wm.get_active_ids()
        assert len(ids) == 2
        assert neurons[1].id not in ids

    async def test_evicted_neuron_becomes_short_term(self, db_conn, neuron_repo):
        cfg = WorkingMemoryConfig(capacity=1)
        wm = WorkingMemoryManager(db_conn, cfg)

        n1 = NeuronFactory.create(content="First")
        n2 = NeuronFactory.create(content="Second")
        await neuron_repo.insert(n1)
        await neuron_repo.insert(n2)

        await wm.add(n1.id, priority=0.3)
        await wm.add(n2.id, priority=0.8)

        fetched = await neuron_repo.get_by_id(n1.id)
        assert fetched.layer == MemoryLayer.SHORT_TERM


class TestWorkingMemoryRemove:
    async def test_remove(self, db_conn, neuron_repo):
        wm = WorkingMemoryManager(db_conn)
        n = NeuronFactory.create(content="To remove")
        await neuron_repo.insert(n)
        await wm.add(n.id)
        await wm.remove(n.id)

        ids = await wm.get_active_ids()
        assert n.id not in ids

    async def test_remove_nonexistent_no_error(self, db_conn):
        wm = WorkingMemoryManager(db_conn)
        await wm.remove("nonexistent-id")


class TestWorkingMemoryCleanup:
    async def test_cleanup_expired(self, db_conn, neuron_repo):
        cfg = WorkingMemoryConfig(ttl_hours=1.0)
        wm = WorkingMemoryManager(db_conn, cfg)

        n = NeuronFactory.create(content="Old entry")
        await neuron_repo.insert(n)

        # Insert with a timestamp 2 hours ago (past TTL)
        two_hours_ago = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        async with db_conn.transaction() as conn:
            await conn.execute(
                "INSERT INTO working_memory (neuron_id, entered_at, priority) VALUES (?, ?, ?)",
                (n.id, two_hours_ago, 0.5),
            )

        expired = await wm.cleanup_expired()
        assert n.id in expired

        ids = await wm.get_active_ids()
        assert n.id not in ids

    async def test_cleanup_keeps_fresh(self, db_conn, neuron_repo):
        cfg = WorkingMemoryConfig(ttl_hours=24.0)
        wm = WorkingMemoryManager(db_conn, cfg)

        n = NeuronFactory.create(content="Fresh")
        await neuron_repo.insert(n)
        await wm.add(n.id)

        expired = await wm.cleanup_expired()
        assert n.id not in expired

        ids = await wm.get_active_ids()
        assert n.id in ids


class TestWorkingMemoryPromotion:
    async def test_get_promotion_candidates(self, db_conn, neuron_repo):
        cfg = WorkingMemoryConfig(
            promotion_strength_threshold=0.5,
            short_term_access_threshold=3,
        )
        wm = WorkingMemoryManager(db_conn, cfg)

        # Strong short-term neuron should be promoted
        n = NeuronFactory.create(content="Strong short-term")
        n.layer = MemoryLayer.SHORT_TERM
        n.strength = 0.8
        await neuron_repo.insert(n)

        candidates = await wm.get_promotion_candidates()
        assert n.id in candidates

    async def test_weak_short_term_not_promoted(self, db_conn, neuron_repo):
        cfg = WorkingMemoryConfig(
            promotion_strength_threshold=0.5,
            short_term_access_threshold=10,
        )
        wm = WorkingMemoryManager(db_conn, cfg)

        n = NeuronFactory.create(content="Weak short-term", importance=0.1)
        n.layer = MemoryLayer.SHORT_TERM
        n.strength = 0.2
        await neuron_repo.insert(n)

        candidates = await wm.get_promotion_candidates()
        assert n.id not in candidates

    async def test_get_short_term_expiry_candidates(self, db_conn, neuron_repo):
        cfg = WorkingMemoryConfig(short_term_to_long_term_hours=24.0)
        wm = WorkingMemoryManager(db_conn, cfg)

        n = NeuronFactory.create(content="Old short-term")
        n.layer = MemoryLayer.SHORT_TERM
        n.created_at = datetime.now(timezone.utc) - timedelta(hours=48)
        await neuron_repo.insert(n)

        candidates = await wm.get_short_term_expiry_candidates()
        assert n.id in candidates


class TestWorkingMemoryStats:
    async def test_stats(self, db_conn, neuron_repo):
        cfg = WorkingMemoryConfig(capacity=20, ttl_hours=2.0)
        wm = WorkingMemoryManager(db_conn, cfg)

        n = NeuronFactory.create(content="Stats test")
        await neuron_repo.insert(n)
        await wm.add(n.id)

        st = await wm.stats()
        assert st["active_count"] == 1
        assert st["capacity"] == 20
        assert st["ttl_hours"] == 2.0

    async def test_stats_empty(self, db_conn):
        wm = WorkingMemoryManager(db_conn)
        st = await wm.stats()
        assert st["active_count"] == 0
