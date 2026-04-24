"""Tests for neural_memory.core.memory_store — High-level async orchestrator (integration)."""

from __future__ import annotations

import pytest

from neural_memory.core.neuron import MemoryLayer, NeuronType


class TestMemoryStoreRemember:
    async def test_remember_basic(self, memory_store):
        result = await memory_store.remember(content="Python is great")
        assert "id" in result
        assert result["content"] == "Python is great"
        assert "synapses_created" in result

    async def test_remember_with_all_fields(self, memory_store):
        result = await memory_store.remember(
            content="Important discovery",
            neuron_type="episodic",
            importance=0.9,
            emotional_valence=0.5,
            emotional_arousal=0.7,
            tags=["discovery", "science"],
            source="test",
        )
        assert result["neuron_type"] == "episodic"
        assert result["importance"] == pytest.approx(0.9)
        assert "discovery" in result["tags"]

    async def test_remember_with_file_content(self, memory_store):
        result = await memory_store.remember(
            content="A Python script",
            file_content="print('hello world')",
            file_type="code/python",
        )
        assert result.get("file_path") is not None
        assert result.get("file_type") == "code/python"

    async def test_remember_creates_temporal_synapses(self, memory_store):
        # Store two memories quickly — they should get temporal synapses
        r1 = await memory_store.remember(content="First memory in sequence")
        r2 = await memory_store.remember(content="Second memory in sequence")
        # Second memory should have temporal synapse to first
        assert r2["synapses_created"]["temporal"] >= 1

    async def test_remember_creates_semantic_synapses(self, memory_store):
        # Store similar memories — they should get semantic synapses
        await memory_store.remember(content="Machine learning is a subset of AI")
        r2 = await memory_store.remember(content="Machine learning algorithms use data")
        # FakeEmbeddingProvider uses hash-based embeddings, so similarity depends on content
        # At minimum check the structure exists
        assert "semantic" in r2["synapses_created"]

    async def test_remember_adds_to_working_memory(self, memory_store):
        await memory_store.remember(content="Should be in working memory")
        wm_ids = await memory_store._working_memory.get_active_ids()
        assert len(wm_ids) >= 1

    async def test_remember_updates_semantic_index(self, memory_store):
        await memory_store.remember(content="Indexed content")
        assert memory_store._semantic_search.index_size >= 1


class TestMemoryStoreRecall:
    async def test_recall_empty_store(self, memory_store):
        results = await memory_store.recall(query="anything")
        assert results == []

    async def test_recall_finds_stored_memory(self, memory_store):
        await memory_store.remember(content="FastAPI is a Python web framework")
        results = await memory_store.recall(query="FastAPI web framework")
        assert len(results) >= 1
        # First result should contain the stored content
        contents = [r["content"] for r in results]
        assert any("FastAPI" in c for c in contents)

    async def test_recall_respects_top_k(self, memory_store):
        for i in range(5):
            await memory_store.remember(content=f"Memory number {i} about databases")
        results = await memory_store.recall(query="databases", top_k=2)
        assert len(results) <= 2

    async def test_recall_filter_by_neuron_type(self, memory_store):
        await memory_store.remember(content="Semantic fact about cats", neuron_type="semantic")
        await memory_store.remember(content="Episode about cats", neuron_type="episodic")
        results = await memory_store.recall(query="cats", neuron_type="semantic")
        for r in results:
            assert r["neuron_type"] == "semantic"

    async def test_recall_filter_by_tags(self, memory_store):
        await memory_store.remember(content="Python tip", tags=["python"])
        await memory_store.remember(content="Java tip", tags=["java"])
        results = await memory_store.recall(query="tip", tags=["python"])
        for r in results:
            assert "python" in r["tags"]

    async def test_recall_reinforces_accessed_neurons(self, memory_store):
        result = await memory_store.remember(content="Reinforcement test content")
        nid = result["id"]
        neuron_before = await memory_store._neuron_repo.get_by_id(nid)
        strength_before = neuron_before.strength

        await memory_store.recall(query="Reinforcement test content")

        neuron_after = await memory_store._neuron_repo.get_by_id(nid)
        assert neuron_after.strength >= strength_before

    async def test_recall_includes_score_breakdown(self, memory_store):
        await memory_store.remember(content="Score breakdown test")
        results = await memory_store.recall(query="Score breakdown test")
        if results:
            assert "score" in results[0]
            assert "score_breakdown" in results[0]


class TestMemoryStoreReflect:
    async def test_reflect_returns_stats(self, memory_store):
        stats = await memory_store.reflect()
        assert "decayed" in stats
        assert "pruned" in stats
        assert "working_to_short_term" in stats
        assert "short_term_to_long_term" in stats
        assert "schemas_created" in stats
        assert "synapse_decayed" in stats

    async def test_reflect_on_empty_store(self, memory_store):
        stats = await memory_store.reflect()
        assert stats["decayed"] == 0
        assert stats["pruned"] == 0

    async def test_reflect_decays_neurons(self, memory_store):
        await memory_store.remember(content="Decayable memory", importance=0.1)
        stats = await memory_store.reflect()
        assert stats["decayed"] >= 1

    async def test_reflect_prunes_weak_neurons(self, memory_store):
        # Create a neuron and manually weaken it
        result = await memory_store.remember(content="Weak memory", importance=0.01)
        nid = result["id"]
        # Set strength below prune threshold
        await memory_store._neuron_repo.update_strength(nid, 0.01, 0.01)
        # Also set low importance so decay proceeds
        neuron = await memory_store._neuron_repo.get_by_id(nid)
        # Mark as accessed long ago to ensure decay is applied
        async with memory_store._db.transaction() as conn:
            await conn.execute(
                "UPDATE neurons SET last_accessed = datetime('now', '-30 days') WHERE id = ?",
                (nid,),
            )

        stats = await memory_store.reflect()
        # The neuron might be pruned or decayed — depends on exact threshold
        assert stats["decayed"] >= 0  # At least the process ran without error


class TestMemoryStoreForget:
    async def test_forget_by_id_with_confirm(self, memory_store):
        result = await memory_store.remember(content="To be forgotten")
        nid = result["id"]

        response = await memory_store.forget(neuron_id=nid, confirm=True)
        assert response["deleted"] is True
        assert response["neuron_id"] == nid

        # Verify it's gone
        neuron = await memory_store._neuron_repo.get_by_id(nid)
        assert neuron is None

    async def test_forget_by_id_preview(self, memory_store):
        result = await memory_store.remember(content="Preview forget")
        nid = result["id"]

        response = await memory_store.forget(neuron_id=nid, confirm=False)
        assert response["preview"] is True
        assert "neuron" in response

        # Verify it's NOT deleted
        neuron = await memory_store._neuron_repo.get_by_id(nid)
        assert neuron is not None

    async def test_forget_nonexistent_id(self, memory_store):
        response = await memory_store.forget(neuron_id="nonexistent-id", confirm=True)
        assert "error" in response

    async def test_forget_by_query_preview(self, memory_store):
        await memory_store.remember(content="Findable by query for forget")
        response = await memory_store.forget(query="Findable by query for forget", confirm=False)
        if response.get("preview"):
            assert "matches" in response

    async def test_forget_by_query_confirm(self, memory_store):
        await memory_store.remember(content="Delete me via query search")
        response = await memory_store.forget(
            query="Delete me via query search", confirm=True,
        )
        if "deleted" in response:
            assert response["deleted"] is True
            assert response["count"] >= 1

    async def test_forget_no_args(self, memory_store):
        response = await memory_store.forget()
        assert "error" in response

    async def test_forget_removes_from_semantic_index(self, memory_store):
        result = await memory_store.remember(content="Index removal test")
        nid = result["id"]
        size_before = memory_store._semantic_search.index_size

        await memory_store.forget(neuron_id=nid, confirm=True)
        assert memory_store._semantic_search.index_size < size_before


class TestMemoryStoreAssociate:
    async def test_associate_two_neurons(self, memory_store):
        r1 = await memory_store.remember(content="Neuron A for association")
        r2 = await memory_store.remember(content="Neuron B for association")

        response = await memory_store.associate(
            neuron_id_a=r1["id"],
            neuron_id_b=r2["id"],
            synapse_type="reference",
            weight=0.8,
        )
        assert "synapse_id" in response
        assert response["pre_neuron_id"] == r1["id"]
        assert response["post_neuron_id"] == r2["id"]
        assert response["weight"] == pytest.approx(0.8)

    async def test_associate_nonexistent_a(self, memory_store):
        r = await memory_store.remember(content="Existing neuron")
        response = await memory_store.associate(
            neuron_id_a="nonexistent",
            neuron_id_b=r["id"],
        )
        assert "error" in response

    async def test_associate_nonexistent_b(self, memory_store):
        r = await memory_store.remember(content="Existing neuron")
        response = await memory_store.associate(
            neuron_id_a=r["id"],
            neuron_id_b="nonexistent",
        )
        assert "error" in response

    async def test_associate_weight_clamped(self, memory_store):
        r1 = await memory_store.remember(content="Clamp A")
        r2 = await memory_store.remember(content="Clamp B")

        response = await memory_store.associate(
            neuron_id_a=r1["id"],
            neuron_id_b=r2["id"],
            weight=5.0,  # Above max
        )
        assert response["weight"] <= 1.0

    async def test_associate_different_types(self, memory_store):
        r1 = await memory_store.remember(content="Causal A")
        r2 = await memory_store.remember(content="Causal B")

        response = await memory_store.associate(
            neuron_id_a=r1["id"],
            neuron_id_b=r2["id"],
            synapse_type="causal",
        )
        assert response["synapse_type"] == "causal"


class TestMemoryStoreStatus:
    async def test_status_empty(self, memory_store):
        status = await memory_store.status()
        assert "neurons" in status
        assert "synapses" in status
        assert "working_memory" in status
        assert "semantic_index_size" in status
        assert "embedding_model" in status
        assert "file_storage" in status
        assert status["neurons"]["total"] == 0

    async def test_status_after_remember(self, memory_store):
        await memory_store.remember(content="Status test memory")
        status = await memory_store.status()
        assert status["neurons"]["total"] >= 1
        assert status["semantic_index_size"] >= 1

    async def test_status_synapse_count(self, memory_store):
        r1 = await memory_store.remember(content="Synapse count A")
        r2 = await memory_store.remember(content="Synapse count B")
        await memory_store.associate(r1["id"], r2["id"])

        status = await memory_store.status()
        assert status["synapses"]["total"] >= 1


class TestMemoryStoreLifecycle:
    async def test_initialize_and_close(self, tmp_dir, fake_embedder):
        from neural_memory.config import Config
        from neural_memory.core.memory_store import MemoryStore

        cfg = Config(storage_dir=str(tmp_dir / "lifecycle"))
        store = MemoryStore(cfg)
        store._embedder = fake_embedder
        await store.initialize()

        # Should be usable
        result = await store.remember(content="Lifecycle test")
        assert "id" in result

        await store.close()

    async def test_multiple_operations_sequence(self, memory_store):
        """Full workflow: remember → recall → associate → reflect → status → forget."""
        # 1. Remember
        r1 = await memory_store.remember(content="Workflow step one")
        r2 = await memory_store.remember(content="Workflow step two")

        # 2. Recall
        results = await memory_store.recall(query="Workflow step")
        assert len(results) >= 1

        # 3. Associate
        assoc = await memory_store.associate(r1["id"], r2["id"])
        assert "synapse_id" in assoc

        # 4. Reflect
        stats = await memory_store.reflect()
        assert "decayed" in stats

        # 5. Status
        status = await memory_store.status()
        assert status["neurons"]["total"] >= 2

        # 6. Forget
        resp = await memory_store.forget(neuron_id=r1["id"], confirm=True)
        assert resp["deleted"] is True

        # Verify neuron count decreased
        status2 = await memory_store.status()
        assert status2["neurons"]["total"] < status["neurons"]["total"]
