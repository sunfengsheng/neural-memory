"""Tests for neural_memory.retrieval.semantic_search — Vector similarity search."""

from __future__ import annotations

import numpy as np
import pytest

from neural_memory.core.neuron import NeuronFactory
from neural_memory.retrieval.semantic_search import SemanticSearch


class TestSemanticSearchBuildIndex:
    async def test_build_index_empty(self, neuron_repo):
        ss = SemanticSearch(neuron_repo, dimension=384)
        count = await ss.build_index()
        assert count == 0
        assert ss.index_size == 0

    async def test_build_index_with_neurons(self, neuron_repo):
        for i in range(3):
            n = NeuronFactory.create(content=f"Neuron {i}")
            n.embedding = np.random.randn(384).astype(np.float32).tobytes()
            await neuron_repo.insert(n)

        ss = SemanticSearch(neuron_repo, dimension=384)
        count = await ss.build_index()
        assert count == 3
        assert ss.index_size == 3

    async def test_build_index_skips_wrong_dimension(self, neuron_repo):
        n = NeuronFactory.create(content="Wrong dim")
        n.embedding = np.random.randn(128).astype(np.float32).tobytes()
        await neuron_repo.insert(n)

        ss = SemanticSearch(neuron_repo, dimension=384)
        count = await ss.build_index()
        assert count == 0

    async def test_build_index_skips_no_embedding(self, neuron_repo):
        n = NeuronFactory.create(content="No embedding")
        await neuron_repo.insert(n)

        ss = SemanticSearch(neuron_repo, dimension=384)
        count = await ss.build_index()
        assert count == 0


class TestSemanticSearchSearch:
    def _make_search_with_data(self):
        """Create a SemanticSearch and manually populate index.

        Vectors are constructed to have guaranteed positive cosine similarity
        with vec[0]: base + decreasing noise so id-a > id-b > id-c in score.
        """
        ss = SemanticSearch.__new__(SemanticSearch)
        ss._dimension = 384
        ss._neuron_repo = None
        ss._index_lock = __import__("asyncio").Lock()

        rng = np.random.RandomState(42)
        base = rng.randn(384).astype(np.float32)
        base /= np.linalg.norm(base)

        ids = ["id-a", "id-b", "id-c"]
        noise_scales = [0.0, 0.3, 0.6]  # id-a = exact, id-b close, id-c less close
        vecs = []
        for scale in noise_scales:
            v = base + scale * rng.randn(384).astype(np.float32)
            v /= np.linalg.norm(v)
            vecs.append(v)
        ss._ids = ids
        ss._matrix = np.stack(vecs)
        return ss, vecs

    def test_search_empty_index(self):
        ss = SemanticSearch.__new__(SemanticSearch)
        ss._dimension = 384
        ss._ids = []
        ss._matrix = None
        ss._index_lock = __import__("asyncio").Lock()

        query = np.random.randn(384).astype(np.float32)
        results = ss.search(query)
        assert results == []

    def test_search_returns_sorted_by_score(self):
        ss, vecs = self._make_search_with_data()
        # Query = first vector → should be most similar to id-a
        results = ss.search(vecs[0], top_k=3)
        assert len(results) == 3
        assert results[0][0] == "id-a"
        assert results[0][1] > 0.99

    def test_search_min_score_filter(self):
        ss, vecs = self._make_search_with_data()
        results = ss.search(vecs[0], top_k=3, min_score=0.99)
        # Only the exact match should pass
        assert len(results) == 1
        assert results[0][0] == "id-a"

    def test_search_top_k_limit(self):
        ss, vecs = self._make_search_with_data()
        results = ss.search(vecs[0], top_k=1)
        assert len(results) == 1


class TestSemanticSearchAddRemove:
    async def test_add_to_empty_index(self, neuron_repo):
        ss = SemanticSearch(neuron_repo, dimension=384)
        vec = np.random.randn(384).astype(np.float32)
        await ss.add_to_index("new-id", vec)
        assert ss.index_size == 1

        results = ss.search(vec, top_k=1)
        assert results[0][0] == "new-id"

    async def test_add_updates_existing(self, neuron_repo):
        ss = SemanticSearch(neuron_repo, dimension=384)
        vec1 = np.random.randn(384).astype(np.float32)
        vec2 = np.random.randn(384).astype(np.float32)
        await ss.add_to_index("same-id", vec1)
        await ss.add_to_index("same-id", vec2)
        assert ss.index_size == 1

        # Searching with vec2 should find same-id
        results = ss.search(vec2, top_k=1)
        assert results[0][0] == "same-id"

    async def test_remove_from_index(self, neuron_repo):
        ss = SemanticSearch(neuron_repo, dimension=384)
        vec = np.random.randn(384).astype(np.float32)
        await ss.add_to_index("to-remove", vec)
        assert ss.index_size == 1

        await ss.remove_from_index("to-remove")
        assert ss.index_size == 0

    async def test_remove_nonexistent_no_error(self, neuron_repo):
        ss = SemanticSearch(neuron_repo, dimension=384)
        await ss.remove_from_index("nonexistent")
        assert ss.index_size == 0

    async def test_remove_with_multiple(self, neuron_repo):
        ss = SemanticSearch(neuron_repo, dimension=384)
        v1 = np.random.randn(384).astype(np.float32)
        v2 = np.random.randn(384).astype(np.float32)
        await ss.add_to_index("id-1", v1)
        await ss.add_to_index("id-2", v2)
        assert ss.index_size == 2

        await ss.remove_from_index("id-1")
        assert ss.index_size == 1

        results = ss.search(v2, top_k=1)
        assert results[0][0] == "id-2"

    async def test_index_size_property(self, neuron_repo):
        ss = SemanticSearch(neuron_repo, dimension=384)
        assert ss.index_size == 0
        vec = np.random.randn(384).astype(np.float32)
        await ss.add_to_index("a", vec)
        assert ss.index_size == 1
