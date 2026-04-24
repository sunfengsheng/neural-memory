"""Tests for neural_memory.embeddings — provider + local_model."""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import FakeEmbeddingProvider


class TestFakeEmbeddingProvider:
    """Test the FakeEmbeddingProvider used in all tests."""

    async def test_embed_returns_correct_shape(self, fake_embedder):
        vec = await fake_embedder.embed("Hello world")
        assert vec.shape == (384,)
        assert vec.dtype == np.float32

    async def test_embed_is_normalized(self, fake_embedder):
        vec = await fake_embedder.embed("Test text")
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-5

    async def test_embed_deterministic(self, fake_embedder):
        v1 = await fake_embedder.embed("Same text")
        v2 = await fake_embedder.embed("Same text")
        np.testing.assert_array_equal(v1, v2)

    async def test_embed_different_texts_different_vectors(self, fake_embedder):
        v1 = await fake_embedder.embed("Apple")
        v2 = await fake_embedder.embed("Banana")
        assert not np.allclose(v1, v2)

    async def test_embed_batch(self, fake_embedder):
        texts = ["Hello", "World", "Python"]
        batch = await fake_embedder.embed_batch(texts)
        assert batch.shape == (3, 384)

        # Each row should match individual embed
        for i, t in enumerate(texts):
            single = await fake_embedder.embed(t)
            np.testing.assert_array_equal(batch[i], single)

    async def test_dimension_property(self, fake_embedder):
        assert fake_embedder.dimension == 384

    async def test_model_name_property(self, fake_embedder):
        assert fake_embedder.model_name == "fake-test-model"


class TestEmbeddingProviderInterface:
    """Ensure our FakeEmbeddingProvider satisfies the abstract interface."""

    def test_is_instance_of_provider(self, fake_embedder):
        from neural_memory.embeddings.provider import EmbeddingProvider
        assert isinstance(fake_embedder, EmbeddingProvider)
