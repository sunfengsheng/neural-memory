"""Shared test fixtures for neural-memory-mcp test suite."""

from __future__ import annotations

import hashlib
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from neural_memory.config import Config
from neural_memory.core.neuron import MemoryLayer, Neuron, NeuronFactory, NeuronType
from neural_memory.core.synapse import Synapse, SynapseType
from neural_memory.db.connection import DatabaseConnection
from neural_memory.db.repositories import NeuronRepository, SynapseRepository
from neural_memory.embeddings.provider import EmbeddingProvider

DIMENSION = 384
SCHEMA_PATH = Path(__file__).parent.parent / "src" / "neural_memory" / "db" / "schema.sql"


class FakeEmbeddingProvider(EmbeddingProvider):
    """Deterministic hash-based embedding for tests. No model download needed."""

    async def embed(self, text: str) -> np.ndarray:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        rng = np.random.RandomState(int.from_bytes(h[:4], "big"))
        vec = rng.randn(DIMENSION).astype(np.float32)
        vec /= np.linalg.norm(vec)
        return vec

    async def embed_batch(self, texts: list[str]) -> np.ndarray:
        vecs = [await self.embed(t) for t in texts]
        return np.stack(vecs)

    @property
    def dimension(self) -> int:
        return DIMENSION

    @property
    def model_name(self) -> str:
        return "fake-test-model"


@pytest.fixture
def tmp_dir(tmp_path):
    """Provide a temporary directory for test storage."""
    return tmp_path


@pytest.fixture
def config(tmp_dir):
    """Create a Config pointing at a temporary directory."""
    return Config(storage_dir=str(tmp_dir))


@pytest.fixture
async def db_conn(tmp_dir):
    """Async fixture: initialized DatabaseConnection with schema loaded."""
    db_path = tmp_dir / "test.db"
    db = DatabaseConnection(db_path)
    await db.initialize()
    await db.initialize_schema(SCHEMA_PATH)
    yield db
    await db.close()


@pytest.fixture
def neuron_repo(db_conn):
    return NeuronRepository(db_conn)


@pytest.fixture
def synapse_repo(db_conn):
    return SynapseRepository(db_conn)


@pytest.fixture
def fake_embedder():
    return FakeEmbeddingProvider()


@pytest.fixture
def sample_neuron():
    """Create a sample neuron for tests."""
    return NeuronFactory.create(
        content="Python is a programming language",
        neuron_type=NeuronType.SEMANTIC,
        importance=0.7,
        tags=["python", "programming"],
        source="test",
    )


@pytest.fixture
def sample_synapse():
    """Create a sample synapse for tests."""
    now = datetime.now(timezone.utc)
    return Synapse(
        id=str(uuid.uuid4()),
        pre_neuron_id="pre-id",
        post_neuron_id="post-id",
        synapse_type=SynapseType.SEMANTIC,
        weight=0.7,
        created_at=now,
        last_activated=now,
    )


@pytest.fixture
async def memory_store(tmp_dir, fake_embedder):
    """Create a MemoryStore with fake embedder (no model download)."""
    from neural_memory.core.memory_store import MemoryStore

    cfg = Config(storage_dir=str(tmp_dir))
    store = MemoryStore(cfg)
    store._embedder = fake_embedder
    await store.initialize()
    yield store
    await store.close()
