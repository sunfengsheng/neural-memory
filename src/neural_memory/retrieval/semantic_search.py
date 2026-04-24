"""Brute-force semantic search using numpy cosine similarity.

Maintains an in-memory index of all neuron embeddings for fast retrieval.
For the expected scale (<100K neurons), brute-force is fast enough
and avoids the complexity of approximate nearest-neighbor libraries.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import numpy as np

from neural_memory.db.repositories import NeuronRepository

logger = logging.getLogger(__name__)


class SemanticSearch:
    """Vector similarity search over neuron embeddings."""

    def __init__(self, neuron_repo: NeuronRepository, dimension: int = 384):
        self._neuron_repo = neuron_repo
        self._dimension = dimension

        # In-memory index
        self._ids: list[str] = []
        self._matrix: np.ndarray | None = None  # Shape (N, D), normalized
        self._index_lock = asyncio.Lock()

    async def build_index(self) -> int:
        """Build/rebuild the in-memory index from the database.

        Returns:
            Number of indexed neurons.
        """
        pairs = await self._neuron_repo.get_embedding_batch()

        async with self._index_lock:
            if not pairs:
                self._ids = []
                self._matrix = None
                return 0

            ids = []
            vecs = []
            for nid, emb_blob in pairs:
                vec = np.frombuffer(emb_blob, dtype=np.float32).copy()
                if vec.shape[0] == self._dimension:
                    ids.append(nid)
                    vecs.append(vec)

            if not ids:
                self._ids = []
                self._matrix = None
                return 0

            self._ids = ids
            matrix = np.stack(vecs)

            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            self._matrix = matrix / norms

            logger.info("Semantic index built: %d vectors of dimension %d", len(ids), self._dimension)
            return len(ids)

    async def add_to_index(self, neuron_id: str, embedding: np.ndarray) -> None:
        """Add a single embedding to the in-memory index (avoids full rebuild)."""
        vec = embedding.astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        async with self._index_lock:
            if self._matrix is None:
                self._ids = [neuron_id]
                self._matrix = vec.reshape(1, -1)
            else:
                if neuron_id in self._ids:
                    idx = self._ids.index(neuron_id)
                    self._matrix[idx] = vec
                else:
                    self._ids.append(neuron_id)
                    self._matrix = np.vstack([self._matrix, vec.reshape(1, -1)])

    async def remove_from_index(self, neuron_id: str) -> None:
        """Remove a neuron from the in-memory index."""
        async with self._index_lock:
            if neuron_id in self._ids:
                idx = self._ids.index(neuron_id)
                self._ids.pop(idx)
                if self._matrix is not None and len(self._ids) > 0:
                    self._matrix = np.delete(self._matrix, idx, axis=0)
                else:
                    self._matrix = None

    def search(
        self, query_embedding: np.ndarray, top_k: int = 20, min_score: float = 0.0
    ) -> list[tuple[str, float]]:
        """Find the most similar neurons to a query embedding.

        Pure numpy computation — kept synchronous since it's fast and read-only.
        """
        if self._matrix is None or len(self._ids) == 0:
            return []

        qvec = query_embedding.astype(np.float32)
        qnorm = np.linalg.norm(qvec)
        if qnorm > 0:
            qvec = qvec / qnorm

        scores = self._matrix @ qvec

        if top_k < len(scores):
            top_indices = np.argpartition(scores, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        else:
            top_indices = np.argsort(scores)[::-1]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score >= min_score:
                results.append((self._ids[idx], score))

        return results

    @property
    def index_size(self) -> int:
        """Number of vectors in the index."""
        return len(self._ids)
