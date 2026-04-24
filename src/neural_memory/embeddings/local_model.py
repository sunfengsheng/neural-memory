"""Local async embedding model using sentence-transformers with caching."""

from __future__ import annotations

import asyncio
import hashlib
import logging

import numpy as np

from neural_memory.embeddings.provider import EmbeddingProvider
from neural_memory.db.connection import DatabaseConnection

logger = logging.getLogger(__name__)


class LocalEmbeddingModel(EmbeddingProvider):
    """Local CPU embedding using sentence-transformers.

    Default model: paraphrase-multilingual-MiniLM-L12-v2
      - 384 dimensions
      - ~470MB download
      - ~10ms per sentence on CPU
      - Supports 50+ languages including Chinese and English
    """

    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        cache_db: DatabaseConnection | None = None,
        device: str = "cpu",
    ):
        self._model_name = model_name
        self._device = device
        self._cache_db = cache_db
        self._model = None  # Lazy load

    def _get_model(self):
        """Lazy-load the model on first use."""
        if self._model is None:
            logger.info("Loading embedding model: %s (this may take a moment)...", self._model_name)
            from sentence_transformers import SentenceTransformer
            try:
                self._model = SentenceTransformer(self._model_name, device=self._device)
            except (OSError, RuntimeError):
                logger.info("Online loading failed, falling back to local cache...")
                self._model = SentenceTransformer(self._model_name, device=self._device, local_files_only=True)
            logger.info("Embedding model loaded successfully.")
        return self._model

    async def warmup(self) -> None:
        """Pre-load the model in a worker thread to avoid blocking the event loop on first use."""
        if self._model is not None:
            return
        await asyncio.to_thread(self._get_model)
        # Run a tiny encode to fully initialize tokenizer/weights
        await asyncio.to_thread(
            self._model.encode, "warmup", normalize_embeddings=True, show_progress_bar=False
        )

    @property
    def dimension(self) -> int:
        return 384

    @property
    def model_name(self) -> str:
        return self._model_name

    async def embed(self, text: str) -> np.ndarray:
        """Embed a single text with caching."""
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

        cached = await self._check_cache(text_hash)
        if cached is not None:
            return cached

        model = self._get_model()
        vec = await asyncio.to_thread(
            model.encode, text, normalize_embeddings=True, show_progress_bar=False
        )
        vec = vec.astype(np.float32)

        await self._store_cache(text_hash, vec)
        return vec

    async def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed multiple texts with caching for individual items."""
        hashes = [hashlib.sha256(t.encode("utf-8")).hexdigest() for t in texts]
        results: list[np.ndarray | None] = []
        for h in hashes:
            results.append(await self._check_cache(h))

        uncached_indices = [i for i, r in enumerate(results) if r is None]
        if uncached_indices:
            uncached_texts = [texts[i] for i in uncached_indices]
            model = self._get_model()
            vecs = await asyncio.to_thread(
                model.encode,
                uncached_texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=32,
            )
            vecs = vecs.astype(np.float32)

            for idx, vec in zip(uncached_indices, vecs):
                results[idx] = vec
                await self._store_cache(hashes[idx], vec)

        return np.stack(results)  # type: ignore

    async def _check_cache(self, text_hash: str) -> np.ndarray | None:
        """Check if an embedding is cached in the database."""
        if self._cache_db is None:
            return None
        try:
            async with self._cache_db.read() as conn:
                cursor = await conn.execute(
                    "SELECT embedding FROM embedding_cache WHERE text_hash = ? AND model = ?",
                    (text_hash, self._model_name),
                )
                row = await cursor.fetchone()
                if row and row["embedding"]:
                    return np.frombuffer(row["embedding"], dtype=np.float32).copy()
        except Exception:
            pass
        return None

    async def _store_cache(self, text_hash: str, vec: np.ndarray) -> None:
        """Store an embedding in the cache database."""
        if self._cache_db is None:
            return
        try:
            async with self._cache_db.transaction() as conn:
                await conn.execute(
                    """INSERT OR REPLACE INTO embedding_cache (text_hash, embedding, model)
                       VALUES (?, ?, ?)""",
                    (text_hash, vec.tobytes(), self._model_name),
                )
        except Exception:
            pass
