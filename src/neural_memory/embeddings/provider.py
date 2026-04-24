"""Abstract async embedding provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class EmbeddingProvider(ABC):
    """Base class for async embedding providers."""

    @abstractmethod
    async def embed(self, text: str) -> np.ndarray:
        """Embed a single text. Returns float32 array of shape (D,)."""
        ...

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed multiple texts. Returns float32 array of shape (N, D)."""
        ...

    async def warmup(self) -> None:
        """Pre-load model weights. Default is a no-op."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding dimension."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name of the embedding model."""
        ...
