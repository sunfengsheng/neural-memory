"""Neuron data model and factory - the core memory unit."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class NeuronType(Enum):
    """Types of neurons, inspired by human memory categories."""
    EPISODIC = "episodic"       # Events, experiences
    SEMANTIC = "semantic"       # Facts, knowledge
    PROCEDURAL = "procedural"   # How-to, procedures, code
    SCHEMA = "schema"           # Consolidated/abstract knowledge


class MemoryLayer(Enum):
    """Memory layers, from volatile to persistent."""
    WORKING = "working"         # Active buffer, limited capacity
    SHORT_TERM = "short_term"   # Recent memories, hours
    LONG_TERM = "long_term"     # Consolidated, persistent


@dataclass
class Neuron:
    """A single memory unit (neuron) in the neural memory system."""

    id: str
    content: str
    neuron_type: NeuronType
    layer: MemoryLayer = MemoryLayer.WORKING

    # Memory dynamics
    strength: float = 1.0       # Current memory strength [0, 1]
    stability: float = 1.0      # How resistant to decay (grows with review)
    importance: float = 0.5     # Intrinsic importance [0, 1]

    # Emotional markers
    emotional_valence: float = 0.0   # [-1, 1]: negative to positive
    emotional_arousal: float = 0.0   # [0, 1]: calm to intense

    # Access tracking
    access_count: int = 0

    # Embedding
    embedding: bytes | None = None
    embedding_model: str | None = None

    # Metadata
    summary: str | None = None
    tags: list[str] = field(default_factory=list)
    source: str = "manual"

    # File reference
    file_path: str | None = None
    file_type: str | None = None
    file_hash: str | None = None

    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_decayed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def tags_json(self) -> str:
        """Serialize tags to JSON string for storage."""
        return json.dumps(self.tags, ensure_ascii=False)

    @property
    def is_emotional(self) -> bool:
        """Check if this memory has significant emotional markers."""
        return abs(self.emotional_valence) > 0.3 or self.emotional_arousal > 0.3

    def to_dict(self) -> dict[str, Any]:
        """Convert to a dictionary for serialization."""
        return {
            "id": self.id,
            "content": self.content,
            "summary": self.summary,
            "neuron_type": self.neuron_type.value,
            "layer": self.layer.value,
            "strength": round(self.strength, 4),
            "stability": round(self.stability, 4),
            "importance": self.importance,
            "emotional_valence": self.emotional_valence,
            "emotional_arousal": self.emotional_arousal,
            "access_count": self.access_count,
            "tags": self.tags,
            "source": self.source,
            "file_path": self.file_path,
            "file_type": self.file_type,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
        }


class NeuronFactory:
    """Factory for creating Neuron instances with proper defaults."""

    @staticmethod
    def create(
        content: str,
        neuron_type: NeuronType = NeuronType.SEMANTIC,
        importance: float = 0.5,
        emotional_valence: float = 0.0,
        emotional_arousal: float = 0.0,
        tags: list[str] | None = None,
        source: str = "manual",
        file_path: str | None = None,
        file_type: str | None = None,
        file_hash: str | None = None,
        summary: str | None = None,
    ) -> Neuron:
        """Create a new neuron with a unique ID and current timestamps."""
        now = datetime.now(timezone.utc)

        # Generate summary if not provided (truncate content)
        if summary is None and len(content) > 200:
            summary = content[:197] + "..."

        return Neuron(
            id=str(uuid.uuid4()),
            content=content,
            summary=summary,
            neuron_type=neuron_type,
            layer=MemoryLayer.WORKING,
            strength=1.0,
            stability=1.0,
            importance=max(0.0, min(1.0, importance)),
            emotional_valence=max(-1.0, min(1.0, emotional_valence)),
            emotional_arousal=max(0.0, min(1.0, emotional_arousal)),
            access_count=0,
            tags=tags or [],
            source=source,
            file_path=file_path,
            file_type=file_type,
            file_hash=file_hash,
            created_at=now,
            last_accessed=now,
            last_decayed=now,
        )
