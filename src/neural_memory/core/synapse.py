"""Synapse data model - weighted connections between neurons."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class SynapseType(Enum):
    """Types of connections between neurons."""
    SEMANTIC = "semantic"           # Meaning-based association
    TEMPORAL = "temporal"           # Time-based co-occurrence
    CAUSAL = "causal"               # Cause-effect relationship
    HIERARCHICAL = "hierarchical"   # Parent-child (schema to instances)
    REFERENCE = "reference"         # Explicit user-created link


@dataclass
class Synapse:
    """A weighted connection between two neurons (synapse)."""

    id: str
    pre_neuron_id: str      # Source neuron
    post_neuron_id: str     # Target neuron
    synapse_type: SynapseType
    weight: float = 0.5     # Connection strength [0, 1]
    activation_count: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_activated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def hebbian_update(self, learning_rate: float = 0.05, max_weight: float = 1.0) -> None:
        """Strengthen this synapse using Hebbian learning.

        Uses asymptotic update to prevent weight explosion:
        w += lr * (max - w)
        """
        self.weight += learning_rate * (max_weight - self.weight)
        self.weight = min(self.weight, max_weight)
        self.activation_count += 1
        self.last_activated = datetime.now(timezone.utc)

    def decay_weight(self, factor: float = 0.995) -> None:
        """Slightly decay synapse weight (for unused connections)."""
        self.weight *= factor
        if self.weight < 0.01:
            self.weight = 0.0
