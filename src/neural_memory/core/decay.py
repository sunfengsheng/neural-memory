"""Ebbinghaus forgetting curve engine.

Memory strength decays exponentially over time:
    S(t) = S0 * exp(-lambda * t / stability)

Higher stability (from repeated access) slows decay.
Important and emotional memories also decay slower.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone

from neural_memory.config import DecayConfig
from neural_memory.core.neuron import Neuron

logger = logging.getLogger(__name__)


class DecayEngine:
    """Applies Ebbinghaus-style exponential decay to neuron strength."""

    def __init__(self, config: DecayConfig | None = None):
        self._cfg = config or DecayConfig()

    def compute_decayed_strength(self, neuron: Neuron, now: datetime | None = None) -> float:
        """Calculate the decayed strength of a neuron at time `now`.

        Args:
            neuron: The neuron to compute decay for.
            now: Reference time (defaults to UTC now).

        Returns:
            New strength value in [0, 1].
        """
        if now is None:
            now = datetime.now(timezone.utc)

        hours_elapsed = (now - neuron.last_decayed).total_seconds() / 3600.0
        if hours_elapsed <= 0:
            return neuron.strength

        # Effective decay rate, modulated by stability
        effective_rate = self._cfg.base_decay_rate / max(neuron.stability, 0.1)

        # Slow decay for important memories
        if neuron.importance > self._cfg.importance_shield:
            effective_rate *= 0.5

        # Slow decay for emotionally charged memories
        if neuron.emotional_arousal > self._cfg.emotional_arousal_shield:
            effective_rate *= 0.6

        new_strength = neuron.strength * math.exp(-effective_rate * hours_elapsed)
        return max(new_strength, 0.0)

    def should_prune(self, neuron: Neuron) -> bool:
        """Check if a neuron should be pruned (forgotten).

        A neuron is pruned if its strength is below the threshold
        AND it is not shielded by high importance or emotional arousal.
        """
        if neuron.strength >= self._cfg.prune_threshold:
            return False
        if neuron.importance > self._cfg.importance_shield:
            return False
        if neuron.emotional_arousal > self._cfg.emotional_arousal_shield:
            return False
        return True

    def reinforce(self, neuron: Neuron) -> None:
        """Reinforce a neuron on access (restore strength, increase stability).

        Also auto-increases importance when the neuron is frequently accessed
        (access_count >= 5), capped at 1.0.

        Modifies the neuron in place.
        """
        neuron.strength = 1.0
        neuron.stability = min(
            neuron.stability + self._cfg.stability_growth_on_review,
            self._cfg.max_stability,
        )
        neuron.access_count += 1
        neuron.last_accessed = datetime.now(timezone.utc)
        neuron.last_decayed = datetime.now(timezone.utc)

        # Importance self-adaptation: auto-boost for frequently accessed memories
        if neuron.access_count >= 5 and neuron.importance < 1.0:
            neuron.importance = min(neuron.importance + 0.05, 1.0)

    def batch_decay(
        self, neurons: list[Neuron], now: datetime | None = None
    ) -> tuple[list[tuple[float, str, str]], list[str]]:
        """Compute decay for a batch of neurons.

        Returns:
            (updates, prune_ids)
            - updates: list of (new_strength, last_decayed_iso, neuron_id) for DB update
            - prune_ids: list of neuron IDs that should be pruned
        """
        if now is None:
            now = datetime.now(timezone.utc)

        updates: list[tuple[float, str, str]] = []
        prune_ids: list[str] = []
        now_iso = now.isoformat()

        for neuron in neurons:
            new_strength = self.compute_decayed_strength(neuron, now)
            neuron.strength = new_strength

            if self.should_prune(neuron):
                prune_ids.append(neuron.id)
            else:
                updates.append((new_strength, now_iso, neuron.id))

        if prune_ids:
            logger.info("Decay pass: pruning %d weak neurons", len(prune_ids))

        return updates, prune_ids
