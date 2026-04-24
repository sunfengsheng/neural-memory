"""Async spreading activation algorithm for associative retrieval.

Mimics neural spreading activation: starting from seed neurons,
activation propagates through synapses with decay at each hop.
Strong connections transmit more activation.
"""

from __future__ import annotations

import heapq
import logging

from neural_memory.config import ActivationConfig
from neural_memory.db.repositories import SynapseRepository

logger = logging.getLogger(__name__)


class SpreadingActivation:
    """Graph-based activation spreading through synapse network."""

    def __init__(self, synapse_repo: SynapseRepository, config: ActivationConfig | None = None):
        self._synapse_repo = synapse_repo
        self._cfg = config or ActivationConfig()

    async def activate(
        self,
        seed_ids: list[str],
        seed_activations: list[float] | None = None,
    ) -> dict[str, float]:
        """Spread activation from seed neurons through the synapse graph.

        Uses priority-queue BFS (best-first) for efficient traversal.
        """
        if not seed_ids:
            return {}

        if seed_activations is None:
            seed_activations = [self._cfg.initial_activation] * len(seed_ids)

        activation: dict[str, float] = {}
        heap: list[tuple[float, int, str]] = []

        for nid, act in zip(seed_ids, seed_activations):
            activation[nid] = act
            heapq.heappush(heap, (-act, 0, nid))

        visited_count = 0

        while heap and visited_count < self._cfg.max_nodes_visited:
            neg_act, hops, current_id = heapq.heappop(heap)
            current_act = -neg_act
            visited_count += 1

            if hops >= self._cfg.max_hops:
                continue
            if current_act < self._cfg.min_activation:
                continue

            synapses = await self._synapse_repo.get_connections(
                current_id, min_weight=0.05, limit=50
            )

            for syn in synapses:
                neighbor_id = (
                    syn.post_neuron_id
                    if syn.pre_neuron_id == current_id
                    else syn.pre_neuron_id
                )

                propagated = current_act * self._cfg.decay_per_hop * (syn.weight ** 1.5)

                if propagated < self._cfg.min_activation:
                    continue

                old = activation.get(neighbor_id, 0.0)
                new_act = old + propagated

                if new_act > old + self._cfg.min_activation * 0.5:
                    activation[neighbor_id] = new_act
                    heapq.heappush(heap, (-new_act, hops + 1, neighbor_id))

        logger.debug(
            "Spreading activation: %d seeds -> %d activated neurons (%d visited)",
            len(seed_ids), len(activation), visited_count,
        )

        return activation

    def get_top_activated(
        self, activation: dict[str, float], top_k: int | None = None
    ) -> list[tuple[str, float]]:
        """Sort activation results and return top-k."""
        k = top_k or self._cfg.top_k
        sorted_items = sorted(activation.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:k]
