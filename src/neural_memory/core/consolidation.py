"""Memory consolidation engine.

Consolidation mimics sleep-like memory processing:
1. Cluster similar long-term memories by embedding similarity
2. Generate schema neurons (abstract summaries) for each cluster
3. Link schema neurons to their source neurons via hierarchical synapses

This reduces redundancy and creates higher-level abstractions.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

import numpy as np

from neural_memory.config import ConsolidationConfig
from neural_memory.core.neuron import Neuron, NeuronType, MemoryLayer, NeuronFactory
from neural_memory.core.synapse import Synapse, SynapseType

logger = logging.getLogger(__name__)


class ConsolidationEngine:
    """Clusters similar memories and creates schema neurons."""

    def __init__(self, config: ConsolidationConfig | None = None):
        self._cfg = config or ConsolidationConfig()

    def find_clusters(
        self, neurons: list[Neuron], embeddings: dict[str, np.ndarray]
    ) -> list[list[Neuron]]:
        """Find clusters of similar neurons using greedy agglomerative clustering.

        Args:
            neurons: List of neurons to cluster (should be long-term memories).
            embeddings: Mapping from neuron ID to embedding vector.

        Returns:
            List of clusters, each cluster is a list of neurons.
        """
        if len(neurons) < self._cfg.min_cluster_size:
            return []

        # Filter neurons that have embeddings
        valid = [(n, embeddings[n.id]) for n in neurons if n.id in embeddings]
        if len(valid) < self._cfg.min_cluster_size:
            return []

        # Build similarity matrix
        ids = [n.id for n, _ in valid]
        vecs = np.stack([v for _, v in valid])
        # Normalize for cosine similarity
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        vecs_norm = vecs / norms
        sim_matrix = vecs_norm @ vecs_norm.T

        # Greedy clustering: assign each neuron to the first cluster it fits
        assigned: set[int] = set()
        clusters: list[list[int]] = []

        for i in range(len(ids)):
            if i in assigned:
                continue

            cluster = [i]
            assigned.add(i)

            for j in range(i + 1, len(ids)):
                if j in assigned:
                    continue
                if len(cluster) >= self._cfg.max_cluster_size:
                    break

                # Check similarity with all cluster members
                avg_sim = np.mean([sim_matrix[j, k] for k in cluster])
                if avg_sim >= self._cfg.similarity_threshold:
                    cluster.append(j)
                    assigned.add(j)

            if len(cluster) >= self._cfg.min_cluster_size:
                clusters.append(cluster)

        # Convert indices back to neurons
        neuron_map = {i: n for i, (n, _) in enumerate(valid)}
        result = [[neuron_map[i] for i in c] for c in clusters]

        logger.info(
            "Consolidation found %d clusters from %d neurons",
            len(result), len(valid),
        )
        return result

    def create_schema_neuron(
        self, cluster: list[Neuron], cluster_embeddings: list[np.ndarray]
    ) -> tuple[Neuron, np.ndarray, list[Synapse]]:
        """Create a schema neuron from a cluster of related memories.

        Args:
            cluster: List of neurons in the cluster.
            cluster_embeddings: Corresponding embedding vectors.

        Returns:
            (schema_neuron, schema_embedding, synapses)
        """
        # Generate content for the schema neuron
        # Combine summaries/content snippets from cluster members
        snippets = []
        for n in cluster:
            text = n.summary or n.content
            if len(text) > 150:
                text = text[:147] + "..."
            snippets.append(text)

        combined = "\n---\n".join(snippets)
        schema_content = f"[Schema: consolidation of {len(cluster)} related memories]\n\n{combined}"

        # Average importance and emotional markers
        avg_importance = np.mean([n.importance for n in cluster])
        max_arousal = max(n.emotional_arousal for n in cluster)
        avg_valence = np.mean([n.emotional_valence for n in cluster])

        # Collect all tags
        all_tags = set()
        for n in cluster:
            all_tags.update(n.tags)

        schema = NeuronFactory.create(
            content=schema_content,
            neuron_type=NeuronType.SCHEMA,
            importance=float(min(avg_importance + 0.1, 1.0)),  # Slight boost
            emotional_valence=float(avg_valence),
            emotional_arousal=float(max_arousal),
            tags=sorted(all_tags),
            source="consolidation",
        )
        # Schema goes directly to long-term memory
        schema.layer = MemoryLayer.LONG_TERM
        schema.stability = 3.0  # Schemas are more stable

        # Compute schema embedding as centroid of cluster
        schema_embedding = np.mean(np.stack(cluster_embeddings), axis=0).astype(np.float32)
        # Re-normalize
        norm = np.linalg.norm(schema_embedding)
        if norm > 0:
            schema_embedding = schema_embedding / norm

        # Create hierarchical synapses: schema → each cluster member
        synapses = []
        now = datetime.now(timezone.utc)
        for n in cluster:
            synapse = Synapse(
                id=str(uuid.uuid4()),
                pre_neuron_id=schema.id,
                post_neuron_id=n.id,
                synapse_type=SynapseType.HIERARCHICAL,
                weight=0.8,
                activation_count=0,
                created_at=now,
                last_activated=now,
            )
            synapses.append(synapse)

        logger.info(
            "Created schema neuron %s from %d members (tags: %s)",
            schema.id[:8], len(cluster), schema.tags,
        )

        return schema, schema_embedding, synapses
