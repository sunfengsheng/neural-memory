"""High-level async memory orchestrator.

Integrates all subsystems:
- NeuronRepository / SynapseRepository (persistence)
- DecayEngine (forgetting curve)
- WorkingMemoryManager (limited-capacity buffer)
- ConsolidationEngine (schema generation)
- SemanticSearch (vector retrieval)
- SpreadingActivation (associative retrieval)
- HybridRanker (multi-signal ranking)
- LocalEmbeddingModel (text → vector)
- FileStore (code/file persistence)
"""

from __future__ import annotations

import logging
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from neural_memory.config import Config
from neural_memory.core.consolidation import ConsolidationEngine
from neural_memory.core.decay import DecayEngine
from neural_memory.core.neuron import MemoryLayer, Neuron, NeuronFactory, NeuronType
from neural_memory.core.synapse import Synapse, SynapseType
from neural_memory.core.working_memory import WorkingMemoryManager
from neural_memory.db.connection import DatabaseConnection
from neural_memory.db.repositories import NeuronRepository, SynapseRepository
from neural_memory.embeddings.local_model import LocalEmbeddingModel
from neural_memory.files.file_store import FileStore
from neural_memory.retrieval.hybrid_ranker import HybridRanker, RankedResult
from neural_memory.retrieval.semantic_search import SemanticSearch
from neural_memory.retrieval.spreading_activation import SpreadingActivation

logger = logging.getLogger(__name__)


class MemoryStore:
    """Top-level async orchestrator for the neural memory system.

    Provides high-level methods for:
    - remember: store new memory
    - recall: hybrid retrieval
    - reflect: decay + consolidation + layer promotion
    - forget: manual deletion
    - associate: manual synapse creation
    - status: system statistics

    Usage:
        store = MemoryStore(config)
        await store.initialize()   # must be called before any other method
    """

    def __init__(self, config: Config):
        self._config = config

        # Database (not yet connected — call initialize())
        self._db = DatabaseConnection(config.db_path)

        # Repositories
        self._neuron_repo = NeuronRepository(self._db)
        self._synapse_repo = SynapseRepository(self._db)

        # Core engines (pure computation, no async needed in __init__)
        self._decay = DecayEngine(config.decay_config)
        self._working_memory = WorkingMemoryManager(self._db, config.working_memory_config)
        self._consolidation = ConsolidationEngine(config.consolidation_config)

        # Embedding
        self._embedder = LocalEmbeddingModel(
            model_name=config.embedding_model,
            cache_db=self._db,
        )

        # Retrieval
        self._semantic_search = SemanticSearch(self._neuron_repo, config.embedding_dimension)
        self._spreading = SpreadingActivation(self._synapse_repo, config.activation_config)
        self._ranker = HybridRanker(config.ranker_config)

        # File storage
        self._file_store = FileStore(config.files_path)

        # Piggyback maintenance tracking
        self._last_maintenance = datetime.now(timezone.utc)

    async def initialize(self) -> None:
        """Async initialization — must be called after __init__.

        Opens the database connection, creates tables, pre-warms the
        embedding model, and builds the semantic search index.
        """
        schema_path = Path(__file__).parent.parent / "db" / "schema.sql"
        await self._db.initialize()
        await self._db.initialize_schema(schema_path)
        await self._embedder.warmup()
        await self._semantic_search.build_index()
        logger.info("MemoryStore initialized (index: %d vectors)", self._semantic_search.index_size)

    # ------------------------------------------------------------------
    # remember: store a new memory
    # ------------------------------------------------------------------
    async def remember(
        self,
        content: str,
        neuron_type: str = "semantic",
        importance: float = 0.5,
        emotional_valence: float = 0.0,
        emotional_arousal: float = 0.0,
        tags: list[str] | None = None,
        source: str = "manual",
        file_content: str | None = None,
        file_type: str | None = None,
        file_extension: str | None = None,
    ) -> dict[str, Any]:
        """Store a new memory (neuron) with auto-embedding and auto-association."""
        # 0. Piggyback maintenance
        await self._maybe_maintenance()

        # 1. File storage (optional)
        file_path = None
        file_hash = None
        if file_content is not None:
            stored = await self._file_store.store(
                content=file_content,
                file_type=file_type,
                original_extension=file_extension,
            )
            file_path = stored.relative_path
            file_type = stored.file_type
            file_hash = stored.file_hash

        # 2. Compute embedding
        embedding_vec = await self._embedder.embed(content)
        embedding_blob = embedding_vec.tobytes()

        # 2.5 Dedup check — if a near-identical memory exists, merge instead of creating new
        dup_id = await self._dedup_check(embedding_vec, threshold=0.9)
        if dup_id is not None:
            existing = await self._neuron_repo.get_by_id(dup_id)
            if existing is not None:
                # Merge: reinforce existing, update importance to max, merge tags
                self._decay.reinforce(existing)
                existing.importance = max(existing.importance, importance)
                merged_tags = list(set(existing.tags or []) | set(tags or []))
                existing.tags = merged_tags
                await self._neuron_repo.update_strength(
                    dup_id, existing.strength, existing.stability,
                )
                await self._neuron_repo.update_importance(dup_id, existing.importance)
                await self._neuron_repo.update_tags(dup_id, merged_tags)

                logger.info(
                    "Dedup merge: new content merged into existing neuron %s (similarity > 0.9)",
                    dup_id[:8],
                )
                result = existing.to_dict()
                result["merged_into"] = dup_id
                result["synapses_created"] = {"temporal": 0, "semantic": 0}
                return result

        # 3. Create neuron
        ntype = NeuronType(neuron_type)
        neuron = NeuronFactory.create(
            content=content,
            neuron_type=ntype,
            importance=importance,
            emotional_valence=emotional_valence,
            emotional_arousal=emotional_arousal,
            tags=tags,
            source=source,
            file_path=file_path,
            file_type=file_type,
            file_hash=file_hash,
        )
        neuron.embedding = embedding_blob
        neuron.embedding_model = self._embedder.model_name

        # 4. Persist
        await self._neuron_repo.insert(neuron)

        # 5. Temporal synapses (memories created within 30 min window)
        temporal_neighbors = await self._neuron_repo.get_temporal_neighbors(
            neuron.created_at, window_seconds=1800, limit=5,
        )
        for neighbor in temporal_neighbors:
            if neighbor.id == neuron.id:
                continue
            syn = Synapse(
                id=str(uuid.uuid4()),
                pre_neuron_id=neuron.id,
                post_neuron_id=neighbor.id,
                synapse_type=SynapseType.TEMPORAL,
                weight=0.4,
                created_at=datetime.now(timezone.utc),
                last_activated=datetime.now(timezone.utc),
            )
            await self._synapse_repo.upsert(syn)

        # 6. Semantic synapses (top-3 most similar, cosine > 0.6)
        sem_hits = self._semantic_search.search(embedding_vec, top_k=3, min_score=0.6)
        for hit_id, sim_score in sem_hits:
            if hit_id == neuron.id:
                continue
            syn = Synapse(
                id=str(uuid.uuid4()),
                pre_neuron_id=neuron.id,
                post_neuron_id=hit_id,
                synapse_type=SynapseType.SEMANTIC,
                weight=min(sim_score, 1.0),
                created_at=datetime.now(timezone.utc),
                last_activated=datetime.now(timezone.utc),
            )
            await self._synapse_repo.upsert(syn)

        # 7. Update semantic index and add to working memory
        await self._semantic_search.add_to_index(neuron.id, embedding_vec)
        await self._working_memory.add(neuron.id, priority=importance)

        logger.info(
            "Remembered: %s (type=%s, importance=%.2f, synapses=%d)",
            neuron.id[:8], neuron_type, importance,
            len(temporal_neighbors) - 1 + len(sem_hits),
        )

        result = neuron.to_dict()
        result["synapses_created"] = {
            "temporal": max(0, len(temporal_neighbors) - 1),
            "semantic": len(sem_hits),
        }
        return result

    # ------------------------------------------------------------------
    # recall: hybrid retrieval
    # ------------------------------------------------------------------
    async def recall(
        self,
        query: str,
        top_k: int = 10,
        neuron_type: str | None = None,
        layer: str | None = None,
        tags: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve memories using hybrid search (semantic + FTS + spreading activation)."""
        # 0. Piggyback maintenance
        await self._maybe_maintenance()

        # 1. Embed query
        query_vec = await self._embedder.embed(query)

        # 2. Semantic search (synchronous — pure numpy)
        sem_results = self._semantic_search.search(query_vec, top_k=top_k * 3, min_score=0.15)
        semantic_scores: dict[str, float] = {nid: score for nid, score in sem_results}

        # 3. FTS search
        fts_neurons = await self._neuron_repo.fts_search(query, limit=top_k * 2)
        fts_scores: dict[str, float] = {}
        for i, n in enumerate(fts_neurons):
            fts_scores[n.id] = 1.0 - (i / max(len(fts_neurons), 1))

        # 4. Spreading activation from top semantic hits
        seed_ids = [nid for nid, _ in sem_results[:5]]
        seed_acts = [score for _, score in sem_results[:5]]
        activation_map: dict[str, float] = {}
        if seed_ids:
            activation_map = await self._spreading.activate(seed_ids, seed_acts)

        # 5. Gather all candidate neurons (single batch query)
        candidate_ids = set(semantic_scores.keys()) | set(fts_scores.keys()) | set(activation_map.keys())

        # Also include working memory neurons for context
        wm_ids = await self._working_memory.get_active_ids()
        candidate_ids.update(wm_ids)

        all_neurons = await self._neuron_repo.get_by_ids(list(candidate_ids))
        candidates: dict[str, Neuron] = {}
        for nid, neuron in all_neurons.items():
            if neuron_type and neuron.neuron_type.value != neuron_type:
                continue
            if layer and neuron.layer.value != layer:
                continue
            if tags and not set(tags).intersection(neuron.tags):
                continue
            candidates[nid] = neuron

        # 6. Hybrid ranking (synchronous — pure computation)
        ranked = self._ranker.rank(
            candidates=candidates,
            semantic_scores=semantic_scores,
            fts_scores=fts_scores,
            activation_scores=activation_map,
            top_k=top_k,
        )

        # 7. Reinforce accessed neurons and apply Hebbian learning
        accessed_ids = [r.neuron.id for r in ranked]
        await self._reinforce_and_hebbian(accessed_ids)

        logger.info(
            "Recall '%s': %d candidates -> %d results",
            query[:50], len(candidates), len(ranked),
        )

        return [r.to_dict() for r in ranked]

    # ------------------------------------------------------------------
    # reflect: maintenance pass (decay, promotion, consolidation)
    # ------------------------------------------------------------------
    async def reflect(self) -> dict[str, Any]:
        """Run a full maintenance cycle."""
        stats: dict[str, Any] = {}
        now = datetime.now(timezone.utc)

        # 1. Decay
        all_neurons = await self._neuron_repo.get_all_for_decay()
        updates, prune_ids = self._decay.batch_decay(all_neurons, now)

        if updates:
            await self._neuron_repo.update_decay_batch(updates)
        stats["decayed"] = len(updates)

        # 2. Prune
        pruned_count = 0
        for nid in prune_ids:
            neuron = await self._neuron_repo.get_by_id(nid)
            if neuron and neuron.file_path:
                await self._file_store.delete(neuron.file_path)
            await self._neuron_repo.delete(nid)
            await self._semantic_search.remove_from_index(nid)
            await self._working_memory.remove(nid)
            pruned_count += 1
        stats["pruned"] = pruned_count

        # 3. Working memory cleanup → short_term
        expired_ids = await self._working_memory.cleanup_expired()
        for nid in expired_ids:
            await self._neuron_repo.update_layer(nid, MemoryLayer.SHORT_TERM)
        stats["working_to_short_term"] = len(expired_ids)

        # 4. Short-term → long-term promotion
        promotion_ids = await self._working_memory.get_promotion_candidates()
        for nid in promotion_ids:
            await self._neuron_repo.update_layer(nid, MemoryLayer.LONG_TERM)
        stats["short_term_to_long_term"] = len(promotion_ids)

        # 5. Consolidation (cluster long-term non-schema neurons)
        long_term_neurons = await self._neuron_repo.get_by_type_and_layer(
            NeuronType.SEMANTIC, MemoryLayer.LONG_TERM, limit=500,
        )
        long_term_neurons += await self._neuron_repo.get_by_type_and_layer(
            NeuronType.EPISODIC, MemoryLayer.LONG_TERM, limit=500,
        )

        schemas_created = 0
        if len(long_term_neurons) >= self._config.consolidation_config.min_cluster_size:
            embeddings: dict[str, np.ndarray] = {}
            for n in long_term_neurons:
                if n.embedding:
                    vec = np.frombuffer(n.embedding, dtype=np.float32).copy()
                    if vec.shape[0] == self._config.embedding_dimension:
                        embeddings[n.id] = vec

            clusters = self._consolidation.find_clusters(long_term_neurons, embeddings)
            for cluster in clusters:
                cluster_embeddings = [
                    embeddings[n.id] for n in cluster if n.id in embeddings
                ]
                if len(cluster_embeddings) != len(cluster):
                    continue

                schema, schema_emb, synapses = self._consolidation.create_schema_neuron(
                    cluster, cluster_embeddings,
                )
                schema.embedding = schema_emb.tobytes()
                schema.embedding_model = self._embedder.model_name

                await self._neuron_repo.insert(schema)
                await self._semantic_search.add_to_index(schema.id, schema_emb)
                for syn in synapses:
                    await self._synapse_repo.upsert(syn)
                schemas_created += 1

        stats["schemas_created"] = schemas_created

        # 6. Synapse decay
        decayed_synapses = await self._synapse_repo.decay_all_weights()
        stats["synapse_decayed"] = decayed_synapses

        logger.info("Reflect complete: %s", stats)
        return stats

    # ------------------------------------------------------------------
    # forget: manual deletion
    # ------------------------------------------------------------------
    async def forget(
        self,
        neuron_id: str | None = None,
        query: str | None = None,
        top_k: int = 5,
        confirm: bool = False,
    ) -> dict[str, Any]:
        """Delete memories manually."""
        # Mode 1: delete by ID
        if neuron_id:
            neuron = await self._neuron_repo.get_by_id(neuron_id)
            if neuron is None:
                return {"error": f"Neuron {neuron_id} not found"}

            if not confirm:
                return {
                    "preview": True,
                    "message": "Set confirm=true to delete this memory",
                    "neuron": neuron.to_dict(),
                }

            if neuron.file_path:
                await self._file_store.delete(neuron.file_path)
            await self._semantic_search.remove_from_index(neuron_id)
            await self._working_memory.remove(neuron_id)
            await self._neuron_repo.delete(neuron_id)

            return {"deleted": True, "neuron_id": neuron_id}

        # Mode 2: delete by query
        if query:
            results = await self.recall(query, top_k=top_k)
            if not results:
                return {"message": "No matching memories found"}

            if not confirm:
                return {
                    "preview": True,
                    "message": f"Found {len(results)} matches. Set confirm=true to delete them.",
                    "matches": results,
                }

            deleted_ids = []
            for r in results:
                nid = r["id"]
                neuron = await self._neuron_repo.get_by_id(nid)
                if neuron and neuron.file_path:
                    await self._file_store.delete(neuron.file_path)
                await self._semantic_search.remove_from_index(nid)
                await self._working_memory.remove(nid)
                await self._neuron_repo.delete(nid)
                deleted_ids.append(nid)

            return {"deleted": True, "count": len(deleted_ids), "neuron_ids": deleted_ids}

        return {"error": "Provide either neuron_id or query"}

    # ------------------------------------------------------------------
    # associate: manual synapse creation
    # ------------------------------------------------------------------
    async def associate(
        self,
        neuron_id_a: str,
        neuron_id_b: str,
        synapse_type: str = "reference",
        weight: float = 0.7,
    ) -> dict[str, Any]:
        """Create a manual connection (synapse) between two neurons."""
        a = await self._neuron_repo.get_by_id(neuron_id_a)
        b = await self._neuron_repo.get_by_id(neuron_id_b)
        if a is None:
            return {"error": f"Neuron {neuron_id_a} not found"}
        if b is None:
            return {"error": f"Neuron {neuron_id_b} not found"}

        stype = SynapseType(synapse_type)
        now = datetime.now(timezone.utc)
        synapse = Synapse(
            id=str(uuid.uuid4()),
            pre_neuron_id=neuron_id_a,
            post_neuron_id=neuron_id_b,
            synapse_type=stype,
            weight=max(0.0, min(1.0, weight)),
            created_at=now,
            last_activated=now,
        )
        await self._synapse_repo.upsert(synapse)

        logger.info(
            "Associated: %s -[%s]-> %s (weight=%.2f)",
            neuron_id_a[:8], synapse_type, neuron_id_b[:8], weight,
        )

        return {
            "synapse_id": synapse.id,
            "pre_neuron_id": neuron_id_a,
            "post_neuron_id": neuron_id_b,
            "synapse_type": synapse_type,
            "weight": synapse.weight,
        }

    # ------------------------------------------------------------------
    # status: system statistics
    # ------------------------------------------------------------------
    async def status(self) -> dict[str, Any]:
        """Get comprehensive system statistics."""
        layer_counts = await self._neuron_repo.count_by_layer()
        wm_stats = await self._working_memory.stats()
        file_stats = await self._file_store.stats()

        total_neurons = sum(layer_counts.values())

        async with self._db.read() as conn:
            cursor = await conn.execute(
                "SELECT COUNT(*) as c FROM synapses"
            )
            syn_count = (await cursor.fetchone())["c"]
            cursor = await conn.execute(
                "SELECT synapse_type, COUNT(*) as c FROM synapses GROUP BY synapse_type"
            )
            syn_by_type = await cursor.fetchall()

        return {
            "neurons": {
                "total": total_neurons,
                "by_layer": layer_counts,
            },
            "synapses": {
                "total": syn_count,
                "by_type": {r["synapse_type"]: r["c"] for r in syn_by_type},
            },
            "working_memory": wm_stats,
            "semantic_index_size": self._semantic_search.index_size,
            "embedding_model": self._embedder.model_name,
            "embedding_dimension": self._config.embedding_dimension,
            "file_storage": file_stats,
        }

    # ------------------------------------------------------------------
    # list_graph: all neurons + synapses for visualization
    # ------------------------------------------------------------------
    async def list_graph(self) -> dict[str, Any]:
        """Return all neurons (without embeddings) and all synapses."""
        all_neurons = await self._neuron_repo.get_all_for_decay()
        all_synapses = await self._synapse_repo.get_all()

        neurons_out = []
        for n in all_neurons:
            d = n.to_dict()
            d.pop("embedding", None)
            d.pop("embedding_model", None)
            neurons_out.append(d)

        synapses_out = [
            {
                "id": s.id,
                "pre_neuron_id": s.pre_neuron_id,
                "post_neuron_id": s.post_neuron_id,
                "synapse_type": s.synapse_type.value,
                "weight": s.weight,
                "activation_count": s.activation_count,
                "created_at": s.created_at.isoformat(),
                "last_activated": s.last_activated.isoformat(),
            }
            for s in all_synapses
        ]

        return {"neurons": neurons_out, "synapses": synapses_out}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    async def _reinforce_and_hebbian(self, accessed_ids: list[str]) -> None:
        """Reinforce accessed neurons and strengthen co-activated synapses."""
        if not accessed_ids:
            return

        neurons = await self._neuron_repo.get_by_ids(accessed_ids)
        for nid, neuron in neurons.items():
            old_importance = neuron.importance
            self._decay.reinforce(neuron)
            await self._neuron_repo.update_strength(
                nid, neuron.strength, neuron.stability,
            )
            # Persist importance if self-adaptation changed it
            if neuron.importance != old_importance:
                await self._neuron_repo.update_importance(nid, neuron.importance)

        # Hebbian: strengthen synapses between co-accessed neurons
        id_set = set(accessed_ids)
        for nid in accessed_ids:
            synapses = await self._synapse_repo.get_connections(nid, min_weight=0.0, limit=100)
            for syn in synapses:
                other = (
                    syn.post_neuron_id
                    if syn.pre_neuron_id == nid
                    else syn.pre_neuron_id
                )
                if other in id_set:
                    syn.hebbian_update()
                    await self._synapse_repo.upsert(syn)

    async def _maybe_maintenance(self) -> None:
        """Lightweight piggyback maintenance — runs at most once every 10 minutes.

        Performs only fast operations: decay + prune + working memory cleanup.
        Skips consolidation and synapse decay to keep it quick.
        """
        now = datetime.now(timezone.utc)
        elapsed = (now - self._last_maintenance).total_seconds()
        if elapsed < 600:  # 10 minutes
            return

        self._last_maintenance = now
        logger.info("Piggyback maintenance triggered (%.0fs since last)", elapsed)

        # Fast decay + prune
        all_neurons = await self._neuron_repo.get_all_for_decay()
        updates, prune_ids = self._decay.batch_decay(all_neurons, now)
        if updates:
            await self._neuron_repo.update_decay_batch(updates)
        for nid in prune_ids:
            neuron = await self._neuron_repo.get_by_id(nid)
            if neuron and neuron.file_path:
                await self._file_store.delete(neuron.file_path)
            await self._neuron_repo.delete(nid)
            await self._semantic_search.remove_from_index(nid)
            await self._working_memory.remove(nid)

        # Working memory cleanup
        expired_ids = await self._working_memory.cleanup_expired()
        for nid in expired_ids:
            await self._neuron_repo.update_layer(nid, MemoryLayer.SHORT_TERM)

        logger.info(
            "Piggyback maintenance done: decayed=%d, pruned=%d, wm_expired=%d",
            len(updates), len(prune_ids), len(expired_ids),
        )

    async def _dedup_check(self, embedding_vec: np.ndarray, threshold: float = 0.9) -> str | None:
        """Check if a very similar memory already exists.

        Returns the neuron_id of the duplicate if found, else None.
        """
        hits = self._semantic_search.search(embedding_vec, top_k=1, min_score=threshold)
        if hits:
            return hits[0][0]
        return None

    async def close(self) -> None:
        """Shut down the memory store."""
        await self._db.close()
        logger.info("MemoryStore closed.")
