"""Async database repositories for neurons and synapses."""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from typing import Any

from neural_memory.db.connection import DatabaseConnection
from neural_memory.core.neuron import Neuron, NeuronType, MemoryLayer
from neural_memory.core.synapse import Synapse, SynapseType


class NeuronRepository:
    """Async data access layer for neurons."""

    def __init__(self, db: DatabaseConnection):
        self._db = db

    def _row_to_neuron(self, row: Any) -> Neuron:
        """Convert a database row to a Neuron object."""
        return Neuron(
            id=row["id"],
            content=row["content"],
            summary=row["summary"],
            neuron_type=NeuronType(row["neuron_type"]),
            layer=MemoryLayer(row["layer"]),
            strength=row["strength"],
            stability=row["stability"],
            importance=row["importance"],
            emotional_valence=row["emotional_valence"] or 0.0,
            emotional_arousal=row["emotional_arousal"] or 0.0,
            access_count=row["access_count"] or 0,
            embedding=row["embedding"],
            embedding_model=row["embedding_model"],
            tags=json.loads(row["tags"]) if row["tags"] else [],
            source=row["source"] or "manual",
            file_path=row["file_path"],
            file_type=row["file_type"],
            file_hash=row["file_hash"],
            created_at=datetime.fromisoformat(row["created_at"]),
            last_accessed=datetime.fromisoformat(row["last_accessed"]),
            last_decayed=datetime.fromisoformat(row["last_decayed"]),
        )

    async def insert(self, neuron: Neuron) -> None:
        """Insert a new neuron into the database."""
        async with self._db.transaction() as conn:
            await conn.execute(
                """INSERT INTO neurons (
                    id, content, summary, neuron_type, layer,
                    strength, stability, importance,
                    emotional_valence, emotional_arousal,
                    access_count, embedding, embedding_model,
                    tags, source,
                    file_path, file_type, file_hash,
                    created_at, last_accessed, last_decayed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    neuron.id, neuron.content, neuron.summary,
                    neuron.neuron_type.value, neuron.layer.value,
                    neuron.strength, neuron.stability, neuron.importance,
                    neuron.emotional_valence, neuron.emotional_arousal,
                    neuron.access_count, neuron.embedding, neuron.embedding_model,
                    neuron.tags_json, neuron.source,
                    neuron.file_path, neuron.file_type, neuron.file_hash,
                    neuron.created_at.isoformat(), neuron.last_accessed.isoformat(),
                    neuron.last_decayed.isoformat(),
                ),
            )

    async def get_by_id(self, neuron_id: str) -> Neuron | None:
        """Fetch a neuron by ID."""
        async with self._db.read() as conn:
            cursor = await conn.execute(
                "SELECT * FROM neurons WHERE id = ?", (neuron_id,)
            )
            row = await cursor.fetchone()
            if row is None:
                return None
            return self._row_to_neuron(row)

    async def get_by_ids(self, neuron_ids: list[str]) -> dict[str, Neuron]:
        """Fetch multiple neurons by ID in a single query."""
        if not neuron_ids:
            return {}
        placeholders = ",".join("?" for _ in neuron_ids)
        async with self._db.read() as conn:
            cursor = await conn.execute(
                f"SELECT * FROM neurons WHERE id IN ({placeholders})",
                tuple(neuron_ids),
            )
            rows = await cursor.fetchall()
            return {r["id"]: self._row_to_neuron(r) for r in rows}

    async def update_strength(
        self, neuron_id: str, strength: float, stability: float | None = None
    ) -> None:
        """Update a neuron's strength and optionally stability."""
        async with self._db.transaction() as conn:
            if stability is not None:
                await conn.execute(
                    """UPDATE neurons SET strength = ?, stability = ?,
                       last_accessed = ?, access_count = access_count + 1
                       WHERE id = ?""",
                    (strength, stability, datetime.now(timezone.utc).isoformat(), neuron_id),
                )
            else:
                await conn.execute(
                    "UPDATE neurons SET strength = ? WHERE id = ?",
                    (strength, neuron_id),
                )

    async def update_layer(self, neuron_id: str, layer: MemoryLayer) -> None:
        """Move a neuron to a different memory layer."""
        async with self._db.transaction() as conn:
            await conn.execute(
                "UPDATE neurons SET layer = ? WHERE id = ?",
                (layer.value, neuron_id),
            )

    async def update_decay_batch(self, updates: list[tuple[float, str, str]]) -> None:
        """Batch update strengths and last_decayed timestamps.

        Args:
            updates: List of (new_strength, last_decayed_iso, neuron_id) tuples.
        """
        async with self._db.transaction() as conn:
            await conn.executemany(
                "UPDATE neurons SET strength = ?, last_decayed = ? WHERE id = ?",
                updates,
            )

    async def get_all_for_decay(self) -> list[Neuron]:
        """Get all neurons that need decay processing."""
        async with self._db.read() as conn:
            cursor = await conn.execute(
                """SELECT * FROM neurons
                   WHERE strength > 0.0
                   ORDER BY last_decayed ASC"""
            )
            rows = await cursor.fetchall()
            return [self._row_to_neuron(r) for r in rows]

    async def get_by_layer(self, layer: MemoryLayer, limit: int = 100) -> list[Neuron]:
        """Get neurons by memory layer."""
        async with self._db.read() as conn:
            cursor = await conn.execute(
                "SELECT * FROM neurons WHERE layer = ? ORDER BY created_at DESC LIMIT ?",
                (layer.value, limit),
            )
            rows = await cursor.fetchall()
            return [self._row_to_neuron(r) for r in rows]

    async def get_temporal_neighbors(
        self, reference_time: datetime, window_seconds: int = 1800, limit: int = 5
    ) -> list[Neuron]:
        """Get neurons created within a time window of a reference time."""
        start = (reference_time - timedelta(seconds=window_seconds)).isoformat()
        end = (reference_time + timedelta(seconds=window_seconds)).isoformat()
        async with self._db.read() as conn:
            cursor = await conn.execute(
                """SELECT * FROM neurons
                   WHERE created_at BETWEEN ? AND ?
                   ORDER BY created_at DESC LIMIT ?""",
                (start, end, limit),
            )
            rows = await cursor.fetchall()
            return [self._row_to_neuron(r) for r in rows]

    async def get_embedding_batch(self) -> list[tuple[str, bytes]]:
        """Get all (id, embedding) pairs for building search index."""
        async with self._db.read() as conn:
            cursor = await conn.execute(
                "SELECT id, embedding FROM neurons WHERE embedding IS NOT NULL"
            )
            rows = await cursor.fetchall()
            return [(r["id"], r["embedding"]) for r in rows]

    @staticmethod
    def _sanitize_fts_query(query: str) -> str:
        """Sanitize a query string for FTS5 MATCH.

        FTS5 treats hyphens as column selectors (``col:term``) and other
        punctuation as special syntax.  We quote each token individually so
        the query is interpreted as a simple term search.
        """
        import re
        # Split on whitespace, strip non-alphanumeric edges, quote each token
        tokens: list[str] = []
        for token in query.split():
            # Remove characters that are problematic inside FTS5 quoted strings
            cleaned = re.sub(r'[^\w]', ' ', token).strip()
            if cleaned:
                # Double-quote each token so FTS5 treats it literally
                tokens.append(f'"{cleaned}"')
        return " ".join(tokens) if tokens else '""'

    async def fts_search(self, query: str, limit: int = 20) -> list[Neuron]:
        """Full-text search using FTS5."""
        safe_query = self._sanitize_fts_query(query)
        async with self._db.read() as conn:
            cursor = await conn.execute(
                """SELECT neurons.* FROM neurons_fts
                   JOIN neurons ON neurons.rowid = neurons_fts.rowid
                   WHERE neurons_fts MATCH ?
                   ORDER BY rank LIMIT ?""",
                (safe_query, limit),
            )
            rows = await cursor.fetchall()
            return [self._row_to_neuron(r) for r in rows]

    async def get_pruning_candidates(self, strength_threshold: float = 0.05, importance_threshold: float = 0.3) -> list[Neuron]:
        """Get neurons that are candidates for pruning (weak and unimportant)."""
        async with self._db.read() as conn:
            cursor = await conn.execute(
                """SELECT * FROM neurons
                   WHERE strength < ? AND importance < ?
                   AND emotional_arousal < 0.5""",
                (strength_threshold, importance_threshold),
            )
            rows = await cursor.fetchall()
            return [self._row_to_neuron(r) for r in rows]

    async def delete(self, neuron_id: str) -> bool:
        """Delete a neuron (synapses cascade via FK)."""
        async with self._db.transaction() as conn:
            cursor = await conn.execute("DELETE FROM neurons WHERE id = ?", (neuron_id,))
            return cursor.rowcount > 0

    async def count_by_layer(self) -> dict[str, int]:
        """Count neurons grouped by layer."""
        async with self._db.read() as conn:
            cursor = await conn.execute(
                "SELECT layer, COUNT(*) as c FROM neurons GROUP BY layer"
            )
            rows = await cursor.fetchall()
            return {r["layer"]: r["c"] for r in rows}

    async def get_by_type_and_layer(
        self, neuron_type: NeuronType, layer: MemoryLayer, limit: int = 500
    ) -> list[Neuron]:
        """Get neurons filtered by type and layer."""
        async with self._db.read() as conn:
            cursor = await conn.execute(
                """SELECT * FROM neurons
                   WHERE neuron_type = ? AND layer = ?
                   ORDER BY created_at DESC LIMIT ?""",
                (neuron_type.value, layer.value, limit),
            )
            rows = await cursor.fetchall()
            return [self._row_to_neuron(r) for r in rows]


class SynapseRepository:
    """Async data access layer for synapses."""

    def __init__(self, db: DatabaseConnection):
        self._db = db

    def _row_to_synapse(self, row: Any) -> Synapse:
        """Convert a database row to a Synapse object."""
        return Synapse(
            id=row["id"],
            pre_neuron_id=row["pre_neuron_id"],
            post_neuron_id=row["post_neuron_id"],
            synapse_type=SynapseType(row["synapse_type"]),
            weight=row["weight"],
            activation_count=row["activation_count"] or 0,
            created_at=datetime.fromisoformat(row["created_at"]),
            last_activated=datetime.fromisoformat(row["last_activated"]),
        )

    async def upsert(self, synapse: Synapse) -> None:
        """Insert or update a synapse."""
        async with self._db.transaction() as conn:
            await conn.execute(
                """INSERT INTO synapses (
                    id, pre_neuron_id, post_neuron_id, synapse_type,
                    weight, activation_count, created_at, last_activated
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    weight = excluded.weight,
                    activation_count = excluded.activation_count,
                    last_activated = excluded.last_activated""",
                (
                    synapse.id, synapse.pre_neuron_id, synapse.post_neuron_id,
                    synapse.synapse_type.value, synapse.weight,
                    synapse.activation_count,
                    synapse.created_at.isoformat(), synapse.last_activated.isoformat(),
                ),
            )

    async def get_connections(
        self, neuron_id: str, min_weight: float = 0.0, limit: int = 50
    ) -> list[Synapse]:
        """Get all synapses connected to a neuron (either direction)."""
        async with self._db.read() as conn:
            cursor = await conn.execute(
                """SELECT * FROM synapses
                   WHERE (pre_neuron_id = ? OR post_neuron_id = ?)
                   AND weight >= ?
                   ORDER BY weight DESC LIMIT ?""",
                (neuron_id, neuron_id, min_weight, limit),
            )
            rows = await cursor.fetchall()
            return [self._row_to_synapse(r) for r in rows]

    async def get_outgoing(self, neuron_id: str, min_weight: float = 0.1) -> list[Synapse]:
        """Get outgoing synapses from a neuron."""
        async with self._db.read() as conn:
            cursor = await conn.execute(
                """SELECT * FROM synapses
                   WHERE pre_neuron_id = ? AND weight >= ?
                   ORDER BY weight DESC""",
                (neuron_id, min_weight),
            )
            rows = await cursor.fetchall()
            return [self._row_to_synapse(r) for r in rows]

    async def get_all_connections_for_neuron(self, neuron_id: str) -> list[Synapse]:
        """Get ALL synapses (both directions) for a neuron."""
        async with self._db.read() as conn:
            cursor = await conn.execute(
                """SELECT * FROM synapses
                   WHERE pre_neuron_id = ? OR post_neuron_id = ?""",
                (neuron_id, neuron_id),
            )
            rows = await cursor.fetchall()
            return [self._row_to_synapse(r) for r in rows]

    async def delete_for_neuron(self, neuron_id: str) -> int:
        """Delete all synapses connected to a neuron."""
        async with self._db.transaction() as conn:
            cursor = await conn.execute(
                "DELETE FROM synapses WHERE pre_neuron_id = ? OR post_neuron_id = ?",
                (neuron_id, neuron_id),
            )
            return cursor.rowcount

    async def decay_all_weights(self, factor: float = 0.995) -> int:
        """Apply decay to all synapse weights."""
        async with self._db.transaction() as conn:
            cursor = await conn.execute(
                "UPDATE synapses SET weight = weight * ? WHERE weight > 0.01",
                (factor,),
            )
            # Clean up dead synapses
            await conn.execute("DELETE FROM synapses WHERE weight <= 0.01")
            return cursor.rowcount
