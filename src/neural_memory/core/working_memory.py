"""Async working memory management - limited-capacity active buffer.

Like the prefrontal cortex, working memory holds a small number
of active memories. Overflow is handled by evicting the lowest
priority items and promoting them to short-term memory.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta

from neural_memory.config import WorkingMemoryConfig
from neural_memory.core.neuron import MemoryLayer
from neural_memory.db.connection import DatabaseConnection

logger = logging.getLogger(__name__)


class WorkingMemoryManager:
    """Manages the working memory buffer with capacity limits and TTL."""

    def __init__(self, db: DatabaseConnection, config: WorkingMemoryConfig | None = None):
        self._db = db
        self._cfg = config or WorkingMemoryConfig()

    async def add(self, neuron_id: str, priority: float = 0.5) -> None:
        """Add a neuron to working memory. Evicts lowest-priority if full."""
        now = datetime.now(timezone.utc)

        async with self._db.transaction() as conn:
            cursor = await conn.execute(
                "SELECT neuron_id FROM working_memory WHERE neuron_id = ?",
                (neuron_id,),
            )
            existing = await cursor.fetchone()

            if existing:
                await conn.execute(
                    "UPDATE working_memory SET priority = ?, entered_at = ? WHERE neuron_id = ?",
                    (priority, now.isoformat(), neuron_id),
                )
                return

            cursor = await conn.execute(
                "SELECT COUNT(*) as c FROM working_memory"
            )
            count = (await cursor.fetchone())["c"]

            if count >= self._cfg.capacity:
                await self._evict_lowest(conn)

            await conn.execute(
                "INSERT INTO working_memory (neuron_id, entered_at, priority) VALUES (?, ?, ?)",
                (neuron_id, now.isoformat(), priority),
            )

    async def remove(self, neuron_id: str) -> None:
        """Remove a neuron from working memory."""
        async with self._db.transaction() as conn:
            await conn.execute(
                "DELETE FROM working_memory WHERE neuron_id = ?",
                (neuron_id,),
            )

    async def get_active_ids(self) -> list[str]:
        """Get all neuron IDs currently in working memory."""
        async with self._db.read() as conn:
            cursor = await conn.execute(
                "SELECT neuron_id FROM working_memory ORDER BY priority DESC"
            )
            rows = await cursor.fetchall()
            return [r["neuron_id"] for r in rows]

    async def cleanup_expired(self) -> list[str]:
        """Remove expired entries (past TTL) and return their neuron IDs."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=self._cfg.ttl_hours)
        cutoff_iso = cutoff.isoformat()

        async with self._db.transaction() as conn:
            cursor = await conn.execute(
                "SELECT neuron_id FROM working_memory WHERE entered_at < ?",
                (cutoff_iso,),
            )
            expired = await cursor.fetchall()
            expired_ids = [r["neuron_id"] for r in expired]

            if expired_ids:
                placeholders = ",".join("?" * len(expired_ids))
                await conn.execute(
                    f"DELETE FROM working_memory WHERE neuron_id IN ({placeholders})",
                    expired_ids,
                )
                logger.info("Evicted %d expired entries from working memory", len(expired_ids))

            return expired_ids

    async def get_promotion_candidates(self) -> list[str]:
        """Get neuron IDs that should be promoted from short-term to long-term."""
        async with self._db.read() as conn:
            cursor = await conn.execute(
                """SELECT n.id FROM neurons n
                   WHERE n.layer = 'short_term'
                   AND (
                       n.strength > ? OR
                       n.importance > 0.4 OR
                       n.access_count >= ?
                   )""",
                (
                    self._cfg.promotion_strength_threshold,
                    self._cfg.short_term_access_threshold,
                ),
            )
            rows = await cursor.fetchall()
            return [r["id"] for r in rows]

    async def get_short_term_expiry_candidates(self) -> list[str]:
        """Get short-term neurons that have been there long enough to evaluate."""
        cutoff = datetime.now(timezone.utc) - timedelta(
            hours=self._cfg.short_term_to_long_term_hours
        )
        async with self._db.read() as conn:
            cursor = await conn.execute(
                """SELECT id FROM neurons
                   WHERE layer = 'short_term'
                   AND created_at < ?""",
                (cutoff.isoformat(),),
            )
            rows = await cursor.fetchall()
            return [r["id"] for r in rows]

    async def _evict_lowest(self, conn) -> None:
        """Evict the lowest-priority entry from working memory."""
        cursor = await conn.execute(
            "SELECT neuron_id FROM working_memory ORDER BY priority ASC LIMIT 1"
        )
        lowest = await cursor.fetchone()

        if lowest:
            neuron_id = lowest["neuron_id"]
            await conn.execute(
                "DELETE FROM working_memory WHERE neuron_id = ?",
                (neuron_id,),
            )
            await conn.execute(
                "UPDATE neurons SET layer = 'short_term' WHERE id = ? AND layer = 'working'",
                (neuron_id,),
            )
            logger.debug("Evicted neuron %s from working memory → short_term", neuron_id)

    async def stats(self) -> dict:
        """Get working memory statistics."""
        async with self._db.read() as conn:
            cursor = await conn.execute(
                "SELECT COUNT(*) as c FROM working_memory"
            )
            count = (await cursor.fetchone())["c"]

            return {
                "active_count": count,
                "capacity": self._cfg.capacity,
                "ttl_hours": self._cfg.ttl_hours,
            }
