"""Async SQLite connection management with WAL mode and context managers."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import aiosqlite

logger = logging.getLogger(__name__)


class DatabaseConnection:
    """Async SQLite connection manager.

    Uses WAL mode for concurrent reads and serialized writes.
    Must call `await initialize()` before use.
    """

    def __init__(self, db_path: Path):
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = asyncio.Lock()
        self._conn: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Open the connection and configure pragmas. Must be called once."""
        self._conn = await aiosqlite.connect(str(self._db_path))
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA foreign_keys=ON")
        await self._conn.execute("PRAGMA busy_timeout=5000")
        await self._conn.commit()

    @asynccontextmanager
    async def read(self) -> AsyncGenerator[aiosqlite.Connection, None]:
        """Read-only context (no lock needed with WAL)."""
        if self._conn is None:
            raise RuntimeError("DatabaseConnection not initialized. Call await initialize() first.")
        yield self._conn

    @asynccontextmanager
    async def transaction(self) -> AsyncGenerator[aiosqlite.Connection, None]:
        """Write transaction with async lock."""
        if self._conn is None:
            raise RuntimeError("DatabaseConnection not initialized. Call await initialize() first.")
        async with self._write_lock:
            try:
                yield self._conn
                await self._conn.commit()
            except Exception:
                await self._conn.rollback()
                raise

    async def initialize_schema(self, schema_path: Path) -> None:
        """Execute schema SQL file to create tables."""
        schema_sql = schema_path.read_text(encoding="utf-8")
        if self._conn is None:
            raise RuntimeError("DatabaseConnection not initialized. Call await initialize() first.")
        async with self._write_lock:
            await self._conn.executescript(schema_sql)
            await self._conn.commit()

    async def close(self) -> None:
        """Close the connection."""
        if self._conn is not None:
            await self._conn.close()
            self._conn = None
