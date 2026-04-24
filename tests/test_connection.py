"""Tests for neural_memory.db.connection — Async SQLite connection management."""

from __future__ import annotations

from pathlib import Path

import pytest

from neural_memory.db.connection import DatabaseConnection

SCHEMA_PATH = Path(__file__).parent.parent / "src" / "neural_memory" / "db" / "schema.sql"


class TestDatabaseConnection:
    async def test_initialize_and_close(self, tmp_path):
        db = DatabaseConnection(tmp_path / "test.db")
        await db.initialize()
        assert db._conn is not None
        await db.close()
        assert db._conn is None

    async def test_read_before_initialize_raises(self, tmp_path):
        db = DatabaseConnection(tmp_path / "test.db")
        with pytest.raises(RuntimeError, match="not initialized"):
            async with db.read():
                pass

    async def test_transaction_before_initialize_raises(self, tmp_path):
        db = DatabaseConnection(tmp_path / "test.db")
        with pytest.raises(RuntimeError, match="not initialized"):
            async with db.transaction():
                pass

    async def test_initialize_schema(self, tmp_path):
        db = DatabaseConnection(tmp_path / "test.db")
        await db.initialize()
        await db.initialize_schema(SCHEMA_PATH)

        async with db.read() as conn:
            cursor = await conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='neurons'"
            )
            row = await cursor.fetchone()
            assert row is not None

        await db.close()

    async def test_wal_mode(self, tmp_path):
        db = DatabaseConnection(tmp_path / "test.db")
        await db.initialize()
        async with db.read() as conn:
            cursor = await conn.execute("PRAGMA journal_mode")
            row = await cursor.fetchone()
            assert row[0] == "wal"
        await db.close()

    async def test_foreign_keys_on(self, tmp_path):
        db = DatabaseConnection(tmp_path / "test.db")
        await db.initialize()
        async with db.read() as conn:
            cursor = await conn.execute("PRAGMA foreign_keys")
            row = await cursor.fetchone()
            assert row[0] == 1
        await db.close()

    async def test_read_context_yields_connection(self, tmp_path):
        db = DatabaseConnection(tmp_path / "test.db")
        await db.initialize()
        async with db.read() as conn:
            cursor = await conn.execute("SELECT 1")
            row = await cursor.fetchone()
            assert row[0] == 1
        await db.close()

    async def test_transaction_commits_on_success(self, tmp_path):
        db = DatabaseConnection(tmp_path / "test.db")
        await db.initialize()

        async with db.transaction() as conn:
            await conn.execute("CREATE TABLE test_tbl (id INTEGER PRIMARY KEY, val TEXT)")
            await conn.execute("INSERT INTO test_tbl (val) VALUES ('hello')")

        async with db.read() as conn:
            cursor = await conn.execute("SELECT val FROM test_tbl")
            row = await cursor.fetchone()
            assert row[0] == "hello"

        await db.close()

    async def test_transaction_rollback_on_error(self, tmp_path):
        db = DatabaseConnection(tmp_path / "test.db")
        await db.initialize()

        async with db.transaction() as conn:
            await conn.execute("CREATE TABLE test_tbl (id INTEGER PRIMARY KEY, val TEXT)")

        with pytest.raises(Exception):
            async with db.transaction() as conn:
                await conn.execute("INSERT INTO test_tbl (val) VALUES ('should_rollback')")
                raise ValueError("forced error")

        async with db.read() as conn:
            cursor = await conn.execute("SELECT COUNT(*) FROM test_tbl")
            count = (await cursor.fetchone())[0]
            assert count == 0

        await db.close()

    async def test_creates_parent_dirs(self, tmp_path):
        db_path = tmp_path / "nested" / "dirs" / "test.db"
        db = DatabaseConnection(db_path)
        assert db_path.parent.exists()
        await db.close()

    async def test_close_idempotent(self, tmp_path):
        db = DatabaseConnection(tmp_path / "test.db")
        await db.initialize()
        await db.close()
        await db.close()  # Should not raise
