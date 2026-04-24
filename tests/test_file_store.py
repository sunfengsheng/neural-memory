"""Tests for neural_memory.files.file_store — File storage with deduplication."""

from __future__ import annotations

import pytest

from neural_memory.files.file_store import (
    EXT_TO_FILE_TYPE,
    FILE_TYPE_MAP,
    FileStore,
    StoredFile,
)


class TestFileStore:
    @pytest.fixture
    def file_store(self, tmp_path):
        return FileStore(tmp_path / "files")

    async def test_store_string_content(self, file_store):
        result = await file_store.store("print('hello')", file_type="code/python")
        assert isinstance(result, StoredFile)
        assert result.file_type == "code/python"
        assert result.size_bytes == len("print('hello')".encode("utf-8"))
        assert len(result.file_hash) == 64  # SHA-256 hex
        assert result.relative_path.endswith(".py")

    async def test_store_bytes_content(self, file_store):
        data = b"\x00\x01\x02\x03binary"
        result = await file_store.store(data, file_type="document/text")
        assert result.size_bytes == len(data)

    async def test_store_deduplication(self, file_store):
        content = "same content twice"
        r1 = await file_store.store(content, file_type="document/text")
        r2 = await file_store.store(content, file_type="document/text")
        assert r1.file_hash == r2.file_hash
        assert r1.relative_path == r2.relative_path

    async def test_store_different_content_different_hash(self, file_store):
        r1 = await file_store.store("content A", file_type="document/text")
        r2 = await file_store.store("content B", file_type="document/text")
        assert r1.file_hash != r2.file_hash

    async def test_read_stored_file(self, file_store):
        content = "readable content"
        result = await file_store.store(content, file_type="document/text")
        data = await file_store.read(result.relative_path)
        assert data == content.encode("utf-8")

    async def test_read_nonexistent_raises(self, file_store):
        with pytest.raises(FileNotFoundError):
            await file_store.read("nonexistent/path.txt")

    async def test_delete_stored_file(self, file_store):
        result = await file_store.store("deleteable", file_type="document/text")
        deleted = await file_store.delete(result.relative_path)
        assert deleted is True
        with pytest.raises(FileNotFoundError):
            await file_store.read(result.relative_path)

    async def test_delete_nonexistent_returns_false(self, file_store):
        deleted = await file_store.delete("nonexistent/path.txt")
        assert deleted is False

    async def test_stats_empty(self, file_store):
        st = await file_store.stats()
        assert st["total_files"] == 0
        assert st["total_bytes"] == 0

    async def test_stats_after_store(self, file_store):
        await file_store.store("file one", file_type="code/python")
        await file_store.store("file two", file_type="data/json", original_extension=".json")
        st = await file_store.stats()
        assert st["total_files"] == 2
        assert st["total_bytes"] > 0
        assert isinstance(st["by_extension"], dict)

    async def test_type_inference_from_extension(self, file_store):
        result = await file_store.store("x = 1", original_extension=".py")
        assert result.file_type == "code/python"
        assert result.relative_path.endswith(".py")

    async def test_default_type_is_document_text(self, file_store):
        result = await file_store.store("plain text")
        assert result.file_type == "document/text"

    async def test_hash_prefix_in_path(self, file_store):
        result = await file_store.store("hash check", file_type="document/text")
        # Path should contain the first 4 chars of hash as subdirectory
        assert result.file_hash[:4] in result.relative_path

    async def test_file_name_contains_hash_prefix(self, file_store):
        result = await file_store.store("name check", file_type="code/python")
        # File name should start with first 16 chars of hash
        import os
        filename = os.path.basename(result.relative_path)
        assert filename.startswith(result.file_hash[:16])


class TestFileTypeMaps:
    def test_all_file_types_have_dir_and_ext(self):
        for ftype, mapping in FILE_TYPE_MAP.items():
            assert "dir" in mapping, f"{ftype} missing 'dir'"
            assert "ext" in mapping, f"{ftype} missing 'ext'"

    def test_ext_to_file_type_maps_back(self):
        # Every ext in EXT_TO_FILE_TYPE should map to a key in FILE_TYPE_MAP
        for ext, ftype in EXT_TO_FILE_TYPE.items():
            assert ftype in FILE_TYPE_MAP, f"{ext} -> {ftype} not in FILE_TYPE_MAP"

    def test_python_extension_maps_correctly(self):
        assert EXT_TO_FILE_TYPE[".py"] == "code/python"

    def test_json_extension_maps_correctly(self):
        assert EXT_TO_FILE_TYPE[".json"] == "data/json"
