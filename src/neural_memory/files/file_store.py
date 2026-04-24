"""File storage with type-based directory organization and SHA-256 deduplication."""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# File type → (subdirectory, default extension)
FILE_TYPE_MAP: dict[str, dict[str, str]] = {
    "code/python":     {"dir": "code/python",     "ext": ".py"},
    "code/javascript": {"dir": "code/javascript", "ext": ".js"},
    "code/typescript": {"dir": "code/typescript", "ext": ".ts"},
    "code/sql":        {"dir": "code/sql",        "ext": ".sql"},
    "code/rust":       {"dir": "code/rust",       "ext": ".rs"},
    "code/go":         {"dir": "code/go",         "ext": ".go"},
    "code/java":       {"dir": "code/java",       "ext": ".java"},
    "code/cpp":        {"dir": "code/cpp",        "ext": ".cpp"},
    "code/c":          {"dir": "code/c",          "ext": ".c"},
    "code/shell":      {"dir": "code/shell",      "ext": ".sh"},
    "code/generic":    {"dir": "code/generic",    "ext": ".txt"},
    "document/markdown": {"dir": "documents",     "ext": ".md"},
    "document/text":     {"dir": "documents",     "ext": ".txt"},
    "document/html":     {"dir": "documents",     "ext": ".html"},
    "data/json":         {"dir": "data",          "ext": ".json"},
    "data/csv":          {"dir": "data",          "ext": ".csv"},
    "data/yaml":         {"dir": "data",          "ext": ".yaml"},
    "data/xml":          {"dir": "data",          "ext": ".xml"},
}

# Extension → file_type reverse lookup
EXT_TO_FILE_TYPE: dict[str, str] = {
    ".py": "code/python", ".pyw": "code/python",
    ".js": "code/javascript", ".mjs": "code/javascript", ".cjs": "code/javascript",
    ".ts": "code/typescript", ".tsx": "code/typescript", ".jsx": "code/javascript",
    ".sql": "code/sql",
    ".rs": "code/rust",
    ".go": "code/go",
    ".java": "code/java",
    ".cpp": "code/cpp", ".cc": "code/cpp", ".cxx": "code/cpp", ".h": "code/c", ".hpp": "code/cpp",
    ".c": "code/c",
    ".sh": "code/shell", ".bash": "code/shell", ".zsh": "code/shell",
    ".md": "document/markdown",
    ".txt": "document/text",
    ".html": "document/html", ".htm": "document/html",
    ".json": "data/json",
    ".csv": "data/csv",
    ".yaml": "data/yaml", ".yml": "data/yaml",
    ".xml": "data/xml",
}


@dataclass
class StoredFile:
    """Metadata about a stored file."""
    relative_path: str   # Relative to storage/files/
    file_type: str
    file_hash: str
    size_bytes: int


class FileStore:
    """Manages files on disk with type-based organization and content-addressable dedup."""

    def __init__(self, base_dir: Path):
        self._base_dir = base_dir
        self._base_dir.mkdir(parents=True, exist_ok=True)

    async def store(
        self,
        content: str | bytes,
        file_type: str | None = None,
        original_extension: str | None = None,
        filename_hint: str | None = None,
    ) -> StoredFile:
        """Store content to disk.

        Deduplication: if identical content already exists (by SHA-256),
        return the existing path without writing again.
        """
        if isinstance(content, str):
            content_bytes = content.encode("utf-8")
        else:
            content_bytes = content

        file_hash = hashlib.sha256(content_bytes).hexdigest()

        if file_type is None and original_extension:
            file_type = EXT_TO_FILE_TYPE.get(original_extension, "document/text")
        elif file_type is None:
            file_type = "document/text"

        mapping = FILE_TYPE_MAP.get(file_type, FILE_TYPE_MAP["document/text"])
        ext = original_extension or mapping["ext"]
        subdir = mapping["dir"]

        hash_prefix = file_hash[:4]
        dir_path = self._base_dir / subdir / hash_prefix
        dir_path.mkdir(parents=True, exist_ok=True)

        file_name = f"{file_hash[:16]}{ext}"
        full_path = dir_path / file_name
        relative_path = str(full_path.relative_to(self._base_dir))

        if not full_path.exists():
            await asyncio.to_thread(full_path.write_bytes, content_bytes)

        return StoredFile(
            relative_path=relative_path,
            file_type=file_type,
            file_hash=file_hash,
            size_bytes=len(content_bytes),
        )

    async def read(self, relative_path: str) -> bytes:
        """Read a stored file by its relative path."""
        full_path = self._base_dir / relative_path
        if not full_path.exists():
            raise FileNotFoundError(f"Stored file not found: {relative_path}")
        return await asyncio.to_thread(full_path.read_bytes)

    async def delete(self, relative_path: str) -> bool:
        """Delete a stored file."""
        full_path = self._base_dir / relative_path
        if full_path.exists():
            await asyncio.to_thread(full_path.unlink)
            return True
        return False

    async def stats(self) -> dict[str, Any]:
        """Get storage statistics."""
        def _compute_stats() -> dict[str, Any]:
            total_files = 0
            total_bytes = 0
            by_type: dict[str, int] = {}

            for f in self._base_dir.rglob("*"):
                if f.is_file():
                    total_files += 1
                    total_bytes += f.stat().st_size
                    suffix = f.suffix
                    by_type[suffix] = by_type.get(suffix, 0) + 1

            return {
                "total_files": total_files,
                "total_bytes": total_bytes,
                "total_mb": round(total_bytes / (1024 * 1024), 2),
                "by_extension": by_type,
            }

        return await asyncio.to_thread(_compute_stats)
