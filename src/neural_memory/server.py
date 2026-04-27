"""MCP Server for Neural Memory - 6 tools for human-like memory management."""

import asyncio
import json
import logging
from typing import Sequence

from mcp.server.fastmcp import FastMCP

from neural_memory.config import load_config
from neural_memory.core.memory_store import MemoryStore

logger = logging.getLogger(__name__)

mcp = FastMCP("neural-memory", instructions="Human-like memory system for LLMs using neuron+synapse model")

# Global memory store, initialized lazily
_store: MemoryStore | None = None
_store_lock: asyncio.Lock | None = None


def _get_lock() -> asyncio.Lock:
    global _store_lock
    if _store_lock is None:
        _store_lock = asyncio.Lock()
    return _store_lock


def _parse_tags(raw: str | list | None) -> list[str] | None:
    """Flexibly parse tags from various input formats.

    Accepts:
      - JSON array string: '["python", "tutorial"]'
      - Comma-separated string: "python, tutorial"
      - list: ["python", "tutorial"]
      - None / empty string: returns None
    """
    if raw is None:
        return None
    if isinstance(raw, list):
        return [str(t).strip() for t in raw if str(t).strip()]
    raw = raw.strip()
    if not raw or raw == "[]":
        return None
    # Try JSON first
    if raw.startswith("["):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(t).strip() for t in parsed if str(t).strip()]
        except (json.JSONDecodeError, ValueError):
            pass
    # Fallback: comma-separated
    return [t.strip() for t in raw.split(",") if t.strip()]


async def _get_store() -> MemoryStore:
    global _store
    if _store is not None:
        return _store
    async with _get_lock():
        if _store is None:
            config = load_config()
            logging.basicConfig(level=getattr(logging, config.log_level, logging.INFO))
            _store = MemoryStore(config)
            await _store.initialize()
            logger.info("Memory store initialized successfully.")
    return _store


# ──────────────────────────────────────────────
# Tool 1: remember
# ──────────────────────────────────────────────
@mcp.tool(
    name="remember",
    description=(
        "Store a new memory. Automatically computes embeddings, creates temporal and "
        "semantic associations, and adds to working memory. Supports optional file "
        "attachment (code, documents, data)."
    ),
)
async def remember(
    content: str,
    neuron_type: str = "semantic",
    importance: float = 0.5,
    emotional_valence: float = 0.0,
    emotional_arousal: float = 0.0,
    tags: str = "[]",
    source: str = "manual",
    file_content: str | None = None,
    file_type: str | None = None,
    file_extension: str | None = None,
) -> str:
    """Store a new memory.

    Args:
        content: The memory content (text).
        neuron_type: One of: episodic, semantic, procedural. Default: semantic.
        importance: Importance level 0.0-1.0. Higher = more resistant to forgetting.
        emotional_valence: Emotional tone -1.0 (negative) to 1.0 (positive).
        emotional_arousal: Emotional intensity 0.0 (calm) to 1.0 (intense).
        tags: Comma-separated tags, e.g. "python, tutorial". Also accepts JSON array format.
        source: Source identifier (e.g. "user", "conversation", "manual").
        file_content: Optional file content to store alongside the memory.
        file_type: File type like "code/python", "data/json". Auto-detected if omitted.
        file_extension: File extension like ".py", ".json". Used for auto-detection.
    """
    store = await _get_store()
    tag_list = _parse_tags(tags) or []
    result = await store.remember(
        content=content,
        neuron_type=neuron_type,
        importance=importance,
        emotional_valence=emotional_valence,
        emotional_arousal=emotional_arousal,
        tags=tag_list,
        source=source,
        file_content=file_content,
        file_type=file_type,
        file_extension=file_extension,
    )
    return json.dumps(result, ensure_ascii=False, indent=2)


# ──────────────────────────────────────────────
# Tool 2: recall
# ──────────────────────────────────────────────
@mcp.tool(
    name="recall",
    description=(
        "Retrieve memories using hybrid search: semantic similarity, full-text search, "
        "and spreading activation through the association graph. Results are ranked by "
        "a weighted combination of signals. Accessing memories reinforces them."
    ),
)
async def recall(
    query: str,
    top_k: int = 10,
    neuron_type: str | None = None,
    layer: str | None = None,
    tags: str | None = None,
) -> str:
    """Search and retrieve memories.

    Args:
        query: Natural language search query.
        top_k: Maximum number of results to return. Default: 10.
        neuron_type: Filter by type: episodic, semantic, procedural, schema.
        layer: Filter by layer: working, short_term, long_term.
        tags: Comma-separated tags to filter by, e.g. "python, tutorial". Also accepts JSON array format.
    """
    store = await _get_store()
    tag_list = _parse_tags(tags)
    results = await store.recall(
        query=query,
        top_k=top_k,
        neuron_type=neuron_type,
        layer=layer,
        tags=tag_list,
    )
    return json.dumps(results, ensure_ascii=False, indent=2)


# ──────────────────────────────────────────────
# Tool 3: reflect
# ──────────────────────────────────────────────
@mcp.tool(
    name="reflect",
    description=(
        "Trigger a memory maintenance cycle: apply forgetting curve decay, prune weak "
        "memories, promote qualified short-term memories to long-term, consolidate "
        "similar long-term memories into schema neurons, and decay synapse weights."
    ),
)
async def reflect() -> str:
    """Run memory maintenance (decay, promotion, consolidation)."""
    store = await _get_store()
    stats = await store.reflect()
    return json.dumps(stats, ensure_ascii=False, indent=2)


# ──────────────────────────────────────────────
# Tool 4: forget
# ──────────────────────────────────────────────
@mcp.tool(
    name="forget",
    description=(
        "Manually delete memories. Can target a specific neuron by ID, or search by "
        "query to preview matches before deletion. Use confirm=true to execute deletion."
    ),
)
async def forget(
    neuron_id: str | None = None,
    query: str | None = None,
    top_k: int = 5,
    confirm: bool = False,
) -> str:
    """Delete memories.

    Args:
        neuron_id: Specific neuron ID to delete.
        query: Search query to find memories to delete.
        top_k: Number of matches to preview when using query. Default: 5.
        confirm: Set to true to actually delete. False shows preview only.
    """
    store = await _get_store()
    result = await store.forget(
        neuron_id=neuron_id,
        query=query,
        top_k=top_k,
        confirm=confirm,
    )
    return json.dumps(result, ensure_ascii=False, indent=2)


# ──────────────────────────────────────────────
# Tool 5: associate
# ──────────────────────────────────────────────
@mcp.tool(
    name="associate",
    description=(
        "Manually create a connection (synapse) between two neurons. Supports types: "
        "semantic, temporal, causal, hierarchical, reference."
    ),
)
async def associate(
    neuron_id_a: str,
    neuron_id_b: str,
    synapse_type: str = "reference",
    weight: float = 0.7,
) -> str:
    """Create a synapse between two neurons.

    Args:
        neuron_id_a: Source neuron ID.
        neuron_id_b: Target neuron ID.
        synapse_type: Connection type: semantic, temporal, causal, hierarchical, reference.
        weight: Connection strength 0.0-1.0. Default: 0.7.
    """
    store = await _get_store()
    result = await store.associate(
        neuron_id_a=neuron_id_a,
        neuron_id_b=neuron_id_b,
        synapse_type=synapse_type,
        weight=weight,
    )
    return json.dumps(result, ensure_ascii=False, indent=2)


# ──────────────────────────────────────────────
# Tool 6: memory_status
# ──────────────────────────────────────────────
@mcp.tool(
    name="memory_status",
    description=(
        "Get comprehensive statistics about the memory system: neuron counts by layer "
        "and type, synapse counts, working memory usage, semantic index size, and "
        "file storage stats."
    ),
)
async def memory_status() -> str:
    """Get memory system statistics."""
    store = await _get_store()
    result = await store.status()
    return json.dumps(result, ensure_ascii=False, indent=2)


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
async def main():
    """Run the Neural Memory MCP server via stdio."""
    logger.info("Neural Memory MCP Server starting...")
    # Start model loading in background so it's ready by first tool call
    # (loading takes ~20s, would timeout if done during handshake or tool call)
    asyncio.create_task(_get_store())
    await mcp.run_stdio_async()
