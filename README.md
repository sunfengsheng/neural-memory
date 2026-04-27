# Neural Memory MCP Server

A [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server that gives LLMs a **human-like memory system**, built on a neuron + synapse model inspired by cognitive neuroscience.

Memories are stored as **neurons** that decay over time following the Ebbinghaus forgetting curve, form **synapses** (associations) with related memories, and get promoted through layers (working memory -> short-term -> long-term) based on access patterns. Retrieval uses a hybrid of semantic search, full-text search, and spreading activation through the association graph.

The entire I/O pipeline is **fully async** (`aiosqlite` + `asyncio`), ensuring the MCP event loop is never blocked — even during embedding computation (`asyncio.to_thread`) or file operations.

## Features

- **Neuron-based storage** — Each memory is a neuron with type (episodic/semantic/procedural), strength, stability, importance, and emotional markers
- **Ebbinghaus forgetting curve** — Memories decay exponentially; frequently accessed memories become more stable
- **Three-layer memory system** — Working memory (capacity-limited, 2h TTL) -> short-term -> long-term, with automatic promotion
- **Associative retrieval** — Spreading activation traverses the synapse graph to find indirectly related memories
- **Hybrid search** — Combines semantic similarity (384-dim embeddings), FTS5 full-text search, spreading activation, recency, and strength into a single ranked result
- **Memory consolidation** — Similar long-term memories are clustered into abstract "schema" neurons
- **Hebbian learning** — Co-accessed memories strengthen their connections ("neurons that fire together wire together")
- **File attachment** — Store code, documents, and data files alongside memories with SHA-256 deduplication
- **Multilingual** — Uses `paraphrase-multilingual-MiniLM-L12-v2` for Chinese/English/50+ language support
- **Fully async** — All DB, embedding, and file I/O is non-blocking; built on `aiosqlite` + `asyncio.to_thread`

## Requirements

- Python 3.10+
- ~470 MB disk space for the embedding model (downloaded on first run)

### Dependencies

| Package | Purpose |
|---------|---------|
| `mcp>=1.0.0` | Model Context Protocol SDK |
| `sentence-transformers>=2.2.0` | Local embedding model |
| `numpy>=1.24.0` | Vector operations |
| `pyyaml>=6.0` | Configuration |
| `aiosqlite>=0.19.0` | Async SQLite (WAL mode, non-blocking I/O) |

## Installation

### Claude Code Plugin (recommended)

One-command install as a Claude Code plugin with bundled model (zero network dependency):

**Prerequisites**: Python 3.10+, [Git LFS](https://git-lfs.com/) (for the 470MB embedding model)

```bash
# Install git-lfs if you haven't already
git lfs install

# Clone and install
git clone https://github.com/sunfengsheng/neural-memory.git
cd neural-memory

# Linux / macOS
bash install.sh

# Windows (PowerShell)
.\install.ps1
```

The installer will:
1. Install Python dependencies
2. Copy files to `~/.claude/plugins/marketplaces/neural-memory/`
3. Set up the cache directory with correct MCP config format
4. Verify the bundled embedding model is present

After installation, **restart Claude Code** — the MCP server will auto-start in new conversations.

### Manual install (standalone)

```bash
git clone https://github.com/sunfengsheng/neural-memory.git
cd neural-memory
pip install -e .
```

## Quick Start

### Run directly

```bash
python -m neural_memory
# or
neural-memory
```

### Configure in Claude Desktop

Add to your Claude Desktop MCP config (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "neural-memory": {
      "command": "python",
      "args": ["-m", "neural_memory"]
    }
  }
}
```

### Configure in Claude Code (manual)

```bash
claude mcp add neural-memory -- python -m neural_memory
```

## MCP Tools

### `remember`

Store a new memory. Automatically computes embeddings, creates temporal and semantic associations, and adds to working memory.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `content` | string | (required) | The memory text |
| `neuron_type` | string | `"semantic"` | `episodic`, `semantic`, or `procedural` |
| `importance` | float | `0.5` | 0.0-1.0, higher = more resistant to forgetting |
| `emotional_valence` | float | `0.0` | -1.0 (negative) to 1.0 (positive) |
| `emotional_arousal` | float | `0.0` | 0.0 (calm) to 1.0 (intense) |
| `tags` | string | `"[]"` | JSON array of tag strings |
| `source` | string | `"manual"` | Source identifier |
| `file_content` | string | null | Optional file content to store |
| `file_type` | string | null | e.g. `"code/python"`, `"data/json"` |
| `file_extension` | string | null | e.g. `".py"`, `".json"` |

### `recall`

Retrieve memories using hybrid search: semantic similarity + full-text search + spreading activation. Accessing memories reinforces them.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | string | (required) | Natural language search query |
| `top_k` | int | `10` | Max results to return |
| `neuron_type` | string | null | Filter: `episodic`, `semantic`, `procedural`, `schema` |
| `layer` | string | null | Filter: `working`, `short_term`, `long_term` |
| `tags` | string | null | JSON array, matches if any tag overlaps |

### `reflect`

Trigger a memory maintenance cycle:

1. Apply forgetting curve decay to all neurons
2. Prune dead memories (strength < 0.05)
3. Move expired working memory to short-term
4. Promote qualified short-term memories to long-term
5. Consolidate similar long-term memories into schema neurons
6. Decay synapse weights

### `forget`

Manually delete memories. Supports preview before deletion.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `neuron_id` | string | null | Specific neuron ID to delete |
| `query` | string | null | Search query to find memories to delete |
| `top_k` | int | `5` | Number of matches to preview |
| `confirm` | bool | `false` | Set `true` to execute deletion |

### `associate`

Manually create a synapse between two neurons.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `neuron_id_a` | string | (required) | Source neuron ID |
| `neuron_id_b` | string | (required) | Target neuron ID |
| `synapse_type` | string | `"reference"` | `semantic`, `temporal`, `causal`, `hierarchical`, `reference` |
| `weight` | float | `0.7` | Connection strength 0.0-1.0 |

### `memory_status`

Get system statistics: neuron counts by layer/type, synapse counts, working memory usage, semantic index size, and file storage stats. No parameters.

## Configuration

Edit `config.yaml` or override with environment variables (prefix `NEURAL_MEMORY_`):

```yaml
storage_dir: "./storage"                          # Data directory
embedding_model: "paraphrase-multilingual-MiniLM-L12-v2"  # Sentence-transformers model
log_level: "INFO"

decay:
  base_decay_rate: 0.15         # Lambda in S(t) = S0 * e^(-lambda*t/stability)
  stability_growth_on_review: 0.3
  max_stability: 10.0
  prune_threshold: 0.05         # Strength below this -> prune
  importance_shield: 0.3        # Importance above this -> slower decay
  emotional_arousal_shield: 0.5

activation:
  decay_per_hop: 0.6            # Spreading activation decay factor per hop
  max_hops: 4
  max_nodes_visited: 200

ranker:
  semantic_weight: 0.40         # Hybrid ranking weights (must sum to 1.0)
  fts_weight: 0.15
  activation_weight: 0.30
  recency_weight: 0.10
  strength_weight: 0.05

working_memory:
  capacity: 20                  # Max neurons in working memory
  ttl_hours: 2.0                # Time before eviction to short-term
  promotion_strength_threshold: 0.6
  short_term_to_long_term_hours: 48.0
  short_term_access_threshold: 3

consolidation:
  similarity_threshold: 0.82    # Cosine similarity to cluster
  min_cluster_size: 3
  max_cluster_size: 15
```

## Architecture

All I/O is fully async. Pure-computation modules (decay, consolidation, ranking, neuron/synapse dataclasses) remain synchronous.

```
src/neural_memory/
├── server.py                 # MCP server + 6 async tool handlers
├── config.py                 # YAML config loading + env overrides
├── core/
│   ├── memory_store.py       # Top-level async orchestrator
│   ├── neuron.py             # Neuron dataclass + factory
│   ├── synapse.py            # Synapse dataclass + Hebbian update
│   ├── decay.py              # Ebbinghaus forgetting curve engine
│   ├── working_memory.py     # Async capacity-limited working memory buffer
│   └── consolidation.py      # Greedy agglomerative clustering -> schema neurons
├── db/
│   ├── connection.py         # aiosqlite WAL mode + asyncio.Lock write serialization
│   ├── schema.sql            # Tables: neurons, synapses, working_memory, FTS5
│   └── repositories.py       # Async NeuronRepository + SynapseRepository
├── embeddings/
│   ├── provider.py           # Abstract async embedding interface
│   └── local_model.py        # sentence-transformers (asyncio.to_thread) + DB cache
├── retrieval/
│   ├── semantic_search.py    # Brute-force cosine similarity (numpy, sync search + async index)
│   ├── spreading_activation.py  # Async priority-queue BFS through synapse graph
│   └── hybrid_ranker.py      # Multi-signal weighted ranking (sync)
└── files/
    └── file_store.py         # Async SHA-256 dedup, type-based directory storage
```

## Memory Lifecycle

```
Created -> Working Memory (capacity=20, TTL=2h)
              |
              | (expired or evicted)
              v
         Short-term Memory
              |
              | (strength > 0.6, or importance > 0.4, or access >= 3)
              v
         Long-term Memory  ---[consolidation]--->  Schema Neuron
              |
              | (strength < 0.05, low importance)
              v
           Pruned (deleted)
```

## Key Algorithms

**Forgetting curve**: `S(t) = S0 * exp(-lambda * t / stability)` where stability grows with each access.

**Hybrid ranking**: `score = 0.40*semantic + 0.15*fts + 0.30*activation + 0.10*recency + 0.05*strength`

**Spreading activation**: BFS with priority queue, 60% decay per hop, max 4 hops, weight^1.5 amplification.

**Hebbian learning**: `w += lr * (w_max - w)` asymptotic update on co-accessed neuron pairs.

**Consolidation**: Greedy agglomerative clustering (cosine > 0.82, min 3 neurons) -> schema neuron with centroid embedding.

## Async Design

The server is fully async to prevent MCP tool-call timeouts (AbortError / -32001) that occur when blocking the event loop.

| Layer | Strategy |
|-------|----------|
| **Database** | `aiosqlite` — single connection, WAL mode for concurrent reads, `asyncio.Lock` for serialized writes |
| **Embedding** | `asyncio.to_thread(model.encode, ...)` — offloads CPU-bound sentence-transformers to the thread pool |
| **File I/O** | `asyncio.to_thread(path.write_bytes, ...)` — keeps the event loop free during disk writes |
| **Semantic search** | `search()` stays **synchronous** (pure numpy, <1ms for 100K vectors); index mutations are async + `asyncio.Lock` |
| **Spreading activation** | `activate()` is async — BFS loop awaits DB queries at each hop |
| **Pure computation** | Decay, consolidation, ranking remain synchronous — no I/O, no blocking |

`MemoryStore.__init__()` is synchronous (safe to call anywhere); the separate `await store.initialize()` opens the DB connection and builds the semantic index.

## License

MIT
