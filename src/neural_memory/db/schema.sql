-- Neural Memory Database Schema
-- Modeled after biological neural networks: neurons + synapses

PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

-- ============================================================
-- Neurons: Core memory units
-- ============================================================
CREATE TABLE IF NOT EXISTS neurons (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    summary TEXT,
    neuron_type TEXT NOT NULL CHECK(neuron_type IN ('episodic', 'semantic', 'procedural', 'schema')),
    layer TEXT NOT NULL DEFAULT 'working' CHECK(layer IN ('working', 'short_term', 'long_term')),

    -- Memory dynamics
    strength REAL NOT NULL DEFAULT 1.0,
    stability REAL NOT NULL DEFAULT 1.0,
    importance REAL NOT NULL DEFAULT 0.5,

    -- Emotional markers
    emotional_valence REAL DEFAULT 0.0,   -- [-1, 1]: negative to positive
    emotional_arousal REAL DEFAULT 0.0,   -- [0, 1]: calm to intense

    -- Access tracking
    access_count INTEGER DEFAULT 0,

    -- Embedding (384-dim float32 vector stored as BLOB)
    embedding BLOB,
    embedding_model TEXT,

    -- Metadata
    tags TEXT DEFAULT '[]',               -- JSON array of strings
    source TEXT DEFAULT 'manual',

    -- File reference
    file_path TEXT,                       -- Relative path in files/ directory
    file_type TEXT,
    file_hash TEXT,

    -- Timestamps (ISO 8601)
    created_at TEXT NOT NULL,
    last_accessed TEXT NOT NULL,
    last_decayed TEXT NOT NULL
);

-- ============================================================
-- Synapses: Weighted connections between neurons
-- ============================================================
CREATE TABLE IF NOT EXISTS synapses (
    id TEXT PRIMARY KEY,
    pre_neuron_id TEXT NOT NULL REFERENCES neurons(id) ON DELETE CASCADE,
    post_neuron_id TEXT NOT NULL REFERENCES neurons(id) ON DELETE CASCADE,
    synapse_type TEXT NOT NULL CHECK(synapse_type IN ('semantic', 'temporal', 'causal', 'hierarchical', 'reference')),
    weight REAL NOT NULL DEFAULT 0.5 CHECK(weight >= 0.0 AND weight <= 1.0),
    activation_count INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    last_activated TEXT NOT NULL
);

-- ============================================================
-- Working Memory: Limited-capacity buffer (like prefrontal cortex)
-- ============================================================
CREATE TABLE IF NOT EXISTS working_memory (
    neuron_id TEXT PRIMARY KEY REFERENCES neurons(id) ON DELETE CASCADE,
    entered_at TEXT NOT NULL,
    priority REAL DEFAULT 0.5
);

-- ============================================================
-- Embedding Cache: Avoid re-computing embeddings
-- ============================================================
CREATE TABLE IF NOT EXISTS embedding_cache (
    text_hash TEXT NOT NULL,
    model TEXT NOT NULL,
    embedding BLOB NOT NULL,
    created_at TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (text_hash, model)
);

-- ============================================================
-- Indexes for efficient queries
-- ============================================================

-- Neuron indexes
CREATE INDEX IF NOT EXISTS idx_neurons_layer ON neurons(layer);
CREATE INDEX IF NOT EXISTS idx_neurons_type ON neurons(neuron_type);
CREATE INDEX IF NOT EXISTS idx_neurons_strength ON neurons(strength);
CREATE INDEX IF NOT EXISTS idx_neurons_importance ON neurons(importance);
CREATE INDEX IF NOT EXISTS idx_neurons_created_at ON neurons(created_at);
CREATE INDEX IF NOT EXISTS idx_neurons_last_accessed ON neurons(last_accessed);
CREATE INDEX IF NOT EXISTS idx_neurons_file_hash ON neurons(file_hash);

-- Synapse indexes
CREATE INDEX IF NOT EXISTS idx_synapses_pre ON synapses(pre_neuron_id);
CREATE INDEX IF NOT EXISTS idx_synapses_post ON synapses(post_neuron_id);
CREATE INDEX IF NOT EXISTS idx_synapses_type ON synapses(synapse_type);
CREATE INDEX IF NOT EXISTS idx_synapses_weight ON synapses(weight);

-- Working memory indexes
CREATE INDEX IF NOT EXISTS idx_wm_priority ON working_memory(priority);
CREATE INDEX IF NOT EXISTS idx_wm_entered_at ON working_memory(entered_at);

-- ============================================================
-- FTS5 Full-Text Search (for text-based retrieval)
-- ============================================================
CREATE VIRTUAL TABLE IF NOT EXISTS neurons_fts USING fts5(
    content,
    summary,
    tags,
    content=neurons,
    content_rowid=rowid
);

-- Triggers to keep FTS index in sync
CREATE TRIGGER IF NOT EXISTS neurons_ai AFTER INSERT ON neurons BEGIN
    INSERT INTO neurons_fts(rowid, content, summary, tags)
    VALUES (new.rowid, new.content, new.summary, new.tags);
END;

CREATE TRIGGER IF NOT EXISTS neurons_ad AFTER DELETE ON neurons BEGIN
    INSERT INTO neurons_fts(neurons_fts, rowid, content, summary, tags)
    VALUES ('delete', old.rowid, old.content, old.summary, old.tags);
END;

CREATE TRIGGER IF NOT EXISTS neurons_au AFTER UPDATE ON neurons BEGIN
    INSERT INTO neurons_fts(neurons_fts, rowid, content, summary, tags)
    VALUES ('delete', old.rowid, old.content, old.summary, old.tags);
    INSERT INTO neurons_fts(rowid, content, summary, tags)
    VALUES (new.rowid, new.content, new.summary, new.tags);
END;
