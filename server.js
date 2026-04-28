const { Client } = require("@modelcontextprotocol/sdk/client/index.js");
const {
  StdioClientTransport,
} = require("@modelcontextprotocol/sdk/client/stdio.js");
const express = require("express");
const path = require("path");

// ── Config ─────────────────────────────────────────────────────────
const PLUGIN_ROOT = path.resolve(
  "C:/Users/32712/.claude/plugins/marketplaces/neural-memory"
);
const STORAGE_DIR = path.resolve(
  process.env.NEURAL_MEMORY_STORAGE || "C:/Users/32712/.neural-memory/storage"
);
const PORT = 3456;

let mcpClient = null;

// ── MCP Client ─────────────────────────────────────────────────────
async function connectMCP() {
  const transport = new StdioClientTransport({
    command: "python",
    args: ["-m", "neural_memory", "--storage-dir", STORAGE_DIR],
    env: {
      ...process.env,
      PYTHONPATH: path.join(PLUGIN_ROOT, "src"),
      HF_HUB_OFFLINE: "1",
      TRANSFORMERS_OFFLINE: "1",
      NEURAL_MEMORY_EMBEDDING_MODEL: path.join(
        PLUGIN_ROOT,
        "models/paraphrase-multilingual-MiniLM-L12-v2"
      ),
    },
    stderr: "inherit",
  });

  mcpClient = new Client({ name: "memory-viz", version: "1.0.0" });
  await mcpClient.connect(transport);
  const caps = mcpClient.getServerCapabilities();
  console.log("[MCP] connected, capabilities:", JSON.stringify(caps));
}

/** Convenience: call an MCP tool and parse the JSON text result */
async function callTool(name, args = {}) {
  const result = await mcpClient.callTool({ name, arguments: args });
  const text = result.content?.find((c) => c.type === "text")?.text;
  return text ? JSON.parse(text) : result;
}

// ── Express ─────────────────────────────────────────────────────────
const app = express();
app.use(express.json());
app.use(express.static(path.join(__dirname, "public")));

// --- Graph data (neurons + synapses) via MCP list_graph ---
app.get("/api/graph", async (_req, res) => {
  try {
    res.json(await callTool("list_graph"));
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// --- MCP: memory_status ---
app.get("/api/status", async (_req, res) => {
  try {
    res.json(await callTool("memory_status"));
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// --- MCP: recall ---
app.get("/api/recall", async (req, res) => {
  try {
    const { query = "", top_k = "20", neuron_type, layer, tags } = req.query;
    const args = { query, top_k: parseInt(top_k, 10) };
    if (neuron_type) args.neuron_type = neuron_type;
    if (layer) args.layer = layer;
    if (tags) args.tags = tags;
    res.json(await callTool("recall", args));
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// --- MCP: remember ---
app.post("/api/remember", async (req, res) => {
  try {
    const {
      content,
      neuron_type = "semantic",
      importance = 0.5,
      tags = "[]",
      source = "web-ui",
    } = req.body;
    if (!content) return res.status(400).json({ error: "content is required" });
    res.json(
      await callTool("remember", {
        content,
        neuron_type,
        importance,
        tags,
        source,
      })
    );
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// --- MCP: forget ---
app.post("/api/forget", async (req, res) => {
  try {
    const { neuron_id, query, confirm = false } = req.body;
    const args = { confirm };
    if (neuron_id) args.neuron_id = neuron_id;
    if (query) args.query = query;
    res.json(await callTool("forget", args));
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// --- MCP: associate ---
app.post("/api/associate", async (req, res) => {
  try {
    const {
      neuron_id_a,
      neuron_id_b,
      synapse_type = "reference",
      weight = 0.7,
    } = req.body;
    if (!neuron_id_a || !neuron_id_b)
      return res
        .status(400)
        .json({ error: "neuron_id_a and neuron_id_b required" });
    res.json(
      await callTool("associate", {
        neuron_id_a,
        neuron_id_b,
        synapse_type,
        weight,
      })
    );
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// --- MCP: reflect ---
app.post("/api/reflect", async (_req, res) => {
  try {
    res.json(await callTool("reflect"));
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// ── Start ───────────────────────────────────────────────────────────
app.listen(PORT, async () => {
  try {
    await connectMCP();
    console.log(`\n  Memory Visualizer running at http://localhost:${PORT}\n`);
  } catch (err) {
    console.error("[MCP] failed to connect:", err.message);
    console.log(
      `\n  Server started at http://localhost:${PORT} (MCP not connected)\n`
    );
  }
});
