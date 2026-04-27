#!/usr/bin/env bash
set -euo pipefail

# Neural Memory Plugin Installer for Claude Code
# Usage: git clone https://github.com/sunfengsheng/neural-memory.git && cd neural-memory && bash install.sh

PLUGIN_NAME="neural-memory"
VERSION="0.1.0"
CLAUDE_DIR="${HOME}/.claude"
MARKETPLACE_DIR="${CLAUDE_DIR}/plugins/marketplaces/${PLUGIN_NAME}"
CACHE_DIR="${CLAUDE_DIR}/plugins/cache/${PLUGIN_NAME}/${PLUGIN_NAME}/${VERSION}"

echo "=== Neural Memory Plugin Installer ==="
echo ""

# 1. Check Python
if ! command -v python &>/dev/null && ! command -v python3 &>/dev/null; then
    echo "Error: Python 3.10+ is required but not found."
    exit 1
fi

PYTHON_CMD=$(command -v python3 || command -v python)
PY_VERSION=$($PYTHON_CMD --version 2>&1 | grep -oP '\d+\.\d+')
echo "[1/5] Python found: $($PYTHON_CMD --version)"

# 2. Install dependencies
echo "[2/5] Installing Python dependencies..."
$PYTHON_CMD -m pip install -e "$(pwd)" --quiet 2>&1 | tail -1 || {
    echo "  pip install failed, trying with --user..."
    $PYTHON_CMD -m pip install -e "$(pwd)" --user --quiet
}
echo "  Done."

# 3. Copy to marketplace (if not already there)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ "$SCRIPT_DIR" != "$MARKETPLACE_DIR" ]; then
    echo "[3/5] Copying to marketplace directory..."
    mkdir -p "$MARKETPLACE_DIR"
    cp -r "$SCRIPT_DIR/"* "$MARKETPLACE_DIR/"
    cp -r "$SCRIPT_DIR/.claude-plugin" "$MARKETPLACE_DIR/" 2>/dev/null || true
    cp "$SCRIPT_DIR/.mcp.json" "$MARKETPLACE_DIR/" 2>/dev/null || true
    cp "$SCRIPT_DIR/.gitattributes" "$MARKETPLACE_DIR/" 2>/dev/null || true
else
    echo "[3/5] Already in marketplace directory, skipping copy."
fi

# 4. Create cache directory with flat-format .mcp.json
echo "[4/5] Setting up cache directory..."
mkdir -p "$CACHE_DIR"
cp -r "$MARKETPLACE_DIR/"* "$CACHE_DIR/"
cp -r "$MARKETPLACE_DIR/.claude-plugin" "$CACHE_DIR/" 2>/dev/null || true

# Write flat-format .mcp.json (cache needs this format, no mcpServers wrapper)
cat > "$CACHE_DIR/.mcp.json" << 'MCPEOF'
{
  "neural-memory": {
    "type": "stdio",
    "command": "python",
    "args": ["-m", "neural_memory"],
    "env": {
      "PYTHONPATH": "${CLAUDE_PLUGIN_ROOT}/src",
      "HF_HUB_OFFLINE": "1",
      "TRANSFORMERS_OFFLINE": "1",
      "NEURAL_MEMORY_EMBEDDING_MODEL": "${CLAUDE_PLUGIN_ROOT}/models/paraphrase-multilingual-MiniLM-L12-v2"
    }
  }
}
MCPEOF
echo "  Done."

# 5. Verify model
echo "[5/5] Checking embedding model..."
MODEL_DIR="$CACHE_DIR/models/paraphrase-multilingual-MiniLM-L12-v2"
if [ -f "$MODEL_DIR/model.safetensors" ]; then
    SIZE=$(du -sh "$MODEL_DIR/model.safetensors" | cut -f1)
    echo "  Model found: $SIZE"
else
    echo "  WARNING: model.safetensors not found!"
    echo "  If you cloned without git-lfs, run:"
    echo "    git lfs install && git lfs pull"
    echo "  Then re-run this script."
    exit 1
fi

echo ""
echo "=== Installation complete! ==="
echo ""
echo "Next steps:"
echo "  1. Restart Claude Code"
echo "  2. In a new conversation, the neural-memory MCP server will auto-start"
echo "  3. Test with: ask Claude to 'remember something' or 'recall'"
echo ""
echo "Directories:"
echo "  Marketplace: $MARKETPLACE_DIR"
echo "  Cache:       $CACHE_DIR"
echo "  Storage:     ~/.neural-memory/storage"
