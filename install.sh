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
PYTHON_CMD=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        PYTHON_CMD="$cmd"
        break
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "Error: Python 3.10+ is required but not found."
    exit 1
fi

echo "[1/6] Python found: $($PYTHON_CMD --version)"

# 2. Verify model (fail-fast before copying files)
echo "[2/6] Checking embedding model..."
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_FILE="$SCRIPT_DIR/models/paraphrase-multilingual-MiniLM-L12-v2/model.safetensors"
if [ -f "$MODEL_FILE" ]; then
    SIZE=$(du -sh "$MODEL_FILE" | cut -f1)
    echo "  Model found: $SIZE"
else
    echo "  ERROR: model.safetensors not found!"
    echo "  If you cloned without git-lfs, run:"
    echo "    git lfs install && git lfs pull"
    echo "  Then re-run this script."
    exit 1
fi

# 3. Install dependencies
echo "[3/6] Installing Python dependencies..."
$PYTHON_CMD -m pip install -e "$SCRIPT_DIR" --quiet 2>&1 | tail -1 || {
    echo "  pip install failed, trying with --user..."
    $PYTHON_CMD -m pip install -e "$SCRIPT_DIR" --user --quiet
}
echo "  Done."

# 4. Copy to marketplace
if [ "$SCRIPT_DIR" != "$MARKETPLACE_DIR" ]; then
    echo "[4/6] Copying to marketplace directory..."
    mkdir -p "$MARKETPLACE_DIR"
    cp -r "$SCRIPT_DIR/"* "$MARKETPLACE_DIR/"
    cp -r "$SCRIPT_DIR/.claude-plugin" "$MARKETPLACE_DIR/" 2>/dev/null || true
    cp "$SCRIPT_DIR/.mcp.json" "$MARKETPLACE_DIR/" 2>/dev/null || true
    cp "$SCRIPT_DIR/.gitattributes" "$MARKETPLACE_DIR/" 2>/dev/null || true
else
    echo "[4/6] Already in marketplace directory, skipping copy."
fi

# 5. Create cache directory
echo "[5/6] Setting up cache directory..."
mkdir -p "$CACHE_DIR"
cp -r "$MARKETPLACE_DIR/"* "$CACHE_DIR/"
cp -r "$MARKETPLACE_DIR/.claude-plugin" "$CACHE_DIR/" 2>/dev/null || true
cp "$MARKETPLACE_DIR/.mcp.json" "$CACHE_DIR/" 2>/dev/null || true
echo "  Done."

# 6. Verify model in cache
echo "[6/6] Verifying model in cache..."
MODEL_DIR="$CACHE_DIR/models/paraphrase-multilingual-MiniLM-L12-v2"
if [ -f "$MODEL_DIR/model.safetensors" ]; then
    SIZE=$(du -sh "$MODEL_DIR/model.safetensors" | cut -f1)
    echo "  Model found: $SIZE"
else
    echo "  ERROR: model.safetensors not found in cache!"
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
