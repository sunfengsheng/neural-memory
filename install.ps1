# Neural Memory Plugin Installer for Claude Code (Windows)
# Usage: git clone https://github.com/sunfengsheng/neural-memory.git; cd neural-memory; .\install.ps1

$ErrorActionPreference = "Stop"

$PluginName = "neural-memory"
$Version = "0.1.0"
$ClaudeDir = "$env:USERPROFILE\.claude"
$MarketplaceDir = "$ClaudeDir\plugins\marketplaces\$PluginName"
$CacheDir = "$ClaudeDir\plugins\cache\$PluginName\$PluginName\$Version"

Write-Host "=== Neural Memory Plugin Installer ===" -ForegroundColor Cyan
Write-Host ""

# 1. Check Python
$PythonCmd = $null
foreach ($cmd in @("python", "python3")) {
    try {
        $ver = & $cmd --version 2>&1
        if ($ver -match "Python 3\.\d+") {
            $PythonCmd = $cmd
            break
        }
    } catch {}
}

if (-not $PythonCmd) {
    Write-Host "Error: Python 3.10+ is required but not found." -ForegroundColor Red
    exit 1
}

Write-Host "[1/5] Python found: $(& $PythonCmd --version)" -ForegroundColor Green

# 2. Install dependencies
Write-Host "[2/5] Installing Python dependencies..." -ForegroundColor Yellow
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
& $PythonCmd -m pip install -e $ScriptDir --quiet 2>&1 | Select-Object -Last 1
Write-Host "  Done." -ForegroundColor Green

# 3. Copy to marketplace
if ((Resolve-Path $ScriptDir).Path -ne (Resolve-Path $MarketplaceDir -ErrorAction SilentlyContinue).Path) {
    Write-Host "[3/5] Copying to marketplace directory..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Path $MarketplaceDir -Force | Out-Null
    Copy-Item -Path "$ScriptDir\*" -Destination $MarketplaceDir -Recurse -Force
    if (Test-Path "$ScriptDir\.claude-plugin") {
        Copy-Item -Path "$ScriptDir\.claude-plugin" -Destination $MarketplaceDir -Recurse -Force
    }
    if (Test-Path "$ScriptDir\.mcp.json") {
        Copy-Item -Path "$ScriptDir\.mcp.json" -Destination $MarketplaceDir -Force
    }
} else {
    Write-Host "[3/5] Already in marketplace directory, skipping copy." -ForegroundColor Green
}

# 4. Create cache directory with flat-format .mcp.json
Write-Host "[4/5] Setting up cache directory..." -ForegroundColor Yellow
New-Item -ItemType Directory -Path $CacheDir -Force | Out-Null
Copy-Item -Path "$MarketplaceDir\*" -Destination $CacheDir -Recurse -Force
if (Test-Path "$MarketplaceDir\.claude-plugin") {
    Copy-Item -Path "$MarketplaceDir\.claude-plugin" -Destination $CacheDir -Recurse -Force
}

# Write flat-format .mcp.json (cache needs this, no mcpServers wrapper)
$McpJson = @'
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
'@
$McpJson | Set-Content -Path "$CacheDir\.mcp.json" -Encoding UTF8
Write-Host "  Done." -ForegroundColor Green

# 5. Verify model
Write-Host "[5/5] Checking embedding model..." -ForegroundColor Yellow
$ModelFile = "$CacheDir\models\paraphrase-multilingual-MiniLM-L12-v2\model.safetensors"
if (Test-Path $ModelFile) {
    $Size = [math]::Round((Get-Item $ModelFile).Length / 1MB, 0)
    Write-Host "  Model found: ${Size}MB" -ForegroundColor Green
} else {
    Write-Host "  WARNING: model.safetensors not found!" -ForegroundColor Red
    Write-Host "  If you cloned without git-lfs, run:" -ForegroundColor Yellow
    Write-Host "    git lfs install" -ForegroundColor White
    Write-Host "    git lfs pull" -ForegroundColor White
    Write-Host "  Then re-run this script." -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "=== Installation complete! ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor White
Write-Host "  1. Restart Claude Code" -ForegroundColor White
Write-Host "  2. The neural-memory MCP server will auto-start in new conversations" -ForegroundColor White
Write-Host "  3. Test: ask Claude to 'remember something' or 'recall'" -ForegroundColor White
Write-Host ""
Write-Host "Directories:" -ForegroundColor White
Write-Host "  Marketplace: $MarketplaceDir" -ForegroundColor Gray
Write-Host "  Cache:       $CacheDir" -ForegroundColor Gray
Write-Host "  Storage:     $env:USERPROFILE\.neural-memory\storage" -ForegroundColor Gray
