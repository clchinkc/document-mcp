#!/bin/bash
# Document-MCP Verification Script
# Tests Claude Code and Claude Desktop integration

set -e

echo "=========================================="
echo "  Document-MCP Verification Script"
echo "=========================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PASS() { echo -e "${GREEN}✓ PASS${NC}: $1"; }
FAIL() { echo -e "${RED}✗ FAIL${NC}: $1"; }
INFO() { echo -e "${YELLOW}ℹ INFO${NC}: $1"; }

# Test 1: Check if document-mcp is installed
echo -e "\n[1/6] Checking installation..."
if command -v document-mcp &> /dev/null; then
    PASS "document-mcp binary found"
    document-mcp --help > /dev/null 2>&1 && PASS "document-mcp runs" || FAIL "document-mcp --help failed"
else
    FAIL "document-mcp not in PATH"
    INFO "Try: pip install document-mcp"
fi

# Test 2: Check Python module
echo -e "\n[2/6] Checking Python module..."
if python3 -c "import document_mcp" 2>/dev/null; then
    PASS "Python module document_mcp imports"
else
    FAIL "Python module not found"
    INFO "Try: pip install document-mcp"
fi

# Test 3: Check MCP library version
echo -e "\n[3/6] Checking MCP library..."
MCP_VERSION=$(python3 -c "import mcp; print(mcp.__version__)" 2>/dev/null || echo "unknown")
INFO "MCP library version: $MCP_VERSION"
if [ "$MCP_VERSION" != "unknown" ]; then
    PASS "MCP library installed"
else
    FAIL "MCP library not found"
fi

# Test 4: Check Claude Code MCP command
echo -e "\n[4/6] Checking Claude Code integration..."
if command -v claude &> /dev/null; then
    PASS "Claude Code CLI found"
    claude mcp list > /dev/null 2>&1 && PASS "Claude Code MCP command works" || INFO "Claude Code not running or no MCP servers"
else
    INFO "Claude Code not installed (skipping)"
fi

# Test 5: Claude Desktop config check
echo -e "\n[5/6] Checking Claude Desktop config..."
DESKTOP_CONFIG="$HOME/Library/Application Support/Claude/claude_desktop_config.json"
if [ -f "$DESKTOP_CONFIG" ]; then
    if grep -q "document-mcp" "$DESKTOP_CONFIG" 2>/dev/null; then
        PASS "Document-MCP configured in Claude Desktop"
    else
        INFO "Claude Desktop config exists but Document-MCP not configured"
    fi
else
    INFO "Claude Desktop config not found (normal if not used)"
fi

# Test 6: Quick server startup test
echo -e "\n[6/6] Testing server startup..."
timeout 3 python3 -m document_mcp.doc_tool_server stdio --help > /dev/null 2>&1 || true
if [ $? -eq 124 ]; then
    PASS "Server starts without immediate errors"
elif python3 -m document_mcp.doc_tool_server --help > /dev/null 2>&1; then
    PASS "Server module loads correctly"
else
    FAIL "Server module failed to load"
fi

# Summary
echo -e "\n=========================================="
echo "  Verification Complete"
echo "=========================================="
echo ""
echo "To add to Claude Code (recommended):"
echo "  claude mcp add document-mcp -s local -- document-mcp stdio"
echo ""
echo "To add to Claude Desktop, add to:"
echo "  ~/Library/Application Support/Claude/claude_desktop_config.json"
echo ""
echo "Config example:"
echo '  { "mcpServers": { "document-mcp": { "command": "document-mcp", "args": ["stdio"] } } }'
echo ""
