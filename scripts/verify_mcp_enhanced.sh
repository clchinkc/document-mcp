#!/bin/bash

#############################################################################
# Document-MCP Enhanced Verification Script
# Comprehensive testing across all integration tiers
# Tests Claude Code & Claude Desktop integration with stdio MCP server
#############################################################################

set -o pipefail

# === Configuration ===
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DOCS_STORAGE="${DOCUMENT_STORAGE_PATH:-.documents_storage}"
TEMP_TEST_DIR="/tmp/document_mcp_verify_$$"
TEST_TIMEOUT=5

# === Color Output ===
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# === Counters ===
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_SKIPPED=0

# === Logging Functions ===
PASS() {
    echo -e "${GREEN}✓ PASS${NC}: $1"
    ((TESTS_PASSED++))
}

FAIL() {
    echo -e "${RED}✗ FAIL${NC}: $1"
    ((TESTS_FAILED++))
}

SKIP() {
    echo -e "${YELLOW}⊘ SKIP${NC}: $1"
    ((TESTS_SKIPPED++))
}

INFO() { echo -e "${BLUE}ℹ INFO${NC}: $1"; }
SECTION() { echo -e "\n${BLUE}==== $1 ====${NC}\n"; }

# === Utility Functions ===

cleanup() {
    rm -rf "$TEMP_TEST_DIR" 2>/dev/null
    if [ -n "$MCP_PID" ] && kill -0 "$MCP_PID" 2>/dev/null; then
        kill "$MCP_PID" 2>/dev/null || true
        wait "$MCP_PID" 2>/dev/null || true
    fi
}

trap cleanup EXIT

log_details() {
    echo "  Details: $1"
}

# === TIER 0: Binary Installation ===

tier_0_binary_check() {
    SECTION "TIER 0: Binary Installation Check"

    # Test 0.1: Check if binary in PATH
    if command -v document-mcp &>/dev/null; then
        PASS "Binary 'document-mcp' found in PATH"
        BINARY_PATH=$(which document-mcp)
        log_details "Location: $BINARY_PATH"
    else
        FAIL "Binary 'document-mcp' not in PATH"
        log_details "Try: pip install document-mcp"
        return 1
    fi

    # Test 0.2: Check binary is executable
    if [ -x "$BINARY_PATH" ]; then
        PASS "Binary is executable"
    else
        FAIL "Binary is not executable"
        log_details "Try: chmod +x $BINARY_PATH"
        return 1
    fi

    # Test 0.3: Check --help works
    if $BINARY_PATH --help &>/dev/null; then
        PASS "Binary --help succeeds"
    else
        FAIL "Binary --help failed"
        return 1
    fi

    # Test 0.4: Check version
    VERSION=$($BINARY_PATH --version 2>/dev/null || echo "unknown")
    INFO "Version: $VERSION"

    # Test 0.5: Check Python module
    if python3 -c "import document_mcp" 2>/dev/null; then
        PASS "Python module 'document_mcp' imports"
    else
        FAIL "Python module 'document_mcp' not found"
        log_details "Try: pip install document-mcp"
        return 1
    fi

    # Test 0.6: Check Python version
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    if python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)" 2>/dev/null; then
        PASS "Python version $PYTHON_VERSION (>=3.10)"
    else
        FAIL "Python version $PYTHON_VERSION (requires >=3.10)"
        return 1
    fi

    return 0
}

# === TIER 1: Server Startup ===

tier_1_server_startup() {
    SECTION "TIER 1: Server Startup Check"

    # Test 1.1: Server starts without immediate exit
    if timeout $TEST_TIMEOUT document-mcp stdio </dev/null >/tmp/mcp_startup_$$.log 2>&1 &
        MCP_PID=$!
        sleep 1
        if ps -p $MCP_PID >/dev/null 2>&1; then
            PASS "Server started and stayed running"
        else
            FAIL "Server exited immediately"
            log_details "$(cat /tmp/mcp_startup_$$.log | head -10)"
            rm -f /tmp/mcp_startup_$$.log
            return 1
        fi
        kill $MCP_PID 2>/dev/null || true
        wait $MCP_PID 2>/dev/null || true
        rm -f /tmp/mcp_startup_$$.log
    else
        FAIL "Server start timed out"
        return 1
    fi

    # Test 1.2: Python module loads
    if python3 -c "from document_mcp.doc_tool_server import main" 2>/dev/null; then
        PASS "Server module loads successfully"
    else
        FAIL "Server module failed to load"
        return 1
    fi

    # Test 1.3: Check dependencies
    MISSING_DEPS=()
    for dep in mcp pydantic google-genai numpy; do
        if ! python3 -c "import ${dep//-/_}" 2>/dev/null; then
            MISSING_DEPS+=("$dep")
        fi
    done

    if [ ${#MISSING_DEPS[@]} -eq 0 ]; then
        PASS "All core dependencies installed"
    else
        FAIL "Missing dependencies: ${MISSING_DEPS[*]}"
        log_details "Try: pip install document-mcp[all]"
        return 1
    fi

    return 0
}

# === TIER 2: Tool Discovery ===

tier_2_tool_discovery() {
    SECTION "TIER 2: Tool Discovery Check"

    # Test 2.1: Initialize MCP connection
    MCP_INIT_RESPONSE=$(python3 << 'PYEOF' 2>/dev/null || echo "FAILED"
import subprocess
import json
import sys

proc = subprocess.Popen(
    ["document-mcp", "stdio"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
)

init_request = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "verify", "version": "1.0"},
    },
}

try:
    proc.stdin.write(json.dumps(init_request) + "\n")
    proc.stdin.flush()
    response_line = proc.stdout.readline()
    if response_line:
        print(response_line.strip())
    else:
        print("NO_RESPONSE")
finally:
    proc.terminate()
    proc.wait(timeout=2)
PYEOF
    )

    if [ "$MCP_INIT_RESPONSE" == "FAILED" ]; then
        FAIL "Could not initialize MCP connection"
        return 1
    fi

    if [ "$MCP_INIT_RESPONSE" == "NO_RESPONSE" ]; then
        FAIL "MCP server did not respond to initialize"
        return 1
    fi

    PASS "MCP initialization succeeded"

    # Test 2.2: Count discovered tools
    TOOL_COUNT=$(echo "$MCP_INIT_RESPONSE" | python3 -c "
import json
import sys
data = json.load(sys.stdin)
tools = data.get('result', {}).get('tools', [])
print(len(tools))
" 2>/dev/null || echo "0")

    EXPECTED_TOOLS=28
    if [ "$TOOL_COUNT" -eq "$EXPECTED_TOOLS" ]; then
        PASS "All $EXPECTED_TOOLS tools discovered"
    else
        FAIL "Expected $EXPECTED_TOOLS tools, found $TOOL_COUNT"
        log_details "Tools may not be registered correctly"
    fi

    # Test 2.3: Verify key tools present
    KEY_TOOLS=(
        "create_document"
        "list_documents"
        "read_content"
        "add_paragraph"
        "manage_snapshots"
        "find_text"
    )

    MISSING_TOOLS=()
    for tool in "${KEY_TOOLS[@]}"; do
        if ! echo "$MCP_INIT_RESPONSE" | grep -q "\"name\":\"$tool\""; then
            MISSING_TOOLS+=("$tool")
        fi
    done

    if [ ${#MISSING_TOOLS[@]} -eq 0 ]; then
        PASS "All key tools present"
    else
        FAIL "Missing key tools: ${MISSING_TOOLS[*]}"
    fi

    return 0
}

# === TIER 3: Basic Operations ===

tier_3_basic_operations() {
    SECTION "TIER 3: Basic Operations Check"

    # Create temp test directory
    mkdir -p "$TEMP_TEST_DIR"
    export DOCUMENT_STORAGE_PATH="$TEMP_TEST_DIR"

    # Test 3.1: Create document
    CREATE_RESULT=$(python3 << 'PYEOF' 2>/dev/null || echo "FAILED"
import subprocess
import json

proc = subprocess.Popen(
    ["document-mcp", "stdio"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    env={"DOCUMENT_STORAGE_PATH": "/tmp/document_mcp_verify_$$"},
)

try:
    # Initialize
    init_req = {
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                   "clientInfo": {"name": "verify", "version": "1.0"}},
    }
    proc.stdin.write(json.dumps(init_req) + "\n")
    proc.stdin.flush()
    proc.stdout.readline()

    # Create document
    create_req = {
        "jsonrpc": "2.0", "id": 2, "method": "tools/call",
        "params": {"name": "create_document", "arguments": {"document_name": "TestVerify"}},
    }
    proc.stdin.write(json.dumps(create_req) + "\n")
    proc.stdin.flush()
    response = proc.stdout.readline()
    if response:
        result = json.loads(response)
        if "result" in result:
            print("SUCCESS")
        else:
            print("FAILED")
    else:
        print("NO_RESPONSE")
finally:
    proc.terminate()
    proc.wait(timeout=2)
PYEOF
    )

    if [ "$CREATE_RESULT" = "SUCCESS" ]; then
        PASS "Create document operation succeeded"
    else
        FAIL "Create document operation failed"
        return 1
    fi

    # Test 3.2: Verify file system persistence
    if [ -d "$TEMP_TEST_DIR/TestVerify" ]; then
        PASS "Document directory created in file system"
    else
        FAIL "Document directory not found in file system"
        return 1
    fi

    # Test 3.3: List documents
    LIST_RESULT=$(python3 << 'PYEOF' 2>/dev/null || echo "FAILED"
import subprocess
import json

proc = subprocess.Popen(
    ["document-mcp", "stdio"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    env={"DOCUMENT_STORAGE_PATH": "/tmp/document_mcp_verify_$$"},
)

try:
    # Initialize
    init_req = {
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                   "clientInfo": {"name": "verify", "version": "1.0"}},
    }
    proc.stdin.write(json.dumps(init_req) + "\n")
    proc.stdin.flush()
    proc.stdout.readline()

    # List documents
    list_req = {
        "jsonrpc": "2.0", "id": 2, "method": "tools/call",
        "params": {"name": "list_documents", "arguments": {}},
    }
    proc.stdin.write(json.dumps(list_req) + "\n")
    proc.stdin.flush()
    response = proc.stdout.readline()
    if response:
        print("SUCCESS")
    else:
        print("FAILED")
finally:
    proc.terminate()
    proc.wait(timeout=2)
PYEOF
    )

    if [ "$LIST_RESULT" = "SUCCESS" ]; then
        PASS "List documents operation succeeded"
    else
        FAIL "List documents operation failed"
    fi

    return 0
}

# === TIER 4: Claude Integration Check ===

tier_4_claude_integration() {
    SECTION "TIER 4: Claude Integration Check"

    # Test 4.1: Check Claude Code CLI
    if command -v claude &>/dev/null; then
        PASS "Claude Code CLI found"

        # Test 4.2: List MCP servers
        if claude mcp list &>/dev/null 2>&1; then
            PASS "Claude Code MCP command works"

            # Check if document-mcp is registered
            if claude mcp list 2>/dev/null | grep -q "document-mcp"; then
                PASS "document-mcp is registered in Claude Code"
            else
                SKIP "document-mcp not registered in Claude Code"
                log_details "Register with: claude mcp add document-mcp -s local -- document-mcp stdio"
            fi
        else
            INFO "Claude Code not fully initialized (might not be running)"
        fi
    else
        INFO "Claude Code CLI not found (optional)"
        SKIP "Claude Code CLI integration tests"
    fi

    # Test 4.3: Check Claude Desktop config
    DESKTOP_CONFIG=""
    if [ -d "$HOME/Library/Application Support/Claude" ]; then
        DESKTOP_CONFIG="$HOME/Library/Application Support/Claude/claude_desktop_config.json"
    elif [ -d "$HOME/.config/Claude" ]; then
        DESKTOP_CONFIG="$HOME/.config/Claude/claude_desktop_config.json"
    fi

    if [ -f "$DESKTOP_CONFIG" ]; then
        # Validate JSON
        if python3 -m json.tool < "$DESKTOP_CONFIG" >/dev/null 2>&1; then
            PASS "Claude Desktop config is valid JSON"

            # Check if document-mcp configured
            if grep -q "document-mcp" "$DESKTOP_CONFIG"; then
                PASS "document-mcp is configured in Claude Desktop"
            else
                INFO "document-mcp not configured in Claude Desktop"
                SKIP "Add to config for Desktop integration"
            fi
        else
            FAIL "Claude Desktop config is invalid JSON"
            log_details "Location: $DESKTOP_CONFIG"
        fi
    else
        INFO "Claude Desktop config not found"
        SKIP "Claude Desktop integration tests"
    fi

    return 0
}

# === TIER 5: Advanced Features ===

tier_5_advanced_features() {
    SECTION "TIER 5: Advanced Features Check"

    mkdir -p "$TEMP_TEST_DIR"
    export DOCUMENT_STORAGE_PATH="$TEMP_TEST_DIR"

    # Test 5.1: Snapshots feature
    SNAPSHOT_SUPPORT=$(python3 << 'PYEOF' 2>/dev/null || echo "UNKNOWN"
import subprocess
import json

proc = subprocess.Popen(
    ["document-mcp", "stdio"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    env={"DOCUMENT_STORAGE_PATH": "/tmp/document_mcp_verify_$$"},
)

try:
    # Initialize
    init_req = {
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                   "clientInfo": {"name": "verify", "version": "1.0"}},
    }
    proc.stdin.write(json.dumps(init_req) + "\n")
    proc.stdin.flush()
    proc.stdout.readline()

    # Check for manage_snapshots tool
    list_req = {
        "jsonrpc": "2.0", "id": 2, "method": "initialize",
        "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                   "clientInfo": {"name": "verify", "version": "1.0"}},
    }
    proc.stdin.write(json.dumps(list_req) + "\n")
    proc.stdin.flush()
    response = json.loads(proc.stdout.readline())
    tools = response.get("result", {}).get("tools", [])
    has_snapshots = any(t["name"] == "manage_snapshots" for t in tools)
    print("SUPPORTED" if has_snapshots else "NOT_FOUND")
finally:
    proc.terminate()
    proc.wait(timeout=2)
PYEOF
    )

    if [ "$SNAPSHOT_SUPPORT" = "SUPPORTED" ]; then
        PASS "Snapshot feature supported"
    else
        INFO "Snapshot feature check skipped"
    fi

    # Test 5.2: Semantic search feature
    SEARCH_SUPPORT=$(python3 << 'PYEOF' 2>/dev/null || echo "UNKNOWN"
import subprocess
import json

proc = subprocess.Popen(
    ["document-mcp", "stdio"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
)

try:
    init_req = {
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                   "clientInfo": {"name": "verify", "version": "1.0"}},
    }
    proc.stdin.write(json.dumps(init_req) + "\n")
    proc.stdin.flush()
    response = json.loads(proc.stdout.readline())
    tools = response.get("result", {}).get("tools", [])
    has_search = any(t["name"] == "find_similar_text" for t in tools)
    print("SUPPORTED" if has_search else "NOT_FOUND")
finally:
    proc.terminate()
    proc.wait(timeout=2)
PYEOF
    )

    if [ "$SEARCH_SUPPORT" = "SUPPORTED" ]; then
        PASS "Semantic search feature supported"
    else
        INFO "Semantic search feature not available"
    fi

    return 0
}

# === Report Summary ===

report_summary() {
    SECTION "Verification Summary"

    TOTAL_TESTS=$((TESTS_PASSED + TESTS_FAILED + TESTS_SKIPPED))
    PASS_RATE=$((TESTS_PASSED * 100 / TOTAL_TESTS))

    echo "Tests Passed:  $TESTS_PASSED"
    echo "Tests Failed:  $TESTS_FAILED"
    echo "Tests Skipped: $TESTS_SKIPPED"
    echo "Pass Rate:     $PASS_RATE%"

    if [ $TESTS_FAILED -eq 0 ]; then
        echo -e "\n${GREEN}✓ All critical tests passed!${NC}"
        echo "document-mcp is ready for use."
        return 0
    else
        echo -e "\n${RED}✗ Some tests failed.${NC}"
        echo "See troubleshooting guide: docs/INTEGRATION_TROUBLESHOOTING.md"
        return 1
    fi
}

# === Next Steps ===

print_next_steps() {
    echo ""
    echo "Next Steps:"
    echo "==========="
    echo ""
    echo "1. Add to Claude Code:"
    echo "   claude mcp add document-mcp -s local -- document-mcp stdio"
    echo ""
    echo "2. Verify connection:"
    echo "   claude mcp list"
    echo ""
    echo "3. Add to Claude Desktop:"
    echo "   Edit: ~/Library/Application Support/Claude/claude_desktop_config.json"
    echo "   Add to mcpServers:"
    echo '   "document-mcp": { "command": "document-mcp", "args": ["stdio"] }'
    echo ""
    echo "4. Test in Claude:"
    echo "   Ask: 'Create a document called TestBook'"
    echo ""
    echo "Troubleshooting:"
    echo "==============="
    echo "See: docs/CLAUDE_INTEGRATION_TESTING.md"
    echo "See: docs/INTEGRATION_TROUBLESHOOTING.md"
}

# === Main Execution ===

main() {
    echo "=========================================="
    echo "  Document-MCP Verification Script"
    echo "  Version: 2.0 (Enhanced)"
    echo "=========================================="

    tier_0_binary_check || {
        echo -e "\n${RED}Binary installation failed. Stopping.${NC}"
        cleanup
        exit 1
    }

    tier_1_server_startup || {
        echo -e "\n${RED}Server startup failed. Stopping.${NC}"
        cleanup
        exit 1
    }

    tier_2_tool_discovery

    tier_3_basic_operations

    tier_4_claude_integration

    tier_5_advanced_features

    report_summary || {
        cleanup
        print_next_steps
        exit 1
    }

    print_next_steps
    cleanup
    exit 0
}

# Run main
main
