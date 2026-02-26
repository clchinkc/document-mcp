# Claude Integration Testing Guide
## Claude Code & Claude Desktop MCP Verification Strategy

**Last Updated**: February 25, 2026
**Audience**: QA Engineers, Developers, Users
**Scope**: stdio-based MCP server integration with Claude clients

---

## Table of Contents
1. [Quick Start](#quick-start)
2. [Test Architecture](#test-architecture)
3. [Claude Code Testing](#claude-code-testing)
4. [Claude Desktop Testing](#claude-desktop-testing)
5. [Automated Verification](#automated-verification)
6. [Test Scenarios](#test-scenarios)
7. [Success Criteria](#success-criteria)
8. [Troubleshooting](#troubleshooting)

---

## Quick Start

### For Impatient Developers
```bash
# 1. Install package
pip install document-mcp

# 2. Run verification script (fastest check)
bash scripts/verify_mcp.sh

# 3. Test with Claude Code
claude mcp add document-mcp -s local -- document-mcp stdio
claude mcp list  # Should show ✓ Connected

# 4. Test with Claude Desktop
# Edit: ~/Library/Application Support/Claude/claude_desktop_config.json
# Add to mcpServers section (see examples below)
```

### Expected Output Timeline
- **Installation**: 30-60 seconds
- **Verification script**: 10-15 seconds
- **Claude Code integration**: 20-30 seconds
- **Claude Desktop restart**: 5-10 seconds
- **First tool execution**: 1-3 seconds (after connection)

---

## Test Architecture

### Component Stack
```
┌─────────────────────────────────────────┐
│  Claude Client (Code or Desktop)        │
│  • Request formatting                   │
│  • Response parsing                     │
│  • Tool discovery                       │
└──────────────┬──────────────────────────┘
               │
        stdio JSON-RPC 2.0
               │
┌──────────────▼──────────────────────────┐
│  MCP Server (document-mcp)              │
│  • Tool registration                    │
│  • JSON serialization                   │
│  • Error handling                       │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│  File System Operations                 │
│  • Document storage                     │
│  • Snapshot versioning                  │
│  • Metadata persistence                 │
└─────────────────────────────────────────┘
```

### Integration Points to Verify
1. **Binary in PATH**: Can Claude locate the `document-mcp` executable
2. **stdio Transport**: Does MCP JSON-RPC communication work bidirectionally
3. **Tool Discovery**: Can Claude list all 28 tools
4. **Tool Execution**: Can Claude call tools and get structured responses
5. **Error Handling**: Does Claude handle MCP errors gracefully
6. **State Persistence**: Do document changes persist between calls

### Test Tiers

| Tier | Scope | Speed | Dependencies | Result |
|------|-------|-------|--------------|--------|
| **Tier 0** | Binary availability | <1s | None | Can execute `document-mcp` |
| **Tier 1** | Server startup | <2s | Python 3.10+ | Server starts without errors |
| **Tier 2** | Tool discovery | <3s | Real MCP connection | Claude sees all 28 tools |
| **Tier 3** | Basic operations | <5s | Server + File system | CRUD operations work |
| **Tier 4** | Complex workflows | <30s | Server + File system + LLM | Multi-step sequences succeed |
| **Tier 5** | Error recovery | <10s | Server + Error injection | Errors handled cleanly |

---

## Claude Code Testing

### Prerequisites
- Claude Code installed and `claude` CLI available
- Python 3.10+ with `pip` access
- Bash shell or equivalent

### Setup Steps

#### Step 1: Install document-mcp
```bash
# Option A: From PyPI (recommended)
pip install document-mcp

# Option B: From source (development)
git clone https://github.com/your-org/document-mcp.git
cd document-mcp
pip install -e .

# Verify installation
which document-mcp
document-mcp --version  # Should show version number
```

#### Step 2: Add MCP Server to Claude Code
```bash
# Add document-mcp as local MCP server
claude mcp add document-mcp -s local -- document-mcp stdio

# List connected servers
claude mcp list

# Expected output:
# document-mcp (local)
#   Status: ✓ Connected
#   Tools: 28
```

#### Step 3: Verify Tool Discovery
```bash
# Use Claude Code to list tools
claude mcp tools document-mcp

# Or interactively:
claude -i
# Then ask: "What tools are available?"
```

### Manual Testing in Claude Code

#### Test 1: Tool List Discovery
```
User: What tools are available?
Expected: List of 28 tools with descriptions
```

#### Test 2: Create Document
```
User: Create a document called "TestBook" and a chapter called "01-Introduction"
Expected: Document and chapter created in .documents_storage/
```

#### Test 3: Write Content
```
User: In TestBook, add a paragraph to chapter 01-Introduction with "Hello World"
Expected: Paragraph appears in file
```

#### Test 4: Read Content
```
User: List all documents and their chapters
Expected: Structured list showing TestBook -> 01-Introduction
```

#### Test 5: Error Handling
```
User: Delete a document that doesn't exist
Expected: Clear error message (not crash)
```

### Claude Code CLI Integration Points

**Configuration Location**:
```bash
~/.claude/mcp_config.json  # Claude Code MCP configuration
```

**Diagnostic Commands**:
```bash
# Check MCP server status
claude mcp status document-mcp

# View MCP server logs
claude mcp logs document-mcp

# Test specific tool
claude mcp call document-mcp list_documents

# Remove and re-add if needed
claude mcp remove document-mcp
claude mcp add document-mcp -s local -- document-mcp stdio
```

---

## Claude Desktop Testing

### Prerequisites
- Claude Desktop installed (macOS, Windows, or Linux)
- Access to `~/Library/Application Support/Claude/` (macOS) or equivalent
- `document-mcp` binary in PATH or know full path

### Setup Steps

#### Step 1: Locate Configuration File
```bash
# macOS
~/Library/Application\ Support/Claude/claude_desktop_config.json

# Windows
%APPDATA%\Claude\claude_desktop_config.json

# Linux
~/.config/Claude/claude_desktop_config.json
```

#### Step 2: Update Configuration

**Option A: Binary in PATH** (recommended)
```json
{
  "mcpServers": {
    "document-mcp": {
      "command": "document-mcp",
      "args": ["stdio"],
      "env": {
        "DOCUMENT_STORAGE_PATH": "/path/to/documents"
      }
    }
  }
}
```

**Option B: Full Path to Binary**
```bash
# Find full path
which document-mcp
# Output: /usr/local/bin/document-mcp
```

```json
{
  "mcpServers": {
    "document-mcp": {
      "command": "/usr/local/bin/document-mcp",
      "args": ["stdio"],
      "env": {}
    }
  }
}
```

**Option C: Virtual Environment**
```json
{
  "mcpServers": {
    "document-mcp": {
      "command": "/path/to/venv/bin/python",
      "args": ["-m", "document_mcp.doc_tool_server", "stdio"],
      "env": {}
    }
  }
}
```

#### Step 3: Restart Claude Desktop
```bash
# macOS: Force quit and restart
killall Claude 2>/dev/null || true
open /Applications/Claude.app

# Windows: Restart via taskbar
# Linux: Standard app restart
```

#### Step 4: Verify Connection
In Claude Desktop chat:
- Ask: "What MCP servers are available?"
- Or: "List all documents"
- Expected: Tools available and working

### Manual Testing in Claude Desktop

#### Test Flow: Create and Manage a Story

**Step 1: Create Document**
```
You: Create a new document called "MyStory"
Claude: [Uses create_document tool]
Result: Success message with document created
```

**Step 2: Create Chapter**
```
You: Add a chapter called "Chapter 1" to MyStory with intro text
Claude: [Uses create_chapter tool]
Result: Chapter file created with metadata
```

**Step 3: Add Content**
```
You: Add a scene to Chapter 1 about "The meeting"
Claude: [Uses add_paragraph or replace_paragraph tool]
Result: Content appended or replaced in chapter
```

**Step 4: Search Content**
```
You: Find all mentions of "meeting" in MyStory
Claude: [Uses find_text tool]
Result: List of matches with context
```

**Step 5: Create Summary**
```
You: Write a summary of Chapter 1
Claude: [Uses write_summary tool]
Result: Summary file created in summaries/ directory
```

### Desktop Configuration Validation

**Check 1: Configuration Syntax**
```bash
# Validate JSON
python3 -m json.tool ~/.config/Claude/claude_desktop_config.json
```

**Check 2: Binary Accessibility**
```bash
# Verify command resolves
which document-mcp

# Verify permissions
ls -la /path/to/document-mcp
# Should show: -rwxr-xr-x (executable)

# Verify execution
document-mcp --help | head -5
```

**Check 3: Test stdio Transport**
```bash
# Manual stdio test (see Tier 1 verification)
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}' | document-mcp stdio | head -c 200
```

---

## Automated Verification

### Tier 0: Binary Installation Check
```bash
#!/bin/bash
# Quick verification that binary is installed and in PATH

check_binary_in_path() {
    if command -v document-mcp &> /dev/null; then
        echo "✓ PASS: document-mcp found in PATH"
        return 0
    else
        echo "✗ FAIL: document-mcp not in PATH"
        echo "  Run: pip install document-mcp"
        return 1
    fi
}

check_binary_executable() {
    if document-mcp --help > /dev/null 2>&1; then
        echo "✓ PASS: document-mcp is executable"
        return 0
    else
        echo "✗ FAIL: document-mcp not executable"
        return 1
    fi
}

check_version() {
    local version=$(document-mcp --version 2>/dev/null || echo "unknown")
    echo "ℹ INFO: Version $version"
}

check_binary_in_path && check_binary_executable && check_version
```

### Tier 1: Server Startup Check
```bash
#!/bin/bash
# Verify MCP server starts successfully

check_server_startup() {
    # Start server in background with timeout
    timeout 3 document-mcp stdio > /tmp/mcp_startup.log 2>&1 &
    local pid=$!

    sleep 1

    # Check if process is still running
    if ps -p $pid > /dev/null 2>&1; then
        kill $pid 2>/dev/null
        echo "✓ PASS: Server started successfully"
        return 0
    else
        echo "✗ FAIL: Server exited immediately"
        cat /tmp/mcp_startup.log
        return 1
    fi
}

check_server_startup
```

### Tier 2: Tool Discovery Check
```python
#!/usr/bin/env python3
"""Verify all 28 tools are discoverable via MCP."""

import asyncio
import json
import subprocess
import sys
from typing import Any

async def test_tool_discovery() -> bool:
    """Test that all tools are discoverable."""

    # Start server
    proc = subprocess.Popen(
        ["document-mcp", "stdio"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Send initialization request
    init_request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test-discovery", "version": "1.0"},
        },
    }

    try:
        # Write and read response
        proc.stdin.write(json.dumps(init_request) + "\n")
        proc.stdin.flush()

        response_line = proc.stdout.readline()
        if not response_line:
            print("✗ FAIL: No response from server")
            return False

        response = json.loads(response_line)

        # Check for tools list
        if "result" not in response:
            print(f"✗ FAIL: Unexpected response: {response}")
            return False

        tools = response.get("result", {}).get("tools", [])

        # Expected tool count: 28
        expected_tools = {
            "list_documents", "create_document", "delete_document",
            "read_summary", "write_summary", "list_summaries",
            "list_chapters", "create_chapter", "delete_chapter",
            "write_chapter_content",
            "add_paragraph", "replace_paragraph", "delete_paragraph",
            "move_paragraph",
            "read_content", "find_text", "replace_text",
            "get_statistics", "find_similar_text", "find_entity",
            "read_metadata", "write_metadata", "list_metadata",
            "manage_snapshots", "check_content_status", "diff_content",
            "get_document_outline", "search_tool",
        }

        discovered_tools = {t["name"] for t in tools}
        missing = expected_tools - discovered_tools

        if missing:
            print(f"✗ FAIL: Missing {len(missing)} tools: {missing}")
            return False

        print(f"✓ PASS: All {len(discovered_tools)} tools discovered")
        return True

    finally:
        proc.terminate()
        proc.wait(timeout=5)

if __name__ == "__main__":
    result = asyncio.run(test_tool_discovery())
    sys.exit(0 if result else 1)
```

### Tier 3: Basic Operations Check
```python
#!/usr/bin/env python3
"""Verify CRUD operations work correctly."""

import asyncio
import json
import subprocess
import tempfile
import os
from pathlib import Path

async def test_crud_operations():
    """Test Create, Read, Update, Delete operations."""

    # Create temporary storage
    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ["DOCUMENT_STORAGE_PATH"] = tmpdir

        proc = subprocess.Popen(
            ["document-mcp", "stdio"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        try:
            # Initialize
            init_msg = {
                "jsonrpc": "2.0", "id": 1, "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05", "capabilities": {},
                    "clientInfo": {"name": "crud-test", "version": "1.0"}
                }
            }
            proc.stdin.write(json.dumps(init_msg) + "\n")
            proc.stdin.flush()

            # Read init response
            proc.stdout.readline()

            # Test 1: Create document
            create_msg = {
                "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {
                    "name": "create_document",
                    "arguments": {"document_name": "TestDoc"}
                }
            }
            proc.stdin.write(json.dumps(create_msg) + "\n")
            proc.stdin.flush()
            response = json.loads(proc.stdout.readline())

            if response.get("error"):
                print(f"✗ FAIL: Create failed: {response['error']}")
                return False

            print("✓ PASS: Create document")

            # Test 2: Verify file created
            doc_path = Path(tmpdir) / "TestDoc"
            if not doc_path.exists():
                print("✗ FAIL: Document directory not created")
                return False

            print("✓ PASS: Document persisted to filesystem")

            # Test 3: List documents
            list_msg = {
                "jsonrpc": "2.0", "id": 3, "method": "tools/call",
                "params": {
                    "name": "list_documents",
                    "arguments": {}
                }
            }
            proc.stdin.write(json.dumps(list_msg) + "\n")
            proc.stdin.flush()
            response = json.loads(proc.stdout.readline())

            if response.get("error"):
                print(f"✗ FAIL: List failed: {response['error']}")
                return False

            print("✓ PASS: List documents")

            return True

        finally:
            proc.terminate()
            proc.wait(timeout=5)

if __name__ == "__main__":
    result = asyncio.run(test_crud_operations())
    import sys
    sys.exit(0 if result else 1)
```

---

## Test Scenarios

### Scenario 1: Single Tool Execution
**Objective**: Verify Claude can call individual tools successfully

**Steps**:
1. Create document: `create_document(name="Test1")`
2. Verify response structure: `{"success": true, "message": "..."}`
3. Check file system: Directory exists at `.documents_storage/Test1/`
4. **Success Criteria**: File created, response valid, no errors

### Scenario 2: Multi-Step Document Creation Workflow
**Objective**: Verify sequential tool calls maintain state

**Steps**:
1. Create document: `create_document(name="Story")`
2. Create chapter: `create_chapter(document="Story", name="01-intro", content="...")`
3. Add paragraph: `add_paragraph(document="Story", chapter="01-intro", text="...")`
4. Read back: `read_content(document="Story", chapter="01-intro")`
5. **Success Criteria**: All operations succeed, content matches in step 4

### Scenario 3: Content Discovery and Search
**Objective**: Verify search and discovery tools work

**Steps**:
1. Create document with chapters and content
2. Search: `find_text(document="Doc", query="keyword")`
3. Get statistics: `get_statistics(document="Doc")`
4. Get outline: `get_document_outline(document="Doc")`
5. **Success Criteria**: Correct results, accurate counts, proper structure

### Scenario 4: Version Control and Snapshots
**Objective**: Verify snapshot system protects changes

**Steps**:
1. Create document and add content
2. Modify content: `replace_paragraph(...)`
3. Check snapshots: `manage_snapshots(document="Doc", action="list")`
4. Restore: `manage_snapshots(document="Doc", action="restore", snapshot_id="...")`
5. **Success Criteria**: Snapshots created, restore works, version control intact

### Scenario 5: Error Conditions
**Objective**: Verify graceful error handling

**Steps**:
1. Delete non-existent document: Should return error, not crash
2. Invalid parameters: Should return validation error
3. Concurrent modifications: Should handle gracefully
4. File system errors: Should report clearly
5. **Success Criteria**: Clear errors, server continues running

### Scenario 6: Large Content Handling
**Objective**: Verify pagination for large documents

**Steps**:
1. Create document with 100KB+ content
2. Read with pagination: `read_content(..., page=1, page_size=50000)`
3. Navigate pages: page 1, 2, 3 with pagination info
4. **Success Criteria**: Content returns in pages, no truncation

---

## Success Criteria

### Installation Success
- [ ] `document-mcp` binary is in PATH
- [ ] `document-mcp --version` returns version number
- [ ] `document-mcp --help` shows command options
- [ ] `pip show document-mcp` shows package details

### Claude Code Integration Success
- [ ] `claude mcp list` shows document-mcp as connected
- [ ] `claude mcp tools document-mcp` lists all 28 tools
- [ ] Claude Code can call tools without timeout
- [ ] Tool responses are structured and valid
- [ ] Changes persist in file system

### Claude Desktop Integration Success
- [ ] Configuration JSON is valid (parseable)
- [ ] Claude Desktop restarts without errors
- [ ] "Mcp" section shows in settings
- [ ] Document-mcp server shows as connected
- [ ] Can execute tools in chat
- [ ] Changes persist between sessions

### Operational Success
- [ ] All 28 tools are discoverable
- [ ] CRUD operations work (Create, Read, Update, Delete)
- [ ] Multi-step workflows succeed
- [ ] Search and discovery work correctly
- [ ] Version control prevents data loss
- [ ] Errors are handled gracefully
- [ ] Large content handles pagination
- [ ] Performance is acceptable (<5s per tool call)

### Acceptance Test Checklist
- [ ] Tier 0-5 tests pass
- [ ] All tool scenarios execute
- [ ] Error scenarios handled
- [ ] Multi-session state persists
- [ ] Performance within limits
- [ ] Documentation matches behavior
- [ ] No data loss scenarios

---

## Troubleshooting

For detailed troubleshooting matrix, see [INTEGRATION_TROUBLESHOOTING.md](./INTEGRATION_TROUBLESHOOTING.md).

### Common Issues Quick Reference

| Issue | Quick Check | Solution |
|-------|------------|----------|
| "spawn document-mcp ENOENT" | `which document-mcp` | Use full path in config |
| "Connection timeout" | `document-mcp stdio --help` | Binary missing or not executable |
| "Module not found" | `python3 -c "import document_mcp"` | Reinstall: `pip install --force-reinstall document-mcp` |
| "Permission denied" | `ls -la /path/to/document-mcp` | `chmod +x /path/to/document-mcp` |
| "No tools available" | Check server startup | Restart Claude and mcp-add |
| "File not found" errors | Check `DOCUMENT_STORAGE_PATH` | Set env var or use default |

---

## Next Steps

### For New Users
1. Run `bash scripts/verify_mcp.sh`
2. Add to Claude Code with `claude mcp add`
3. Test basic operations in chat
4. See [INTEGRATION_TROUBLESHOOTING.md](./INTEGRATION_TROUBLESHOOTING.md) if issues

### For CI/CD Integration
1. See [tests/integration/test_mcp_claude_integration.py](../tests/integration/test_mcp_claude_integration.py)
2. Use test scenarios from above
3. Validate all 5 tiers pass

### For Production Deployment
1. Complete all acceptance tests
2. Document any custom configuration
3. See [docs/DEPLOYMENT.md](./DEPLOYMENT.md) for hosting options
4. Monitor with observability tools

---

## References

- [MCP Specification](https://modelcontextprotocol.io/)
- [FastMCP Documentation](https://github.com/joshrosenhanst/fastmcp)
- [Claude Code Documentation](https://claude.ai/docs)
- [INTEGRATION_TROUBLESHOOTING.md](./INTEGRATION_TROUBLESHOOTING.md)
