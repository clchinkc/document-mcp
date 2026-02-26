# Claude Integration Deployment Guide
## Production-Ready MCP Server Setup and Validation

**Last Updated**: February 25, 2026
**Scope**: Deployment to Claude Code and Claude Desktop
**Audience**: QA Engineers, Release Managers, System Administrators

---

## Table of Contents
1. [Pre-Deployment Checklist](#pre-deployment-checklist)
2. [Installation Methods](#installation-methods)
3. [Configuration Management](#configuration-management)
4. [Validation Procedures](#validation-procedures)
5. [Monitoring and Observability](#monitoring-and-observability)
6. [Rollback Procedures](#rollback-procedures)
7. [Release Notes](#release-notes)

---

## Pre-Deployment Checklist

### Code Quality Gates

Before deploying to production, ensure all items pass:

- [ ] All unit tests pass: `uv run pytest tests/unit/ -v`
- [ ] All integration tests pass: `uv run pytest tests/integration/ -v`
- [ ] All Tier 0-5 verification tests pass: `bash scripts/verify_mcp_enhanced.sh`
- [ ] Code coverage >= 60%: `uv run pytest --cov=document_mcp`
- [ ] No ruff lint errors: `uv run ruff check document_mcp/`
- [ ] No mypy errors: `uv run mypy document_mcp/`
- [ ] All tools discoverable (28 tools)
- [ ] No data loss in snapshot tests

### Performance Benchmarks

- [ ] Tool execution time <= 1s for CRUD operations
- [ ] Tool execution time <= 5s for complex operations
- [ ] Server startup time <= 2s
- [ ] Memory usage <= 100MB baseline
- [ ] No memory leaks during sustained use

### Compatibility Testing

- [ ] Tested with Claude Code CLI (latest version)
- [ ] Tested with Claude Desktop (latest version)
- [ ] Tested on macOS (Intel and Apple Silicon)
- [ ] Tested on Linux (Ubuntu 20.04+)
- [ ] Tested on Windows 10/11
- [ ] Python 3.10+ requirement met

### Documentation Quality

- [ ] README.md is current and accurate
- [ ] CLAUDE_INTEGRATION_TESTING.md complete and tested
- [ ] INTEGRATION_TROUBLESHOOTING.md covers known issues
- [ ] API documentation generated
- [ ] Example code works end-to-end
- [ ] Troubleshooting matrix tested

---

## Installation Methods

### Method 1: PyPI Package (Recommended for Users)

**Prerequisites**:
- Python 3.10 or higher
- pip package manager

**Installation**:
```bash
# Install latest version
pip install document-mcp

# Verify installation
document-mcp --version
which document-mcp
```

**Verification**:
```bash
# Test server startup
document-mcp stdio < /dev/null &
sleep 1
ps aux | grep document-mcp
kill %1
```

**Uninstall**:
```bash
pip uninstall -y document-mcp
```

### Method 2: From Source (Development)

**Prerequisites**:
- Git
- Python 3.10+
- uv package manager

**Installation**:
```bash
# Clone repository
git clone https://github.com/your-org/document-mcp.git
cd document-mcp

# Install dependencies
uv sync --all-extras

# Install in development mode
uv pip install -e .

# Verify installation
document-mcp --version
```

**Verification**:
```bash
# Run full test suite
uv run pytest tests/integration/ tests/unit/ -v

# Run verification script
bash scripts/verify_mcp_enhanced.sh
```

### Method 3: Virtual Environment (Recommended for CI/CD)

**Setup**:
```bash
# Create and activate virtual environment
python3 -m venv document_mcp_env
source document_mcp_env/bin/activate

# Install package
pip install document-mcp

# Configure Claude Code to use venv
claude mcp add document-mcp -s local -- \
  "$(pwd)/document_mcp_env/bin/python" \
  -m document_mcp.doc_tool_server stdio
```

**Verification**:
```bash
# Verify venv binary works
./document_mcp_env/bin/python -m document_mcp.doc_tool_server --help
```

### Method 4: Docker Container (Advanced)

**Dockerfile**:
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install document-mcp
RUN pip install --no-cache-dir document-mcp

# Expose metrics port (optional)
EXPOSE 8001

# Run server in stdio mode (for MCP)
ENTRYPOINT ["document-mcp"]
CMD ["stdio"]
```

**Build and Test**:
```bash
# Build image
docker build -t document-mcp:latest .

# Test container
docker run --rm document-mcp:latest --version

# Run server
docker run -it document-mcp:latest stdio
```

---

## Configuration Management

### Environment Variables

#### Standard Configuration
```bash
# Document storage path (default: .documents_storage)
export DOCUMENT_STORAGE_PATH="/path/to/documents"

# Storage backend (default: local)
export DOCUMENT_STORAGE_BACKEND="local"  # or "gcs"

# Automatic snapshots (default: true)
export ENABLE_AUTOMATIC_SNAPSHOTS="true"

# MCP observability (default: true)
export MCP_OBSERVABILITY_ENABLED="true"

# Request timeout in seconds (default: 30)
export MCP_TIMEOUT="30"
```

#### GCP Integration (Optional)
```bash
# Cloud Logging integration
export GOOGLE_CLOUD_PROJECT="your-project-id"
export MCP_ENABLE_GCP_LOGGING="true"

# Cloud Trace integration
export MCP_ENABLE_GCP_TRACE="true"

# Cloud Monitoring integration
export MCP_ENABLE_GCP_MONITORING="true"
```

#### Debugging
```bash
# Enable debug logging
export DEBUG="1"

# Enable verbose output
export VERBOSE="1"

# Log to file
export MCP_LOG_FILE="/var/log/document-mcp.log"
```

### Claude Code Configuration

**Location**: `~/.claude/mcp_config.json`

**Basic Setup**:
```bash
# Add document-mcp to Claude Code
claude mcp add document-mcp -s local -- document-mcp stdio

# Verify
claude mcp list
```

**Advanced Setup with Environment Variables**:
```bash
# Create wrapper script
cat > /usr/local/bin/document-mcp-prod << 'EOF'
#!/bin/bash
export DOCUMENT_STORAGE_PATH="/data/documents"
export MCP_OBSERVABILITY_ENABLED="true"
exec document-mcp stdio
EOF

chmod +x /usr/local/bin/document-mcp-prod

# Add with wrapper
claude mcp add document-mcp -s local -- /usr/local/bin/document-mcp-prod
```

### Claude Desktop Configuration

**Location**:
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`
- Linux: `~/.config/Claude/claude_desktop_config.json`

**Basic Configuration**:
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

**Production Configuration**:
```json
{
  "mcpServers": {
    "document-mcp": {
      "command": "/usr/local/bin/document-mcp",
      "args": ["stdio"],
      "env": {
        "DOCUMENT_STORAGE_PATH": "/data/documents",
        "MCP_OBSERVABILITY_ENABLED": "true",
        "MCP_TIMEOUT": "30",
        "GOOGLE_CLOUD_PROJECT": "your-project-id"
      }
    }
  }
}
```

**Multi-Server Setup**:
```json
{
  "mcpServers": {
    "document-mcp-dev": {
      "command": "document-mcp",
      "args": ["stdio"],
      "env": {
        "DOCUMENT_STORAGE_PATH": "/dev/documents"
      }
    },
    "document-mcp-prod": {
      "command": "document-mcp",
      "args": ["stdio"],
      "env": {
        "DOCUMENT_STORAGE_PATH": "/prod/documents",
        "MCP_OBSERVABILITY_ENABLED": "true"
      }
    }
  }
}
```

---

## Validation Procedures

### Tier 0: Binary Validation

```bash
#!/bin/bash
# Validate binary installation and accessibility

echo "Checking binary installation..."
which document-mcp || { echo "FAIL: Not in PATH"; exit 1; }
test -x $(which document-mcp) || { echo "FAIL: Not executable"; exit 1; }
document-mcp --version || { echo "FAIL: Version check failed"; exit 1; }
echo "✓ Binary validation passed"
```

### Tier 1: Server Validation

```bash
#!/bin/bash
# Validate server startup and basic operation

echo "Testing server startup..."
timeout 3 document-mcp stdio < /dev/null > /tmp/server_test.log 2>&1 &
SERVER_PID=$!
sleep 1

if ps -p $SERVER_PID > /dev/null; then
    echo "✓ Server started"
    kill $SERVER_PID
else
    echo "FAIL: Server didn't start"
    cat /tmp/server_test.log
    exit 1
fi
```

### Tier 2: Tool Discovery Validation

```python
#!/usr/bin/env python3
"""Validate all tools are discoverable."""

import asyncio
import sys
from pydantic_ai.mcp import MCPServerStdio

async def validate_tools():
    server = MCPServerStdio(
        command="document-mcp",
        args=["stdio"],
        timeout=10.0,
    )

    async with server as s:
        tools = await s._client.list_tools()
        if len(tools) != 28:
            print(f"FAIL: Expected 28 tools, got {len(tools)}")
            return False

        print(f"✓ All {len(tools)} tools discovered")
        return True

result = asyncio.run(validate_tools())
sys.exit(0 if result else 1)
```

### Tier 3: Operation Validation

```bash
#!/bin/bash
# Validate CRUD operations work

TEST_DIR="/tmp/document_mcp_validation_$$"
mkdir -p "$TEST_DIR"

export DOCUMENT_STORAGE_PATH="$TEST_DIR"

python3 << 'EOF' || exit 1
import asyncio
from pydantic_ai.mcp import MCPServerStdio
import json

async def test_operations():
    server = MCPServerStdio(
        command="document-mcp",
        args=["stdio"],
        timeout=10.0,
    )

    async with server as s:
        # Create
        result = await s._client.call_tool(
            "create_document",
            {"document_name": "ValidationTest"}
        )
        assert json.loads(result.content[0].text)["success"]

        # List
        result = await s._client.call_tool("list_documents", {})
        assert "ValidationTest" in result.content[0].text

    print("✓ CRUD operations validated")

asyncio.run(test_operations())
EOF

rm -rf "$TEST_DIR"
```

### Tier 4: Integration Validation

```bash
#!/bin/bash
# Validate Claude integration

if command -v claude &>/dev/null; then
    echo "Checking Claude Code integration..."
    claude mcp list | grep -q "document-mcp" || {
        echo "WARNING: document-mcp not in Claude Code"
        echo "Register with: claude mcp add document-mcp -s local -- document-mcp stdio"
    }
fi

DESKTOP_CONFIG="$HOME/Library/Application Support/Claude/claude_desktop_config.json"
if [ -f "$DESKTOP_CONFIG" ]; then
    echo "Checking Claude Desktop configuration..."
    grep -q "document-mcp" "$DESKTOP_CONFIG" && echo "✓ Configured" || {
        echo "INFO: Not configured in Claude Desktop"
    }
fi
```

### Complete Validation Script

```bash
#!/bin/bash
# Run all tiers of validation

echo "========== Document-MCP Deployment Validation =========="

# Tier 0
bash scripts/verify_mcp_enhanced.sh || exit 1

# Tier 1-5
uv run pytest tests/integration/test_mcp_claude_integration.py -v || exit 1

echo "========== Validation Successful =========="
```

---

## Monitoring and Observability

### Logging

**Log Locations**:
- `document_mcp/doc_operations.log` - Tool operations
- `document_mcp/mcp_calls.log` - MCP protocol messages
- `document_mcp/errors.log` - Error events

**Log Format**:
```json
{
  "timestamp": "2026-02-25T12:34:56.789Z",
  "level": "INFO",
  "tool": "create_document",
  "user": "user_id",
  "status": "success",
  "duration_ms": 45
}
```

**Monitor Logs**:
```bash
# Watch operations in real-time
tail -f document_mcp/doc_operations.log | jq .

# Count operations by type
grep "tool" document_mcp/doc_operations.log | jq '.tool' | sort | uniq -c

# Find errors
grep "ERROR\|error" document_mcp/errors.log
```

### Metrics

**Prometheus Endpoint**: `http://localhost:8001/metrics`

**Key Metrics**:
- `document_mcp_tool_calls_total` - Total tools called
- `document_mcp_tool_duration_seconds` - Tool execution time
- `document_mcp_errors_total` - Total errors
- `document_mcp_documents_total` - Total documents

**Scrape Configuration**:
```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'document-mcp'
    static_configs:
      - targets: ['localhost:8001']
    metrics_path: '/metrics'
    scrape_interval: 15s
```

### Health Checks

**Liveness Check**:
```bash
# Server is running
curl -f http://localhost:8001/metrics || exit 1
```

**Readiness Check**:
```bash
# Server responds to MCP calls
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"health","version":"1.0"}}}' | \
  timeout 5 document-mcp stdio | \
  grep -q "result" || exit 1
```

### Alerting

**Alert Rules** (Prometheus):
```yaml
groups:
  - name: document_mcp
    rules:
      - alert: HighErrorRate
        expr: rate(document_mcp_errors_total[5m]) > 0.1
        for: 5m
        annotations:
          summary: "High error rate in document-mcp"

      - alert: SlowToolExecution
        expr: histogram_quantile(0.95, document_mcp_tool_duration_seconds) > 5
        for: 5m
        annotations:
          summary: "Tool execution times degraded"
```

---

## Rollback Procedures

### Quick Rollback

```bash
#!/bin/bash
# Immediate rollback to previous version

VERSION_TO_RESTORE="0.0.3"

echo "Rolling back to version $VERSION_TO_RESTORE..."

# Uninstall current
pip uninstall -y document-mcp

# Install previous version
pip install document-mcp==$VERSION_TO_RESTORE

# Verify
document-mcp --version

echo "Rollback complete. Restart Claude to reconnect."
```

### Configuration Rollback

```bash
#!/bin/bash
# Restore previous configuration

BACKUP_DIR="$HOME/.claude_mcp_backups"
LATEST_BACKUP=$(ls -t "$BACKUP_DIR"/*.json | head -1)

echo "Restoring from: $LATEST_BACKUP"

if [ -f "$LATEST_BACKUP" ]; then
    cp "$LATEST_BACKUP" \
       "$HOME/Library/Application Support/Claude/claude_desktop_config.json"
    echo "Configuration restored"
else
    echo "No backup found"
    exit 1
fi
```

### Data Recovery

```bash
#!/bin/bash
# Restore documents from snapshots

DOCUMENT="MyDocument"
SNAPSHOT_ID="snapshot_20260225_120000"

echo "Recovering $DOCUMENT from snapshot $SNAPSHOT_ID..."

SNAPSHOT_PATH=".documents_storage/$DOCUMENT/.snapshots/$SNAPSHOT_ID"

if [ -d "$SNAPSHOT_PATH" ]; then
    cp "$SNAPSHOT_PATH"/content.md \
       ".documents_storage/$DOCUMENT/01-recovered.md"
    echo "Document recovered to 01-recovered.md"
else
    echo "Snapshot not found"
    exit 1
fi
```

---

## Release Notes Template

### Version X.X.X - Release Date

**What's New**:
- [ ] New feature A (describe benefit)
- [ ] New feature B (describe benefit)

**Improvements**:
- [ ] Performance improvement A (metric)
- [ ] Compatibility improvement B

**Bug Fixes**:
- [ ] Fixed issue #123 (brief description)
- [ ] Fixed issue #124 (brief description)

**Known Issues**:
- [ ] Issue A with workaround
- [ ] Issue B with workaround

**Breaking Changes**:
- [ ] None / List any breaking changes

**Migration Guide**:
```bash
# Upgrade
pip install --upgrade document-mcp

# Restart Claude
claude mcp remove document-mcp
claude mcp add document-mcp -s local -- document-mcp stdio
```

**Support**:
- Documentation: https://github.com/your-org/document-mcp/docs
- Issues: https://github.com/your-org/document-mcp/issues
- Discussions: https://github.com/your-org/document-mcp/discussions

---

## Deployment Checklist

### Pre-Release
- [ ] All tests pass
- [ ] Performance benchmarks met
- [ ] Documentation updated
- [ ] Release notes written
- [ ] Version bumped
- [ ] Changelog updated

### Release Day
- [ ] Tag release in git
- [ ] Build artifacts
- [ ] Publish to PyPI
- [ ] Create GitHub release
- [ ] Update documentation site
- [ ] Announce in channels

### Post-Release
- [ ] Monitor error rates
- [ ] Respond to feedback
- [ ] Plan hot fixes if needed
- [ ] Prepare patch release if needed

---

## References

- [CLAUDE_INTEGRATION_TESTING.md](./CLAUDE_INTEGRATION_TESTING.md)
- [INTEGRATION_TROUBLESHOOTING.md](./INTEGRATION_TROUBLESHOOTING.md)
- [README.md](../README.md)
- [GitHub Releases](https://github.com/your-org/document-mcp/releases)
