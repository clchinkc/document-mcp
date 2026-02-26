# Claude Integration Troubleshooting Matrix
## Comprehensive Diagnostic Guide for MCP Integration Issues

**Last Updated**: February 25, 2026
**Scope**: Claude Code and Claude Desktop MCP integration
**Severity Levels**: CRITICAL | HIGH | MEDIUM | LOW | INFO

---

## Issue Categories

- [Installation Issues](#installation-issues)
- [Binary/Command Issues](#binarycommand-issues)
- [Connection Issues](#connection-issues)
- [Tool Discovery Issues](#tool-discovery-issues)
- [Tool Execution Issues](#tool-execution-issues)
- [File System Issues](#file-system-issues)
- [Configuration Issues](#configuration-issues)
- [Performance Issues](#performance-issues)
- [Crash Issues](#crash-issues)
- [Data Integrity Issues](#data-integrity-issues)

---

## Installation Issues

### Issue: "ModuleNotFoundError: No module named 'document_mcp'"
**Severity**: CRITICAL
**When It Occurs**: During server startup or import

**Root Causes**:
- Package not installed
- Wrong Python environment
- Incompatible Python version
- Corrupted installation

**Diagnosis**:
```bash
# Check if installed
pip show document-mcp

# Check Python version (needs 3.10+)
python3 --version

# Check import directly
python3 -c "import document_mcp; print(document_mcp.__version__)"

# Check where it's installed
python3 -c "import document_mcp; print(document_mcp.__file__)"
```

**Solutions** (in order):
1. **Fresh install**:
   ```bash
   pip uninstall -y document-mcp
   pip cache purge
   pip install document-mcp
   ```

2. **Verify Python version**:
   ```bash
   # If using python alias, try python3 explicitly
   python3 -m pip install document-mcp
   python3 -m document_mcp.doc_tool_server --help
   ```

3. **Install to user directory**:
   ```bash
   pip install --user document-mcp
   ```

4. **Install from source** (development):
   ```bash
   git clone https://github.com/your-org/document-mcp.git
   cd document-mcp
   pip install -e .
   ```

5. **Check system PATH conflicts**:
   ```bash
   # See all Python versions
   which -a python3

   # Install to specific Python
   /usr/bin/python3 -m pip install document-mcp
   ```

**Prevention**:
- Use `python3` explicitly (not `python`)
- Verify Python version meets requirement (3.10+)
- Consider using virtual environment

---

### Issue: "Requirement already satisfied, but version mismatch"
**Severity**: HIGH
**When It Occurs**: After upgrade attempt

**Root Causes**:
- Old version not removed
- Multiple installations
- Version lock in pip cache
- System package conflict

**Diagnosis**:
```bash
# Check all installed versions
pip list | grep document-mcp

# Check pip cache
pip cache info

# Check both site-packages and user
find ~ -name "*document_mcp*" -type d
```

**Solutions**:
1. **Remove all traces**:
   ```bash
   pip uninstall -y document-mcp
   find ~/.local -type d -name "*document_mcp*" -exec rm -rf {} + 2>/dev/null
   find /usr -type d -name "*document_mcp*" -exec rm -rf {} + 2>/dev/null
   ```

2. **Clear pip cache and reinstall**:
   ```bash
   pip cache purge
   pip install --no-cache-dir --force-reinstall document-mcp
   ```

3. **Reinstall with specific version**:
   ```bash
   pip install --force-reinstall document-mcp==0.0.4
   ```

**Prevention**:
- Use `pip install --upgrade` for updates
- Use virtual environments to isolate versions

---

## Binary/Command Issues

### Issue: "spawn document-mcp ENOENT" (Claude Desktop/Code)
**Severity**: CRITICAL
**When It Occurs**: Claude tries to start server

**Root Causes**:
- Binary not in PATH
- Binary not executable
- Wrong binary path in config
- Binary deleted or moved
- Installation incomplete

**Diagnosis**:
```bash
# Check if binary exists
which document-mcp

# If not found, check where it should be
pip show -f document-mcp | grep Location

# Check binary permissions
ls -la $(which document-mcp 2>/dev/null || echo /usr/local/bin/document-mcp)

# Try running directly
document-mcp --help

# Try running via Python module
python3 -m document_mcp.doc_tool_server --help
```

**Solutions**:

1. **Add to PATH** (if Python Scripts not in PATH):
   ```bash
   # Find the bin directory
   pip show -f document-mcp | grep bin

   # Add to PATH (macOS/Linux)
   echo 'export PATH="/path/to/bin:$PATH"' >> ~/.zshrc
   source ~/.zshrc

   # Verify
   which document-mcp
   ```

2. **Make binary executable**:
   ```bash
   chmod +x $(which document-mcp)

   # Verify
   ls -la $(which document-mcp)
   ```

3. **Use full path in configuration** (if not in PATH):
   ```bash
   # Find full path
   FULL_PATH=$(python3 -c "import site; import os; \
    print(os.path.join(site.getsitepackages()[0], '../../bin/document-mcp'))")
   echo "Full path: $FULL_PATH"

   # Verify it works
   $FULL_PATH --help
   ```

4. **Configure Claude Desktop with full path**:
   ```json
   {
     "mcpServers": {
       "document-mcp": {
         "command": "/full/path/to/document-mcp",
         "args": ["stdio"],
         "env": {}
       }
     }
   }
   ```

5. **Use Python module method**:
   ```json
   {
     "mcpServers": {
       "document-mcp": {
         "command": "python3",
         "args": ["-m", "document_mcp.doc_tool_server", "stdio"],
         "env": {}
       }
     }
   }
   ```

**Prevention**:
- Install to system Python or virtual environment in PATH
- Verify `which document-mcp` returns a path before use
- Use full path in config if binary not in PATH

---

### Issue: "Permission denied" when running document-mcp
**Severity**: CRITICAL
**When It Occurs**: Executing binary directly

**Root Causes**:
- Binary not executable
- User doesn't have permission
- File mounted as noexec
- SELinux restrictions

**Diagnosis**:
```bash
# Check permissions
ls -la $(which document-mcp)

# Check if executable
test -x $(which document-mcp) && echo "Executable" || echo "Not executable"

# Check user
whoami

# Check group
id

# Check mount options
mount | grep $(which document-mcp | xargs dirname)
```

**Solutions**:
1. **Make executable**:
   ```bash
   chmod +x $(which document-mcp)
   ```

2. **Verify execution**:
   ```bash
   document-mcp --version
   ```

3. **Use Python module** (bypasses permission issues):
   ```bash
   python3 -m document_mcp.doc_tool_server --version
   ```

**Prevention**:
- Use virtual environment
- Install to user directory with `--user`
- Avoid mounting home directory with noexec

---

## Connection Issues

### Issue: "Connection timeout" (Claude Desktop/Code)
**Severity**: CRITICAL
**When It Occurs**: Claude tries to initialize MCP server

**Root Causes**:
- Server not starting
- Server crashes immediately
- Network/stdio connectivity issue
- Server hanging
- Configuration error

**Diagnosis**:
```bash
# Test basic server startup
timeout 3 document-mcp stdio 2>&1 | head -20

# Test with debug output
DEBUG=1 timeout 3 document-mcp stdio 2>&1 | head -20

# Test module import
python3 -c "from document_mcp.doc_tool_server import main; print('Import OK')"

# Check for errors
python3 -m document_mcp.doc_tool_server stdio --help
```

**Solutions**:

1. **Check server startup directly**:
   ```bash
   # Start server and try to interact
   (echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}'; sleep 1) | document-mcp stdio
   ```

2. **Verify environment variables**:
   ```bash
   # Check storage path exists or is writable
   mkdir -p "${DOCUMENT_STORAGE_PATH:-.documents_storage}"

   # Test with explicit path
   DOCUMENT_STORAGE_PATH=/tmp/test_docs command-mcp stdio
   ```

3. **Restart Claude and retry**:
   - Claude Desktop: Force quit and restart
   - Claude Code: `claude mcp remove document-mcp && claude mcp add document-mcp -s local -- document-mcp stdio`

4. **Check for port conflicts** (if using HTTP):
   ```bash
   lsof -i :3001
   # Kill if needed: kill -9 PID
   ```

5. **Use Python module method**:
   ```bash
   python3 -m document_mcp.doc_tool_server stdio < /dev/null
   ```

**Prevention**:
- Test server startup manually before adding to Claude
- Use timeout protection
- Verify file system writable

---

### Issue: "Connection refused" (specific error from Claude)
**Severity**: HIGH
**When It Occurs**: MCP client tries to connect

**Root Causes**:
- Server exited
- Server on wrong port
- Firewall blocking
- stdio not properly initialized

**Diagnosis**:
```bash
# Check if server is running
ps aux | grep document-mcp

# Check ports (if using HTTP)
netstat -tuln | grep 3001

# Test stdio directly
echo "test" | document-mcp stdio 2>&1
```

**Solutions**:
1. **Verify server stays running**:
   ```bash
   timeout 30 document-mcp stdio < /dev/null > /tmp/mcp.log 2>&1 &
   sleep 2
   ps aux | grep document-mcp
   ```

2. **Check firewall** (if using network):
   ```bash
   # macOS
   sudo lsof -i :3001

   # Check firewall status
   sudo defaults read /Library/Preferences/com.apple.alf globalstate
   ```

3. **Use stdio mode** (recommended):
   ```json
   {
     "mcpServers": {
       "document-mcp": {
         "command": "document-mcp",
         "args": ["stdio"],
         "env": {}
       }
     }
   }
   ```

**Prevention**:
- Use stdio transport (no network needed)
- Verify server stays running with tests
- Monitor Claude logs

---

## Tool Discovery Issues

### Issue: "No tools available" or empty tool list
**Severity**: HIGH
**When It Occurs**: Claude queries tool list

**Root Causes**:
- Server not fully initialized
- Tool registration failed
- Tool definitions corrupted
- Version mismatch

**Diagnosis**:
```bash
# Test tool discovery directly
python3 -c "
import asyncio
from pydantic_ai.mcp import MCPServerStdio
import sys

async def test():
    server = MCPServerStdio(command='document-mcp', args=['stdio'], timeout=10.0)
    async with server:
        tools = await server._client.list_tools()
        print(f'Found {len(tools)} tools')
        for t in tools[:5]:
            print(f'  - {t.name}')

asyncio.run(test())
"

# Or check raw MCP response
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}' | document-mcp stdio | jq '.result.tools | length'
```

**Solutions**:

1. **Restart MCP connection** (Claude Code):
   ```bash
   claude mcp remove document-mcp
   claude mcp add document-mcp -s local -- document-mcp stdio
   claude mcp list
   ```

2. **Restart Claude Desktop**:
   - macOS: `killall Claude; open /Applications/Claude.app`
   - Windows: Restart from taskbar
   - Linux: Standard app restart

3. **Verify tool registration**:
   ```bash
   python3 -c "
from document_mcp.doc_tool_server import mcp_server
# This will fail if tools not registered
print(f'Tools: {len(mcp_server._tools)}')
"
   ```

4. **Check version compatibility**:
   ```bash
   pip show document-mcp mcp pydantic-ai
   ```

**Prevention**:
- Verify tools discover after installation
- Test with `scripts/verify_mcp.sh`
- Keep dependencies up to date

---

### Issue: "Tool not found" for specific tool
**Severity**: MEDIUM
**When It Occurs**: Claude tries to call specific tool

**Root Causes**:
- Tool not registered
- Tool name typo
- Tool requires special setup
- Tool disabled/deprecated

**Diagnosis**:
```bash
# List all available tools
python3 -c "
import asyncio
from pydantic_ai.mcp import MCPServerStdio

async def list_tools():
    server = MCPServerStdio(command='document-mcp', args=['stdio'], timeout=10.0)
    async with server:
        tools = await server._client.list_tools()
        for t in tools:
            print(f'{t.name:25} - {t.description[:50]}...')

asyncio.run(list_tools())
"

# Search for specific tool
python3 -c "
import asyncio
from pydantic_ai.mcp import MCPServerStdio

async def find_tool(name):
    server = MCPServerStdio(command='document-mcp', args=['stdio'], timeout=10.0)
    async with server:
        tools = await server._client.list_tools()
        matches = [t for t in tools if name.lower() in t.name.lower()]
        if matches:
            for t in matches:
                print(f'Found: {t.name}')
        else:
            print(f'No tools matching {name}')

asyncio.run(find_tool('create'))
"
```

**Solutions**:
1. **Check tool name exactly**:
   - Use exact name from tool list
   - Tool names are case-sensitive
   - Common mistakes: `create_document` not `createDocument`

2. **Update tool descriptions**:
   ```bash
   python3 -m src.agents.shared.tool_descriptions
   ```

3. **Verify tool registration in code**:
   ```bash
   grep -r "def.*tool_name" document_mcp/tools/
   ```

**Prevention**:
- Use tool discovery before assuming tool exists
- Refer to documentation for exact names
- Test with `scripts/verify_mcp.sh`

---

## Tool Execution Issues

### Issue: "Tool execution failed" or empty response
**Severity**: MEDIUM
**When It Occurs**: Claude calls tool and waits for result

**Root Causes**:
- Tool implementation error
- Invalid parameters
- File system issue
- Timeout
- Resource exhaustion

**Diagnosis**:
```bash
# Test tool directly
python3 << 'EOF'
import asyncio
import json
from pydantic_ai.mcp import MCPServerStdio

async def test_tool():
    server = MCPServerStdio(command='document-mcp', args=['stdio'], timeout=10.0)
    async with server:
        # Test list_documents (should always work)
        result = await server._client.call_tool(
            "list_documents",
            {}
        )
        print(f"Response: {result}")

asyncio.run(test_tool())
EOF

# Check for errors in stdout
document-mcp stdio 2>&1 | head -50
```

**Solutions**:

1. **Test simplest tool first**:
   ```bash
   # list_documents should always succeed
   # If this fails, there's a core issue
   ```

2. **Check parameter types**:
   ```bash
   # Verify parameters match tool schema
   # Use tool description for correct format
   ```

3. **Increase timeout**:
   ```json
   {
     "mcpServers": {
       "document-mcp": {
         "command": "document-mcp",
         "args": ["stdio"],
         "env": {
           "MCP_TIMEOUT": "30"
         }
       }
     }
   }
   ```

4. **Check file system permissions**:
   ```bash
   # Verify storage directory writable
   ls -la .documents_storage/
   touch .documents_storage/test_write
   rm .documents_storage/test_write
   ```

5. **View error details**:
   ```bash
   # Run with debug
   DEBUG=1 python3 -m document_mcp.doc_tool_server stdio 2>&1 | head -100
   ```

**Prevention**:
- Validate parameters before calling
- Check file system before operations
- Monitor tool execution logs

---

### Issue: "Tool timed out" after N seconds
**Severity**: MEDIUM
**When It Occurs**: Long-running tool or unresponsive server

**Root Causes**:
- Slow file system
- Large document processing
- Infinite loop in tool
- Deadlock
- Resource exhaustion

**Diagnosis**:
```bash
# Measure tool execution time
time document-mcp stdio < /dev/null

# Test with large document
dd if=/dev/zero bs=1M count=100 | tr '\0' 'x' > /tmp/large_doc.txt
wc -c /tmp/large_doc.txt

# Monitor resource usage
top -c -p $(pgrep -f document-mcp)
```

**Solutions**:

1. **Increase MCP timeout**:
   ```json
   {
     "mcpServers": {
       "document-mcp": {
         "command": "document-mcp",
         "args": ["stdio"],
         "env": {
           "MCP_TIMEOUT": "60"
         }
       }
     }
   }
   ```

2. **Use pagination for large documents**:
   ```python
   # Instead of read_content(full=true)
   # Use read_content(page=1, page_size=50000)
   ```

3. **Optimize slow file system**:
   - Move documents to faster storage
   - Check disk I/O: `iostat -x 1`
   - Use SSD if available

4. **Monitor resource usage**:
   ```bash
   # Check memory
   ps aux | grep document-mcp

   # Check disk space
   df -h .documents_storage/
   ```

**Prevention**:
- Test with realistic document sizes
- Use pagination for large content
- Monitor performance in production

---

## File System Issues

### Issue: "Document not found" or "Permission denied"
**Severity**: MEDIUM
**When It Occurs**: Tool tries to access file

**Root Causes**:
- Document deleted externally
- Wrong storage path
- Permission issue
- File system corruption
- Concurrent access

**Diagnosis**:
```bash
# Check storage path
echo $DOCUMENT_STORAGE_PATH  # or default .documents_storage

# List documents
ls -la .documents_storage/ 2>/dev/null || ls -la $(python3 -c "from document_mcp.utils.file_operations import DOCS_ROOT_PATH; print(DOCS_ROOT_PATH)")

# Check permissions
ls -la .documents_storage/DocumentName/

# Verify file exists
find .documents_storage -name "*.md" -type f
```

**Solutions**:

1. **Verify storage path**:
   ```bash
   # Check what path is configured
   python3 -c "
from document_mcp.config import get_settings
settings = get_settings()
print(f'Storage path: {settings.document_storage_path}')
"
   ```

2. **Set explicit storage path**:
   ```bash
   export DOCUMENT_STORAGE_PATH=/path/to/documents
   document-mcp stdio
   ```

3. **Fix permissions**:
   ```bash
   # Make storage directory readable/writable
   chmod -R u+rwX .documents_storage/

   # Fix owner if needed
   chown -R $(whoami) .documents_storage/
   ```

4. **Recover from concurrent access**:
   ```bash
   # Stop server
   killall document-mcp

   # Verify no locks
   find .documents_storage -name "*.lock" -type f -delete

   # Restart
   document-mcp stdio
   ```

**Prevention**:
- Don't edit documents externally while server running
- Use exclusive lock for concurrent access control
- Regular backups of documents

---

### Issue: "Snapshot not found" or version control issues
**Severity**: MEDIUM
**When It Occurs**: Trying to restore or list snapshots

**Root Causes**:
- Snapshot corrupted
- Automatic snapshots disabled
- Storage full
- Permission issue
- Snapshot deleted

**Diagnosis**:
```bash
# List all snapshots
ls -la .documents_storage/DocumentName/.snapshots/ 2>/dev/null

# Check snapshot format
ls -la .documents_storage/DocumentName/.snapshots/snapshot_*/ | head -20

# Verify snapshot content
cat .documents_storage/DocumentName/.snapshots/*/content.md | head -20
```

**Solutions**:

1. **Verify automatic snapshots enabled**:
   ```python
   # Check settings
   python3 -c "
from document_mcp.config import get_settings
s = get_settings()
print(f'Snapshots enabled: {s.enable_automatic_snapshots}')
"
   ```

2. **Manually trigger snapshot**:
   ```python
   # Use manage_snapshots tool with action="create"
   ```

3. **Recover from missing snapshots**:
   ```bash
   # Files are in .snapshots/ - can manually copy back
   ls .documents_storage/DocumentName/.snapshots/
   ```

4. **Clean up old snapshots**:
   ```bash
   # Keep only recent snapshots
   ls -t .documents_storage/DocumentName/.snapshots/ | tail -n +11 | xargs rm -rf
   ```

**Prevention**:
- Verify automatic snapshots enabled
- Monitor disk space
- Regular backup of .snapshots/ directory

---

## Configuration Issues

### Issue: "Invalid configuration" in Claude Desktop
**Severity**: HIGH
**When It Occurs**: Claude Desktop reads config file

**Root Causes**:
- JSON syntax error
- Missing required fields
- Invalid argument format
- Encoding issue

**Diagnosis**:
```bash
# Validate JSON syntax
python3 -m json.tool < ~/Library/Application\ Support/Claude/claude_desktop_config.json

# Check for common errors
cat ~/Library/Application\ Support/Claude/claude_desktop_config.json | grep -E "mcp|document"

# Test configuration
python3 << 'EOF'
import json

config_path = "~/Library/Application Support/Claude/claude_desktop_config.json"
try:
    with open(config_path, 'r') as f:
        config = json.load(f)
    print("✓ JSON valid")

    if 'mcpServers' in config:
        print(f"✓ Found {len(config['mcpServers'])} MCP servers")
        for name, settings in config['mcpServers'].items():
            print(f"  - {name}: {settings.get('command', 'NO COMMAND')}")
    else:
        print("✗ No mcpServers section")
except json.JSONDecodeError as e:
    print(f"✗ JSON error: {e}")
EOF
```

**Solutions**:

1. **Validate JSON**:
   ```bash
   # Use online validator or:
   python3 -m json.tool < ~/Library/Application\ Support/Claude/claude_desktop_config.json > /dev/null
   ```

2. **Check syntax carefully**:
   ```json
   {
     "mcpServers": {
       "document-mcp": {
         "command": "document-mcp",    // ← comma required
         "args": ["stdio"],              // ← comma required
         "env": {}                       // ← no comma on last item
       }
     }
   }
   ```

3. **Common fixes**:
   - Add missing commas between objects
   - Remove trailing commas from last items
   - Quote all strings
   - Use `"args": ["stdio"]` not `"args": "stdio"`

**Prevention**:
- Use JSON validator before saving
- Copy working examples
- Use text editor with JSON support

---

## Performance Issues

### Issue: Tool calls are slow (>5-10 seconds)
**Severity**: LOW-MEDIUM
**When It Occurs**: During normal operation

**Root Causes**:
- Slow file system
- Large document processing
- Network latency (if applicable)
- Resource contention
- Inefficient algorithm

**Diagnosis**:
```bash
# Measure file system performance
time ls -la .documents_storage/

# Measure MCP overhead
time echo '{}' | document-mcp stdio > /dev/null

# Check system resources
top
free -h
df -h

# Monitor during tool call
watch -n 0.1 'top -b -n 1 | grep document-mcp'
```

**Solutions**:

1. **Use pagination for large documents**:
   ```python
   # Instead of loading entire document
   # Use page=1, page_size=50000
   ```

2. **Optimize file system**:
   - Move to SSD if on HDD
   - Check for disk fragmentation
   - Use local storage not network

3. **Reduce snapshots**:
   ```bash
   # Prune old snapshots
   find .documents_storage -name ".snapshots" -type d | xargs -I {} \
     sh -c 'ls -t {} | tail -n +11 | xargs -I {} rm -rf {}/'
   ```

4. **Monitor resource usage**:
   ```bash
   # Watch for memory leaks
   watch -n 1 'ps aux | grep document-mcp'

   # Check disk I/O
   iostat -x 1
   ```

**Prevention**:
- Design documents with pagination in mind
- Test with realistic document sizes
- Monitor performance metrics

---

## Crash Issues

### Issue: Server crashes or exits unexpectedly
**Severity**: CRITICAL
**When It Occurs**: During operation

**Root Causes**:
- Unhandled exception
- Out of memory
- Segmentation fault
- Invalid state
- External signal

**Diagnosis**:
```bash
# Capture crash output
document-mcp stdio 2>&1 | tee /tmp/mcp_crash.log

# Check system logs
journalctl -xe | grep document-mcp

# Run with debug
DEBUG=1 python3 -m document_mcp.doc_tool_server stdio 2>&1

# Check resource limits
ulimit -a
```

**Solutions**:

1. **Capture full error**:
   ```bash
   # Run with verbose output
   python3 -m document_mcp.doc_tool_server stdio 2>&1 | head -100

   # Save to file for analysis
   python3 -m document_mcp.doc_tool_server stdio &> /tmp/mcp.log &
   # Trigger crash
   kill %1
   cat /tmp/mcp.log
   ```

2. **Check dependencies**:
   ```bash
   pip check
   ```

3. **Increase resource limits**:
   ```bash
   # Increase max open files
   ulimit -n 4096
   ```

4. **Restart with fresh state**:
   ```bash
   # Clean temp files
   rm -rf /tmp/mcp_*

   # Restart server
   document-mcp stdio
   ```

5. **Report crash**:
   - Capture full output
   - Check document size
   - Note reproduction steps
   - Submit GitHub issue

**Prevention**:
- Test with realistic scenarios
- Monitor resource usage
- Use process supervisor (systemd, supervisord)

---

## Data Integrity Issues

### Issue: Documents corrupted or content missing
**Severity**: CRITICAL
**When It Occurs**: After operations or crashes

**Root Causes**:
- Concurrent write
- Crash during write
- File system error
- Snapshot system failure
- External modification

**Diagnosis**:
```bash
# List document structure
find .documents_storage/DocumentName -type f

# Check file integrity
file .documents_storage/DocumentName/*.md

# Verify snapshots
ls -la .documents_storage/DocumentName/.snapshots/

# Check git status if versioned
cd .documents_storage/DocumentName
git status
```

**Solutions**:

1. **Restore from snapshot**:
   ```python
   # Use manage_snapshots with action="restore" and snapshot_id
   ```

2. **Recover from file system**:
   ```bash
   # Check if file still exists in .snapshots/
   ls .documents_storage/DocumentName/.snapshots/*/content.md

   # Restore older version
   cp .documents_storage/DocumentName/.snapshots/*/content.md \
      .documents_storage/DocumentName/01-chapter.md
   ```

3. **Manual recovery**:
   ```bash
   # If using git backend
   cd .documents_storage/DocumentName
   git log --oneline
   git checkout COMMIT_SHA -- 01-chapter.md
   ```

4. **Verify data integrity**:
   ```bash
   # Check all files readable
   find .documents_storage -type f -exec file {} \;
   ```

**Prevention**:
- Use automatic snapshots (enabled by default)
- Regular backups to external storage
- Don't modify files externally
- Use exclusive access control

---

## Testing Your Troubleshooting

### Quick Verification Checklist
- [ ] `document-mcp --version` works
- [ ] `document-mcp stdio` starts without hanging
- [ ] Can list 28 tools
- [ ] Can create document
- [ ] Can persist changes
- [ ] Can recover from snapshots
- [ ] No crash during normal use

### Where to Get Help
1. Run `scripts/verify_mcp.sh` and check output
2. Check this troubleshooting guide
3. Review logs in `document_mcp/` directory
4. Test manually with Python
5. Open GitHub issue with full diagnostics

### Data to Include in Bug Report
```bash
# Collect all diagnostics
bash scripts/verify_mcp.sh > diagnostics.txt 2>&1
python3 -c "import document_mcp; print(document_mcp.__version__)" >> diagnostics.txt
pip show document-mcp mcp pydantic-ai >> diagnostics.txt
python3 --version >> diagnostics.txt
# Attach diagnostics.txt to GitHub issue
```

---

## References
- [CLAUDE_INTEGRATION_TESTING.md](./CLAUDE_INTEGRATION_TESTING.md)
- [docs/MCP_DESIGN_PATTERNS.md](./MCP_DESIGN_PATTERNS.md)
- [GitHub Issues](https://github.com/your-org/document-mcp/issues)
