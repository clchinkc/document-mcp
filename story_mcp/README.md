# Story-MCP: The Developer's Story Engine

[![PyPI version](https://badge.fury.io/py/story-mcp.svg)](https://badge.fury.io/py/story-mcp)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**Story-MCP is a powerful, local-first MCP server for building, managing, and automating complex, structured stories with confidence. It's designed for developers, writers, and storytellers who need robust, safe, and programmatic control over their narratives.**

While other tools focus on simple note-taking, Story-MCP is built for creating and managing multi-chapter stories like novels, screenplays, and interactive narratives. It provides a suite of developer-focused features, including write-safety, automatic snapshots, version control, pagination-based content access, and batch processing.

### Recent Updates (v0.0.5)
- ✅ **Pagination System**: Industry-standard pagination for complete large document access
- ✅ **Enhanced Testing**: 477 comprehensive tests with 100% pass rate (61% coverage)
- ✅ **GCP Observability**: Native Cloud Logging, Tracing, and Monitoring integration (works locally too)
- ✅ **Improved Documentation**: Updated guides and API references
- ✅ **Performance Optimization**: Better timeout handling and reliability

## Why Story-MCP?

| Feature | Story-MCP | Basic Memory | MCP-Obsidian |
| --- | --- | --- | --- |
| **Use Case** | Complex, structured documents (books, reports) | Simple notes and knowledge snippets | Notes within an Obsidian vault |
| **Safety** | Snapshots, versioning, transactions | Basic file operations | Basic file operations |
| **Developer Focus** | Metrics, pagination, batch processing, granular control | Simple API for conversational knowledge | Obsidian-centric API |
| **Structure** | Multi-chapter documents | Single notes with semantic tags | Standard Markdown notes |

**Choose Story-MCP if you need to:**

*   **Build complex documents:** Create and manage documents with multiple chapters, and have granular control over paragraphs and content with pagination support.
*   **Ensure data safety:** Protect your work with automatic snapshots and versioning, and perform complex operations with atomic transactions.
*   **Automate your workflows:** Use batch processing to perform multiple operations at once, and monitor everything with detailed performance metrics.
*   **Work with code:** Story-MCP is designed to be used programmatically, making it ideal for developers and technical users.
*   **Handle large documents:** Industry-standard pagination ensures complete access to documents of any size.

## Installation

```bash
# Install with pip (traditional)
pip install story-mcp

# Install with uv (recommended - 10-100x faster)
uv add story-mcp

# For development with modern toolchain
uv sync
```

## Quick Start

### Document Structure

This MCP server treats "documents" as directories containing multiple "chapter" files (Markdown .md files). Chapters are ordered alphanumerically by their filenames (e.g., `01-introduction.md`, `02-main_content.md`).

```
.documents_storage/           # Root directory for all documents
├── my_book/                     # A document (directory)
│   ├── 01-introduction.md       # Chapter 1 (alphanumeric ordering)
│   ├── 02-main_content.md       # Chapter 2
│   ├── 03-conclusion.md         # Chapter 3
│   └── _manifest.json           # Optional: For future explicit chapter ordering
└── research_paper/              # Another document
    ├── 00-abstract.md
    ├── 01-methodology.md
    └── 02-results.md
```

### Setup for Claude Code

**Option 1: Automatic Setup (Recommended)**

1.  **Install the package:**
    ```bash
    pip install story-mcp
    ```

2.  **Add to Claude Code automatically (recommended: local scope):**
    ```bash
    claude mcp add story-mcp -s local -- story-mcp stdio
    ```

3.  **Verify installation:**
    ```bash
    claude mcp list
    ```
    You should see `story-mcp: ... - ✓ Connected`

**Option 2: Manual Configuration**

If the automatic method doesn't work, use manual configuration:

1.  **Install and find the binary path:**
    ```bash
    # Install the package
    pip install story-mcp
    
    # Find where it's installed (try these commands in order):
    which story-mcp
    # If not found, try:
    python3 -c "import subprocess; print(subprocess.check_output(['which', 'story-mcp']).decode().strip())"
    # If still not found, try:
    python3 -c "import story_mcp, os; print(os.path.join(os.path.dirname(story_mcp.__file__), '..', '..', '..', 'bin', 'story-mcp'))"
    ```

2.  **Add MCP server configuration:**

    Add this configuration to your MCP settings file:
    - **Claude Code/Cursor:** `~/.cursor/mcp.json`
    - **Claude Desktop:** `~/Library/Application Support/Claude/claude_desktop_config.json` (on macOS)

    ```json
    {
      "mcpServers": {
        "story-mcp": {
          "command": "story-mcp",
          "args": ["stdio"],
          "env": {}
        }
      }
    }
    ```

    **If `story-mcp` is not in your PATH**, use the full path you found above:
    ```json
    {
      "mcpServers": {
        "story-mcp": {
          "command": "/full/path/to/story-mcp",
          "args": ["stdio"],
          "env": {}
        }
      }
    }
    ```

**Option 3: Development Installation**

For development or if you're working from source:

1.  **Clone and install:**
    ```bash
    git clone https://github.com/clchinkc/story-mcp.git
    cd story-mcp
    pip install -e .
    ```

2.  **Find the development binary:**
    ```bash
    python3 -c "import sys, os; print(os.path.join(sys.prefix, 'bin', 'story-mcp'))"
    ```

3.  **Add using the full path (local scope):**
    ```bash
    claude mcp add story-mcp -s local -- /path/to/your/venv/bin/story-mcp stdio
    ```

### Troubleshooting Claude Code Integration

**Common Issues:**

- **"spawn story-mcp ENOENT" error**: The binary is not in your PATH. Use Option 2 with the full path.
- **Connection timeout**: Verify the server starts correctly: `story-mcp --help`
- **Permission errors**: Ensure the binary has execute permissions: `chmod +x /path/to/story-mcp`
- **Module not found**: Reinstall the package: `pip uninstall story-mcp && pip install story-mcp`

**Quick Troubleshooting:**

| Problem | Solution |
|---------|----------|
| "Can't find story-mcp command" | Use full path: `/path/to/story-mcp` in config |
| "Connection timeout" | Restart Claude Code and check `claude mcp list` |
| "Document not found" | Check exact document name - names are case-sensitive |
| "Chapter already exists" | Use different chapter name or delete existing one first |

**Verification Commands:**
```bash
# Test the binary directly
story-mcp --help

# Test with python module  
python3 -m story_mcp.doc_tool_server --help

# Check Claude Code integration
claude mcp list

# Run full verification script
bash scripts/verify_mcp.sh
```

### Setup for Other MCP Clients

For other MCP-compatible clients, use the same configuration with your client's specific config format. The server supports both `stdio` and `sse` transports.

## Quick Start Guide

### Your First Document (3 Simple Steps)

**Step 1: Create a Document**
```
"Create a new document called 'My Project'"
```

**Step 2: Add Content**
```
"Add chapter '01-intro.md' to 'My Project' with content: 
# Introduction
This is my project guide."
```

**Step 3: View Results**
```
"Show me the full document 'My Project'"
```

### Essential Commands You'll Use

**Document Management:**
- `"Create a new document called 'Project Name'"`
- `"List all my documents"`
- `"Delete document 'Old Project'"`

**Adding Content:**
- `"Add chapter '01-intro.md' to 'Project Name' with content: [your text]"`
- `"Append paragraph 'New content here' to chapter '01-intro.md' in 'Project Name'"`

**Viewing Content:**
- `"Show me document 'Project Name'"`
- `"Read chapter '01-intro.md' from 'Project Name'"`
- `"Get statistics for 'Project Name'"`

**Safety & Versions:**
- `"Create snapshot of 'Project Name' with message 'Before big changes'"`
- `"Show snapshots for 'Project Name'"`

### Setup Tips for Success

**1. Document Names**
- Use descriptive names: `"Marketing Plan 2025"`
- Avoid special characters: Use spaces or hyphens
- Keep under 50 characters

**2. Chapter Names**  
- Use numbered prefixes: `01-intro.md`, `02-overview.md`
- Always end with `.md`
- Be descriptive: `01-getting-started.md` not `01-part1.md`

**3. Content Organization**
- Start with `# Title` for chapter headers
- Use `## Section` and `### Subsection` for structure
- Keep paragraphs focused on one topic

**4. Safety First**
- Create snapshots before major changes
- Use descriptive snapshot messages: `"Draft complete"` not `"backup"`

### Using Claude Code in the Obsidian Terminal

You can use Story-MCP with the Obsidian Terminal extension to automate your documentation workflows.

1.  **Install the Obsidian Terminal extension.**
2.  **Install `story-mcp`** in the terminal's environment.
3.  **Start the Story-MCP server** in the terminal:
    ```bash
    story-mcp sse --host localhost --port 3001
    ```
4.  **Use Claude Code** to interact with the server using `curl` or a Python script.

**Example using `curl`:**

```bash
# Create a new document
curl -X POST http://localhost:3001/create_document -d '{"document_name": "my-obsidian-doc"}'

# Add a new chapter
curl -X POST http://localhost:3001/create_chapter -d '{"document_name": "my-obsidian-doc", "chapter_name": "01-daily-notes", "initial_content": "# Daily Notes"}'
```

## Use Cases

*   **Automated Technical Documentation:** Use an AI assistant to generate and maintain technical documentation for your codebase.
*   **AI-Powered Research:** Build and organize a research paper with an AI assistant, with each section as a separate chapter.
*   **Creative Writing:** Write a multi-chapter novel or screenplay with an AI writing partner, using snapshots to save your progress.

## Anonymous Usage Analytics

Story-MCP includes **automatic, anonymous usage analytics** that help improve the system for all users. This follows industry-standard practices used by npm, VS Code extensions, and Docker.

### What's Collected (Anonymous Only)
- **Tool usage patterns**: Which MCP tools are used most frequently
- **Performance metrics**: Real-world execution times across different environments  
- **Error rates**: Common failure patterns to prioritize fixes
- **System information**: Service version, environment type

### What's NOT Collected
- ❌ **No document content** or file names
- ❌ **No personal information** or user identification
- ❌ **No file paths** or directory structures
- ❌ **No reversible tracking** or individual user data

### Privacy Controls
- **Automatic collection**: Enabled by default for community benefit
- **Easy opt-out**: Set `MCP_METRICS_ENABLED=false` to disable completely
- **Transparent**: All collection documented and open-source
- **Anonymous**: No way to identify individual users

This data helps prioritize development efforts on the tools and features that matter most to real users.

## Advanced Usage & Development

This package documentation focuses on installing and using the Story-MCP server. For comprehensive information about the full project, including:

- **🤖 AI Agents**: Example agents (Simple, ReAct) with step-by-step workflows
- **🛠️ Development Setup**: Complete development environment and testing
- **📊 Performance Optimization**: Automated prompt optimization and benchmarking
- **🔧 Advanced Configuration**: Environment variables, logging, metrics
- **📖 User Workflows**: Creative writing, research, and documentation workflows
- **🏗️ MCP Design Patterns**: Production-ready patterns for context management and partial hydration

Please see the **[Main Project Documentation](../README.md)** and **[MCP Design Patterns Guide](../docs/MCP_DESIGN_PATTERNS.md)**.

## Technical Details

For more detailed information on the available tools, data models, and advanced features like monitoring and logging, please see the [Technical Reference](TECHNICAL_REFERENCE.md).

## Contributing

Contributions are welcome! Please see the [Contributing Guide](../README.md#-contributing) and the [Main Project Documentation](../README.md) for development setup instructions.

## License

MIT License
