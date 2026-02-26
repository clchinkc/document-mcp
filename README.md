[![codecov](https://codecov.io/gh/clchinkc/story-mcp/graph/badge.svg?token=TEGUTD2DIF)](https://codecov.io/gh/clchinkc/story-mcp)
[![Python Tests with Coverage](https://github.com/clchinkc/story-mcp/actions/workflows/python-test.yml/badge.svg)](https://github.com/clchinkc/story-mcp/actions/workflows/python-test.yml)
# Story MCP

[![PyPI version](https://badge.fury.io/py/story-mcp.svg)](https://badge.fury.io/py/story-mcp)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Story MCP gives writers, researchers, and knowledge-managers **first-class control over large-scale Markdown documents** with **built-in safety features** that prevent content loss. Manage books, research papers, and documentation with **37 AI-powered tools** including context management, git-backed version history, and semantic search.

> **Phase 4 Complete** ✅ - v0.0.5 Production Ready (February 26, 2026)

## 🚀 Quick Start

### Option 1: Hosted Service (Recommended)

**For Claude Desktop users** - No installation required. Just add to your Claude Desktop config:

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "story-mcp": {
      "url": "https://story-mcp-451560119112.asia-east1.run.app"
    }
  }
}
```

Restart Claude Desktop. When you first connect:
1. Your browser opens for Google OAuth authentication
2. Sign in with your Google account
3. Claude Desktop securely stores your access token
4. Start managing documents immediately!

**What you get:**
- 37 MCP tools for document management
- Your own isolated document storage
- Automatic snapshots and git-backed version history
- Cross-session context management
- Semantic search with embeddings
- No setup, no API keys, no maintenance

---

### Option 2: Local Installation (For Claude Code / Developers)

**For Claude Code users** or those who want local document storage:

```bash
pip install story-mcp
```

Add to your Claude Code MCP settings:

```json
{
  "mcpServers": {
    "story-mcp": {
      "command": "python",
      "args": ["-m", "story_mcp.doc_tool_server", "stdio"]
    }
  }
}
```

See the **[Package Installation Guide](story_mcp/README.md)** for detailed setup with universal path finding.

---

## 📖 What is Story MCP?

Story MCP provides a structured way to manage large stories and documents composed of multiple chapters. Think of it as a file system specifically designed for novels, screenplays, research papers, documentation, or any content that benefits from being split into manageable sections.

### Key Features

- **37 MCP Tools** (Phase 4 Complete ✅):
  - Story management, chapter operations, paragraph editing
  - Semantic search with embeddings
  - Git-backed version history
  - Cross-session context management (OneContext-inspired)
  - Entity tracking, metadata, and safety features
- **Built-in Safety**: Git-backed version control, automatic commits, snapshots, and conflict detection
- **Pagination System**: Page-based content access for large documents (50K chars per page)
- **User Isolation**: Each authenticated user gets their own isolated storage (hosted version)
- **Local-First Option**: Keep your stories on your own machine (PyPI version)

### Document Organization

```
.documents_storage/
├── my_novel/                    # A story/document
│   ├── 01-prologue.md          # Chapters ordered by filename
│   ├── 02-chapter-one.md
│   └── 03-chapter-two.md
└── research_paper/             # Another document
    ├── 00-abstract.md
    ├── 01-introduction.md
    └── 02-methodology.md
```

## 🛡️ Safety Features

Story MCP includes safety features designed to prevent content loss:

- **Automatic Snapshots**: Created before every destructive operation
- **Named Checkpoints**: Create restore points with `snapshot_document`
- **Version Restoration**: Roll back to any previous version with `restore_snapshot`
- **Conflict Detection**: Warns about potential overwrites from external modifications
- **Audit Trail**: Complete modification history with timestamps

## 🌐 Hosted Service Details

The hosted version runs on Google Cloud Run:

| Feature | Details |
|---------|---------|
| **Authentication** | OAuth 2.1 with PKCE via Google |
| **Region** | asia-east1 (Taiwan) |
| **Scaling** | Auto-scales 0-10 instances based on load |
| **Cost** | Free for users (scales to zero when idle) |

## 🔧 Tool Categories

Story MCP provides **37 tools** organized into **10 categories**:

| Category | Tools | Description |
|----------|-------|-------------|
| **Document** | 6 | Create, delete, list documents; manage summaries |
| **Chapter** | 4 | Add, edit, delete, list chapters with frontmatter |
| **Paragraph** | 4 | Atomic paragraph operations (insert, replace, delete, move) |
| **Content** | 6 | Read, search, replace, statistics, semantic search, entity tracking |
| **Metadata** | 3 | Chapter frontmatter, entities, timeline management |
| **Safety** | 3 | Git history, restore, diff comparison |
| **Overview** | 1 | Document outline with metadata |
| **Discovery** | 1 | Tool search and discovery |
| **Context** | 6 | Store/recall memories, export/import, list memories |
| **Version** | 3 | Get history, checkout version, compare versions |
| **Discovery** | 1 | Tool search and discovery |

## 🤖 Example Workflows

### Basic Document Management
```
👤 User: Create a new document called 'My Novel'
🤖 Claude: ✅ Created document 'My Novel'

👤 User: Add a chapter called '01-introduction.md' with content '# Chapter 1\n\nIt was a dark and stormy night...'
🤖 Claude: ✅ Created chapter '01-introduction.md' in 'My Novel'

👤 User: List all my documents
🤖 Claude: ✅ Found 1 document: 'My Novel' with 1 chapter
```

### Safety Features in Action
```
👤 User: Delete paragraph 3 from chapter '02-climax.md' in 'My Novel'
🤖 Claude: ✅ Deleted paragraph 3. Automatic snapshot created for recovery.

👤 User: Actually, restore the last snapshot
🤖 Claude: ✅ Restored from snapshot. Paragraph 3 is back.
```

### Semantic Search
```
👤 User: Find content similar to "the hero's journey" in my novel
🤖 Claude: ✅ Found 3 paragraphs with similar themes:
   - Chapter 2, paragraph 5 (similarity: 0.89)
   - Chapter 4, paragraph 12 (similarity: 0.82)
   - Chapter 1, paragraph 3 (similarity: 0.78)
```

## 🛠️ Development

### Prerequisites
- Python 3.10+
- Git

### Local Development Setup

```bash
# Clone the repository
git clone https://github.com/clchinkc/story-mcp.git
cd document-mcp

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e ".[dev]"
```

### Running Tests

```bash
# All tests (528 tests)
uv run pytest

# By tier
uv run pytest tests/unit/           # Fast, isolated tests
uv run pytest tests/integration/    # Real MCP, mocked LLM
uv run pytest tests/e2e/            # Full system (requires API keys)

# Code quality
uv run ruff check --fix && uv run ruff format
uv run mypy document_mcp/
```

### Running the MCP Server Locally

```bash
# Start MCP server
uv run python -m document_mcp.doc_tool_server stdio

# Or with PyPI installation
story-mcp stdio
```

## 📚 Documentation

| Guide | Description |
|-------|-------------|
| **[Package Installation](document_mcp/README.md)** | PyPI setup for Claude Code |
| **[Manual Testing](docs/manual_testing.md)** | Creative writing workflows |
| **[MCP Design Patterns](docs/MCP_DESIGN_PATTERNS.md)** | Production patterns and best practices |
| **[Testing Strategy](tests/README.md)** | 4-tier testing architecture |

## 🤝 Contributing

Contributions welcome! Please run the test suite before submitting PRs:

```bash
uv run pytest && uv run ruff check && uv run mypy document_mcp/
```

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- Built with [Model Context Protocol (MCP)](https://github.com/modelcontextprotocol)
- Powered by [Pydantic AI](https://github.com/pydantic/pydantic-ai)
- Hosted on [Google Cloud Run](https://cloud.google.com/run)

---

⭐ **Star this repo** if you find it useful!
