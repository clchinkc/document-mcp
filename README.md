[![codecov](https://codecov.io/gh/clchinkc/document-mcp/graph/badge.svg?token=TEGUTD2DIF)](https://codecov.io/gh/clchinkc/document-mcp)
[![Python Tests with Coverage](https://github.com/clchinkc/document-mcp/actions/workflows/python-test.yml/badge.svg)](https://github.com/clchinkc/document-mcp/actions/workflows/python-test.yml)
# Document MCP

[![PyPI version](https://badge.fury.io/py/document-mcp.svg)](https://badge.fury.io/py/document-mcp)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Document MCP gives writers, researchers, and knowledge-managers **first-class control over large-scale Markdown documents** with **built-in safety features** that prevent content loss. Manage books, research papers, and documentation with 26 AI-powered tools.

## 🚀 Quick Start

### Option 1: Hosted Service (Recommended)

**For Claude Desktop users** - No installation required. Just add to your Claude Desktop config:

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "document-mcp": {
      "url": "https://document-mcp-451560119112.asia-east1.run.app"
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
- 26 MCP tools for document management
- Your own isolated document storage
- Automatic snapshots and version control
- No setup, no API keys, no maintenance

---

### Option 2: Local Installation (For Claude Code / Developers)

**For Claude Code users** or those who want local document storage:

```bash
pip install document-mcp
```

Add to your Claude Code MCP settings:

```json
{
  "mcpServers": {
    "document-mcp": {
      "command": "python",
      "args": ["-m", "document_mcp.doc_tool_server", "stdio"]
    }
  }
}
```

See the **[Package Installation Guide](document_mcp/README.md)** for detailed setup with universal path finding.

---

## 📖 What is Document MCP?

Document MCP provides a structured way to manage large documents composed of multiple chapters. Think of it as a file system specifically designed for books, research papers, documentation, or any content that benefits from being split into manageable sections.

### Key Features

- **26 MCP Tools**: Document management, chapter operations, paragraph editing, semantic search, and version control
- **Built-in Safety**: Automatic snapshots before destructive operations, version history, and conflict detection
- **Pagination System**: Industry-standard pagination for large documents (50K chars per page)
- **User Isolation**: Each authenticated user gets their own isolated storage (hosted version)
- **Local-First Option**: Keep your documents on your own machine (PyPI version)

### Document Organization

```
.documents_storage/
├── my_novel/                    # A document
│   ├── 01-prologue.md          # Chapters ordered by filename
│   ├── 02-chapter-one.md
│   └── 03-chapter-two.md
└── research_paper/             # Another document
    ├── 00-abstract.md
    ├── 01-introduction.md
    └── 02-methodology.md
```

## 🛡️ Safety Features

Document MCP includes comprehensive safety features designed to prevent content loss:

- **Automatic Snapshots**: Created before every destructive operation
- **Named Checkpoints**: Create restore points with `snapshot_document`
- **Version Restoration**: Roll back to any previous version with `restore_snapshot`
- **Conflict Detection**: Warns about potential overwrites from external modifications
- **Audit Trail**: Complete modification history with timestamps

## 🌐 Hosted Deployment Details

The hosted version runs on Google Cloud Run:

| Feature | Details |
|---------|---------|
| **Authentication** | OAuth 2.1 with PKCE via Google |
| **Region** | asia-east1 (Taiwan) |
| **Scaling** | Auto-scales 0-10 instances based on load |
| **Cost** | Free for users (scales to zero when idle) |

To deploy your own instance, see the [Self-Hosting Guide](#self-hosting-guide) below.

## 🔧 Tool Categories

Document MCP provides 26 tools organized into 6 categories:

| Category | Tools | Description |
|----------|-------|-------------|
| **Document** | 6 | Create, delete, list documents; manage summaries |
| **Chapter** | 5 | Add, edit, rename, delete, reorder chapters |
| **Paragraph** | 7 | Atomic paragraph operations (insert, replace, delete, move) |
| **Content** | 5 | Read, search, replace, statistics, semantic search |
| **Safety** | 3 | Snapshots, restore, diff comparison |

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
git clone https://github.com/clchinkc/document-mcp.git
cd document-mcp

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e ".[dev]"
```

### Running Tests

```bash
# All tests (352 tests, 100% pass rate)
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
document-mcp stdio
```

## 📋 Self-Hosting Guide

Deploy your own Document MCP instance on Google Cloud:

### Prerequisites
- Google Cloud project with billing enabled
- `gcloud` CLI installed and configured

### 1. Create Google OAuth Credentials

1. Go to [Google Cloud Console → APIs & Services → Credentials](https://console.cloud.google.com/apis/credentials)
2. Create OAuth 2.0 Client ID (Web application)
3. Add authorized redirect URI: `https://YOUR-SERVICE-URL/oauth/callback`
4. Note the Client ID and Client Secret

### 2. Store Secrets

```bash
echo -n "YOUR_CLIENT_ID" | gcloud secrets create GOOGLE_OAUTH_CLIENT_ID --data-file=-
echo -n "YOUR_CLIENT_SECRET" | gcloud secrets create GOOGLE_OAUTH_CLIENT_SECRET --data-file=-
```

### 3. Deploy to Cloud Run

```bash
gcloud run deploy document-mcp \
  --source . \
  --region asia-east1 \
  --allow-unauthenticated \
  --set-secrets "GOOGLE_OAUTH_CLIENT_ID=GOOGLE_OAUTH_CLIENT_ID:latest,GOOGLE_OAUTH_CLIENT_SECRET=GOOGLE_OAUTH_CLIENT_SECRET:latest" \
  --set-env-vars "SERVER_URL=https://YOUR-SERVICE-URL"
```

The server uses **Firestore** by default for OAuth state storage (free tier, no setup required).

### 4. Update OAuth Redirect URI

After deployment, update your OAuth credentials with the actual Cloud Run URL:
`https://document-mcp-XXXXXX.asia-east1.run.app/oauth/callback`

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
