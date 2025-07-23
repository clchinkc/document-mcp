# Document-MCP: The Developer's Document Store

[![PyPI version](https://badge.fury.io/py/document-mcp.svg)](https://badge.fury.io/py/document-mcp)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**Document-MCP is a powerful, local-first MCP server for building, managing, and automating complex, structured documents with confidence. It's designed for developers, technical writers, and researchers who need robust, safe, and programmatic control over their content.**

While other tools focus on simple note-taking, Document-MCP is built for creating and managing multi-chapter documents like books, reports, and technical documentation. It provides a suite of developer-focused features, including write-safety, automatic snapshots, version control, and batch processing.

## Why Document-MCP?

| Feature | Document-MCP | Basic Memory | MCP-Obsidian |
| --- | --- | --- | --- |
| **Use Case** | Complex, structured documents (books, reports) | Simple notes and knowledge snippets | Notes within an Obsidian vault |
| **Safety** | Snapshots, versioning, transactions | Basic file operations | Basic file operations |
| **Developer Focus** | Metrics, batch processing, granular control | Simple API for conversational knowledge | Obsidian-centric API |
| **Structure** | Multi-chapter documents | Single notes with semantic tags | Standard Markdown notes |

**Choose Document-MCP if you need to:**

*   **Build complex documents:** Create and manage documents with multiple chapters, and have granular control over paragraphs and content.
*   **Ensure data safety:** Protect your work with automatic snapshots and versioning, and perform complex operations with atomic transactions.
*   **Automate your workflows:** Use batch processing to perform multiple operations at once, and monitor everything with detailed performance metrics.
*   **Work with code:** Document-MCP is designed to be used programmatically, making it ideal for developers and technical users.

## Installation

```bash
# Install with pip (traditional)
pip install document-mcp

# Install with uv (recommended - 10-100x faster)
uv add document-mcp

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

### Setup for Cursor and Claude Desktop

1.  **Install the package:**
    ```bash
    pip install document-mcp
    ```

2.  **Add MCP server configuration:**

    Add this configuration to your IDE's MCP settings file:

    *   **Cursor:** `~/.cursor/mcp.json`
    *   **Claude Desktop:** `~/Library/Application Support/Claude/claude_desktop_config.json` (on macOS)

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

3.  **Enable the MCP server:**
    *   Open Cursor or Claude Desktop.
    *   Enable the MCP server in the settings.
    *   The Document-MCP tools will now be available in your chat.

### Workflow Instructions for Cursor and Claude

With Document-MCP, you can use natural language to manage your documents. Here are some examples:

**Creating a new document:**

> "Create a new document called 'my-novel'."

**Adding chapters:**

> "Create a new chapter in 'my-novel' called '01-introduction' with the content '# My Novel\n\nThis is the introduction to my novel.'"

**Writing and editing:**

> "Append the paragraph 'This is a new paragraph.' to the chapter '01-introduction' in the document 'my-novel'."
>
> "Replace the first paragraph in the chapter '01-introduction' of the document 'my-novel' with 'This is the new first paragraph.'"

**Working with snapshots:**

> "Create a snapshot of the document 'my-novel' with the message 'First draft complete'."
>
> "List the snapshots for the document 'my-novel'."
>
> "Restore the snapshot with the ID '...' for the document 'my-novel'."

### Using Claude Code in the Obsidian Terminal

You can use Document-MCP with the Obsidian Terminal extension to automate your documentation workflows.

1.  **Install the Obsidian Terminal extension.**
2.  **Install `document-mcp`** in the terminal's environment.
3.  **Start the Document-MCP server** in the terminal:
    ```bash
    document-mcp sse --host localhost --port 3001
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

## Technical Details

For more detailed information on the available tools, data models, and advanced features like monitoring and logging, please see the [Technical Reference](TECHNICAL_REFERENCE.md).

## Contributing

Contributions are welcome! Please see the [Contributing Guide](CONTRIBUTING.md) for more information.

## License

MIT License
