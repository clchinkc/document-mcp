# Document MCP Server

[![PyPI version](https://badge.fury.io/py/document-mcp.svg)](https://badge.fury.io/py/document-mcp)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A Model Context Protocol (MCP) server for managing structured Markdown documents. This server provides tools to create, read, update, and analyze documents composed of multiple chapters.

## Installation

```bash
pip install document-mcp
```

## Quick Start

### Setup for MCP-Compatible IDEs

1. **Install the package:**
   ```bash
   pip install document-mcp
   ```

2. **Add MCP server configuration:**
   
   Add this configuration to your IDE's MCP settings file:
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

3. **IDE-specific configuration locations:**
   
   - **Cursor**: `~/.cursor/mcp.json`
   - **Windsurf**: Add to Windsurf MCP configuration file
   - **Claude Desktop**: Add to Claude Desktop MCP settings
   - **Other MCP clients**: Refer to your client's MCP configuration documentation

4. **Enable the MCP server:**
   - Open your IDE
   - Enable/turn on the MCP server (usually in settings or toolbar)
   - Wait for the connection indicator to turn green
   - The document-mcp tools are now available!

### Manual Server Mode

```bash
# Start the MCP server manually
python -m document_mcp.doc_tool_server sse --host localhost --port 3001
```

When running, the server exposes several key endpoints:
- **SSE**: `http://localhost:3001/sse`
- **Health**: `http://localhost:3001/health`
- **Metrics**: `http://localhost:3001/metrics`

## Overview

This MCP server treats "documents" as directories containing multiple "chapter" files (Markdown .md files). Chapters are ordered alphanumerically by their filenames (e.g., `01-introduction.md`, `02-main_content.md`).

### Document Structure

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

## Running the Server

The server supports both HTTP SSE and stdio transports. HTTP SSE is the default and recommended transport.

### HTTP SSE Transport (Recommended)

```bash
# Run with HTTP SSE transport (default)
python -m document_mcp.doc_tool_server sse --host localhost --port 3001

# Or specify arguments explicitly
python -m document_mcp.doc_tool_server sse --host 0.0.0.0 --port 8000
```

### Stdio Transport

```bash
# Run with stdio transport
python -m document_mcp.doc_tool_server stdio
```

## Monitoring and Metrics

The server exposes detailed performance and usage metrics via a Prometheus-compatible endpoint. This allows for real-time monitoring of tool calls, execution times, errors, and more.

### Viewing Metrics

- **Endpoint**: `http://localhost:3001/metrics` (when running with default SSE settings)
- **Installation**: Ensure `prometheus-client` is installed. It is included with the `[dev]` extras:
  ```bash
  pip install "document-mcp[dev]"
  # or manually
  pip install "prometheus-client>=0.17.0"
  ```
- **Usage**: Simply `curl` the endpoint or point your Prometheus scraper to it.

```bash
curl http://localhost:3001/metrics
```

### Key Metrics Collected

- `mcp_tool_calls_total`: A counter for how many times each tool is called, labeled by tool name and status (success/error).
- `mcp_tool_duration_seconds`: A histogram of tool execution times, allowing for calculation of averages and percentiles.
- `mcp_tool_errors_total`: A counter for errors, labeled by tool name and error type.
- `mcp_tool_argument_sizes_bytes`: A histogram of the size of arguments passed to tools.

### Configuration

Metrics are enabled by default. To disable them, set the following environment variable:

```bash
export MCP_METRICS_ENABLED=false
```

### Integration with Prometheus & Grafana

To scrape these metrics with Prometheus, add the following to your `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'document-mcp'
    static_configs:
      - targets: ['localhost:3001']
```

**Useful Grafana Queries:**
```promql
# Tool usage rate (per second, over 5m) by tool
sum(rate(mcp_tool_calls_total[5m])) by (tool_name)

# P95 response time over 5m by tool
histogram_quantile(0.95, sum(rate(mcp_tool_duration_seconds_bucket[5m])) by (le, tool_name))

# Error rate as a percentage of total calls over 5m
sum(rate(mcp_tool_errors_total[5m])) / sum(rate(mcp_tool_calls_total[5m]))
```

## MCP Tools Reference

The server exposes the following tools via the Model Context Protocol:

### Document Management

| Tool | Parameters | Description |
|------|------------|-------------|
| `list_documents` | - | Lists all available documents with metadata |
| `create_document` | `document_name: str` | Creates a new document directory |
| `delete_document` | `document_name: str` | Deletes a document and all its chapters |

### Chapter Management

| Tool | Parameters | Description |
|------|------------|-------------|
| `list_chapters` | `document_name: str` | Lists all chapters in a document, ordered by filename |
| `create_chapter` | `document_name: str`, `chapter_name: str`, `initial_content: str = ""` | Creates a new chapter file |
| `delete_chapter` | `document_name: str`, `chapter_name: str` | Deletes a chapter from a document |

### Content Operations

| Tool | Parameters | Description |
|------|------------|-------------|
| `read_chapter_content` | `document_name: str`, `chapter_name: str` | Reads the content and metadata of a specific chapter |
| `read_paragraph_content` | `document_name: str`, `chapter_name: str`, `paragraph_index_in_chapter: int` | Reads a specific paragraph from a chapter |
| `read_full_document` | `document_name: str` | Reads the entire document, concatenating all chapters |
| `write_chapter_content` | `document_name: str`, `chapter_name: str`, `new_content: str` | Overwrites the entire content of a chapter |
| `modify_paragraph_content` | `document_name: str`, `chapter_name: str`, `paragraph_index: int`, `new_paragraph_content: str`, `mode: str` | Modifies a paragraph (`replace`, `insert_before`, `insert_after`, `delete`) |
| `append_paragraph_to_chapter` | `document_name: str`, `chapter_name: str`, `paragraph_content: str` | Appends a new paragraph to the end of a chapter |

### Text Operations

| Tool | Parameters | Description |
|------|------------|-------------|
| `replace_text_in_chapter` | `document_name: str`, `chapter_name: str`, `text_to_find: str`, `replacement_text: str` | Replaces all occurrences of text in a specific chapter |
| `replace_text_in_document` | `document_name: str`, `text_to_find: str`, `replacement_text: str` | Replaces all occurrences of text throughout all chapters |

### Analysis Tools

| Tool | Parameters | Description |
|------|------------|-------------|
| `get_chapter_statistics` | `document_name: str`, `chapter_name: str` | Retrieves statistics (word count, paragraph count) for a chapter |
| `get_document_statistics` | `document_name: str` | Retrieves aggregated statistics for an entire document |

### Search Tools

| Tool | Parameters | Description |
|------|------------|-------------|
| `find_text_in_chapter` | `document_name: str`, `chapter_name: str`, `query: str`, `case_sensitive: bool = False` | Finds paragraphs containing the query string in a specific chapter |
| `find_text_in_document` | `document_name: str`, `query: str`, `case_sensitive: bool = False` | Finds paragraphs containing the query string across all chapters |

## Data Models

The server uses Pydantic models for structured data exchange:

- `DocumentInfo`: Metadata for a document
- `ChapterMetadata`: Metadata for a chapter
- `ChapterContent`: Full content and metadata of a chapter
- `ParagraphDetail`: Information about a specific paragraph
- `FullDocumentContent`: Complete content of a document
- `StatisticsReport`: Word and paragraph count statistics
- `OperationStatus`: Success/failure status for operations

## Requirements

- Python 3.8+
- fastapi
- uvicorn[standard]
- pydantic-ai
- mcp[cli]
- python-dotenv
- google-generativeai

## Testing

The MCP server uses a three-tier testing approach:

1. **Unit Tests**: Mock all dependencies for fast, reliable component testing
2. **Integration Tests**: Real MCP server with mocked AI for tool validation
3. **E2E Tests**: Real MCP server with real AI for complete system validation (runs in CI/CD)

Tests cover all MCP tools, error handling, boundary conditions, and multi-step workflows.

## Examples and Documentation

For comprehensive examples, tutorials, and usage guides, visit the [GitHub repository](https://github.com/document-mcp/document-mcp).

## License

MIT License

## Links

- **GitHub Repository**: [https://github.com/document-mcp/document-mcp](https://github.com/document-mcp)
- **Bug Reports**: [GitHub Issues](https://github.com/document-mcp/issues) 