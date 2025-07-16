# Document MCP Server

[![PyPI version](https://badge.fury.io/py/document-mcp.svg)](https://badge.fury.io/py/document-mcp)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

A Model Context Protocol (MCP) server for managing structured Markdown documents with built-in safety features. This server provides tools to create, read, update, and analyze documents composed of multiple chapters, with write-safety, automatic snapshots, and version control.

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
    ���── 01-methodology.md
    └── 02-results.md
```

## New Features

### Unified Tools
To improve usability and reduce the number of tools, several granular tools have been consolidated into unified, action-based tools. For example, `snapshot_document`, `list_snapshots`, and `restore_snapshot` are now all handled by the `manage_snapshots` tool. This provides a more consistent and intuitive API for interacting with the server.

### Batch Processing
The server now supports batch processing of operations via the `apply_batch_operations` tool. This allows multiple operations to be executed sequentially in a single request, with support for atomic transactions and rollbacks on failure. This is ideal for complex, multi-step modifications to documents.

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

The server exposes 23 MCP tools organized in 6 functional categories via the Model Context Protocol:

### Document Tools (4 tools)

| Tool | Parameters | Description |
|------|------------|-------------|
| `list_documents` | - | Lists all available documents with metadata |
| `create_document` | `document_name: str` | Creates a new document directory |
| `delete_document` | `document_name: str` | Deletes a document and all its chapters |
| `read_document_summary` | `document_name: str` | Retrieve the content of a document's summary file (_SUMMARY.md). |

### Chapter Tools (5 tools)

| Tool | Parameters | Description |
|------|------------|-------------|
| `list_chapters` | `document_name: str` | Lists all chapters in a document, ordered by filename |
| `create_chapter` | `document_name: str`, `chapter_name: str`, `initial_content: str = ""` | Creates a new chapter file |
| `delete_chapter` | `document_name: str`, `chapter_name: str` | Deletes a chapter from a document |
| `write_chapter_content` | `document_name: str`, `chapter_name: str`, `new_content: str` | Overwrites the entire content of a chapter |
| `append_to_chapter_content` | `document_name: str`, `chapter_name: str`, `content_to_append: str` | Append content to an existing chapter file without replacing it. |

### Paragraph Tools (7 tools)

| Tool | Parameters | Description |
|------|------------|-------------|
| `replace_paragraph` | `document_name: str`, `chapter_name: str`, `paragraph_index: int`, `new_content: str` | Replaces a specific paragraph with new content. |
| `insert_paragraph_before` | `document_name: str`, `chapter_name: str`, `paragraph_index: int`, `new_content: str` | Inserts a new paragraph before the specified index. |
| `insert_paragraph_after` | `document_name: str`, `chapter_name: str`, `paragraph_index: int`, `new_content: str` | Inserts a new paragraph after the specified index. |
| `delete_paragraph` | `document_name: str`, `chapter_name: str`, `paragraph_index: int` | Deletes a specific paragraph by index. |
| `move_paragraph_before` | `document_name: str`, `chapter_name: str`, `paragraph_to_move_index: int`, `target_paragraph_index: int` | Moves a paragraph to appear before another paragraph. |
| `move_paragraph_to_end` | `document_name: str`, `chapter_name: str`, `paragraph_to_move_index: int` | Moves a paragraph to the end of the chapter. |
| `append_paragraph_to_chapter` | `document_name: str`, `chapter_name: str`, `paragraph_content: str` | Appends a new paragraph to the end of a chapter |

### Content Tools (4 tools)

| Tool | Parameters | Description |
|------|------------|-------------|
| `read_content` | `document_name: str`, `scope: str`, `chapter_name: str?`, `paragraph_index: int?` | Unified tool to read content at the document, chapter, or paragraph level. |
| `find_text` | `document_name: str`, `search_text: str`, `scope: str`, `chapter_name: str?`, `case_sensitive: bool?` | Unified tool to find text at the document or chapter level. |
| `replace_text` | `document_name: str`, `find_text: str`, `replace_text: str`, `scope: str`, `chapter_name: str?` | Unified tool to replace text at the document or chapter level. |
| `get_statistics` | `document_name: str`, `scope: str`, `chapter_name: str?` | Unified tool to get statistics for a document or chapter. |

### Safety Tools (3 tools)

| Tool | Parameters | Description |
|------|------------|-------------|
| `manage_snapshots` | `document_name: str`, `action: str`, `snapshot_id: str?`, `message: str?`, `auto_cleanup: bool?` | Unified tool to create, list, and restore snapshots. |
| `check_content_status` | `document_name: str`, `chapter_name: str?`, `include_history: bool?`, `time_window: str?`, `last_known_modified: str?` | Unified tool to check content freshness and get modification history. |
| `diff_content` | `document_name: str`, `source_type: str`, `source_id: str?`, `target_type: str`, `target_id: str?`, `output_format: str?`, `chapter_name: str?` | Unified tool to compare content between snapshots, files, or current state. |

### Batch Tools (1 tool)

| Tool | Parameters | Description |
|------|------------|-------------|
| `batch_apply_operations` | `operations: List[Dict]`, `atomic: bool?`, `validate_only: bool?`, `snapshot_before: bool?`, `continue_on_error: bool?` | Executes a batch of operations with support for atomic transactions and rollbacks. |

## Data Models

The server uses Pydantic models for structured data exchange:

- `DocumentInfo`: Metadata for a document
- `ChapterMetadata`: Metadata for a chapter
- `ChapterContent`: Full content and metadata of a chapter
- `ParagraphDetail`: Information about a specific paragraph
- `FullDocumentContent`: Complete content of a document
- `DocumentSummary`: Content of a document's summary file
- `StatisticsReport`: Word and paragraph count statistics
- `OperationStatus`: Success/failure status for operations
- `ContentFreshnessStatus`: Information about content freshness
- `ModificationHistory`: List of modification entries
- `SnapshotInfo`: Metadata for a single snapshot
- `SnapshotsList`: A list of snapshots for a document
- `BatchOperation`: A single operation within a batch
- `BatchApplyResult`: The result of a batch operation

## Requirements

- Python 3.8+
- fastapi
- uvicorn[standard]
- pydantic-ai
- mcp[cli]
- python-dotenv
- google-generativeai

## Testing

The MCP server uses a three-tier testing strategy:

1. **Unit Tests**: Mock all dependencies for fast, reliable component testing
2. **Integration Tests**: Real MCP server with mocked AI for tool validation
3. **E2E Tests**: Real MCP server with real AI for complete system validation (runs in CI/CD)

Tests cover all MCP tools, error handling, boundary conditions, and multi-step workflows. For more details on the testing strategy, see the [Testing Guidelines](https://github.com/clchinkc/document-mcp/blob/main/tests/README.md) in the main project repository.

## Examples and Documentation

For comprehensive examples, tutorials, and usage guides, visit the [GitHub repository](https://github.com/clchinkc/document-mcp).

## License

MIT License

## Links

- **GitHub Repository**: [https://github.com/clchinkc/document-mcp](https://github.com/clchinkc/document-mcp)
- **Bug Reports**: [GitHub Issues](https://github.com/clchinkc/document-mcp/issues)

## Enhanced Error Handling and Logging

The Document MCP Server features advanced error handling and structured logging for operational insights and debugging:

### Logging Architecture

The server uses a dual-logging approach:

1. **MCP Call Logs** (`mcp_calls.log`): Traditional format logs for MCP tool calls and operations
2. **Structured Error Logs** (`errors.log`): JSON-formatted logs for detailed error analysis

### Error Categories

All errors are categorized for proper prioritization:

- **CRITICAL**: System-breaking errors requiring immediate attention
- **ERROR**: Functional errors that prevent operation completion  
- **WARNING**: Non-blocking issues that should be monitored
- **INFO**: Informational messages for debugging and expected conditions

### Structured Error Context

Each error log includes comprehensive context:

```json
{
  "timestamp": "2024-01-15T10:30:00.000Z",
  "level": "ERROR",
  "logger": "error_logger",
  "message": "Failed to read chapter file: 01-intro.md",
  "error_category": "ERROR",
  "operation": "read_chapter_content",
  "document_name": "user_guide",
  "chapter_file_path": "/path/to/user_guide/01-intro.md",
  "file_exists": false,
  "exception": {
    "type": "FileNotFoundError",
    "message": "No such file or directory",
    "traceback": ["...", "..."]
  }
}
```

### Integration with Log Analysis Tools

#### ELK Stack (Elasticsearch, Logstash, Kibana)

1. **Logstash Configuration** (`logstash.conf`):
```ruby
input {
  file {
    path => "/path/to/document_mcp/errors.log"
    start_position => "beginning"
    codec => "json"
  }
}

filter {
  date {
    match => [ "timestamp", "ISO8601" ]
  }
  
  if [error_category] == "CRITICAL" {
    mutate {
      add_tag => ["alert"]
    }
  }
}

output {
  elasticsearch {
    hosts => ["localhost:9200"]
    index => "document-mcp-errors-%{+YYYY.MM.dd}"
  }
}
```

2. **Kibana Dashboards**: Create visualizations for:
   - Error trends by category
   - Top failing operations
   - Document/chapter error hotspots
   - Exception type distribution

#### Splunk Integration

1. **Input Configuration** (`inputs.conf`):
```ini
[monitor:///path/to/document_mcp/errors.log]
disabled = false
index = document_mcp
sourcetype = json_auto
```

2. **Search Queries**:
```splunk
# Critical errors in last 24h
index=document_mcp error_category=CRITICAL earliest=-24h

# Most common error operations
index=document_mcp | stats count by operation | sort -count

# File I/O errors by document
index=document_mcp operation="read_*" level=ERROR | stats count by document_name
```

#### Prometheus/Grafana Integration

Create custom metrics collector to parse JSON logs:

```python
from prometheus_client import Counter, Histogram, start_http_server
import json
import time

error_counter = Counter('document_mcp_errors_total', 
                       'Total errors', ['category', 'operation'])

def parse_log_line(line):
    try:
        log_entry = json.loads(line)
        if log_entry.get('error_category'):
            error_counter.labels(
                category=log_entry['error_category'],
                operation=log_entry.get('operation', 'unknown')
            ).inc()
    except json.JSONDecodeError:
        pass
```

#### Custom Log Analysis

For custom analysis tools, the JSON structure enables easy querying:

```python
import json
import pandas as pd

# Load and analyze error patterns
with open('errors.log', 'r') as f:
    logs = [json.loads(line) for line in f if line.strip()]

df = pd.DataFrame(logs)

# Analyze error trends
error_trends = df.groupby(['error_category', 'operation']).size()
print(error_trends)

# Find problematic documents
doc_errors = df[df['document_name'].notna()].groupby('document_name').size()
print(doc_errors.sort_values(ascending=False))
```

### Monitoring and Alerting

Set up alerts based on error patterns:

1. **Critical Error Alerts**: Immediate notification for CRITICAL category errors
2. **Error Rate Alerts**: Alert when error rate exceeds threshold (e.g., >10 errors/minute)
3. **Operation-Specific Alerts**: Monitor specific operations like file I/O for frequent failures
4. **Document Health**: Alert when specific documents show high error rates

### Log Retention and Rotation

Both log files use rotating file handlers:
- Maximum file size: 10MB
- Backup count: 5 files
- Total storage: ~50MB per log type

Configure your log analysis tools to handle log rotation appropriately.