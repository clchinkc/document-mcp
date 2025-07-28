# Document-MCP Technical Reference

This document provides detailed technical information about the Document-MCP server, including the full list of MCP tools, data models, and advanced features.

## MCP Tools Reference

The server exposes 26 MCP tools organized in 5 functional categories via the Model Context Protocol. All tools include comprehensive error handling, structured logging, and automatic safety features.

**Key Features:**
- 🛡️ **Universal Safety**: Automatic snapshots for all write operations
- 📊 **Structured Logging**: Comprehensive operation tracking with OpenTelemetry
- 🎯 **Scope-Based Operations**: Unified API for document/chapter/paragraph operations
- 🔍 **Semantic Search**: AI-powered content discovery with embedding cache

### Document Tools (6 tools)

| Tool | Parameters | Description |
|------|------------|-------------|
| `list_documents` | - | Lists all available documents with metadata |
| `create_document` | `document_name: str` | Creates a new document directory |
| `delete_document` | `document_name: str` | Deletes a document and all its chapters |
| `read_summary` | `document_name: str`, `scope: str = "document"`, `target_name: str \| None = None` | Read summary files with flexible scope (document, chapter, section) |
| `write_summary` | `document_name: str`, `summary_content: str`, `scope: str = "document"`, `target_name: str \| None = None` | Write or update summary files with flexible scope (document, chapter, section) |
| `list_summaries` | `document_name: str` | List all available summary files for a document |

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

### Content Tools (5 tools)

| Tool | Parameters | Description |
|------|------------|-------------|
| `read_content` | `document_name: str`, `scope: str`, `chapter_name: str?`, `paragraph_index: int?` | Unified tool to read content at the document, chapter, or paragraph level. |
| `find_text` | `document_name: str`, `search_text: str`, `scope: str`, `chapter_name: str?`, `case_sensitive: bool?` | Unified tool to find text at the document or chapter level. |
| `find_similar_text` | `document_name: str`, `query_text: str`, `scope: str`, `chapter_name: str?`, `similarity_threshold: float?`, `max_results: int?` | Semantic search using embeddings to find contextually similar content with similarity scoring. |
| `replace_text` | `document_name: str`, `find_text: str`, `replace_text: str`, `scope: str`, `chapter_name: str?` | Unified tool to replace text at the document or chapter level. |
| `get_statistics` | `document_name: str`, `scope: str`, `chapter_name: str?` | Unified tool to get statistics for a document or chapter. |

### Safety Tools (3 tools)

| Tool | Parameters | Description |
|------|------------|-------------|
| `manage_snapshots` | `document_name: str`, `action: str`, `snapshot_id: str?`, `message: str?`, `auto_cleanup: bool?` | Unified tool to create, list, and restore snapshots. |
| `check_content_status` | `document_name: str`, `chapter_name: str?`, `include_history: bool?`, `time_window: str?`, `last_known_modified: str?` | Unified tool to check content freshness and get modification history. |
| `diff_content` | `document_name: str`, `source_type: str`, `source_id: str?`, `target_type: str`, `target_id: str?`, `output_format: str?`, `chapter_name: str?` | Unified tool to compare content between snapshots, files, or current state. |

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
- `DocumentSummary`: Summary content with scope and target information

## Fine-Grain Summary System

The Document MCP system implements a sophisticated summary management system that supports multiple scopes and organized storage:

### Summary Scopes

- **Document Scope** (`scope="document"`): Overall document summaries stored as `summaries/document.md`
- **Chapter Scope** (`scope="chapter"`, `target_name="chapter_name"`): Chapter-specific summaries stored as `summaries/{chapter_name}`
- **Section Scope** (`scope="section"`, `target_name="section_name"`): Thematic summaries stored as `summaries/{section_name}.md`

### Storage Organization

```
document_name/
├── 01-chapter.md
├── 02-chapter.md
└── summaries/          # Organized summary storage
    ├── document.md     # Document-level summary
    ├── 01-chapter.md   # Chapter summary
    └── overview.md     # Section summary
```

### Key Features

- **Organized Storage**: Clean separation from content files in dedicated `summaries/` directory
- **Flexible Scoping**: Support for document, chapter, and section-level summaries
- **No Legacy Support**: No backward compatibility with old `_SUMMARY.md` files
- **Safety Features**: Automatic snapshot protection for all summary modifications

## Requirements

- Python 3.10+
- fastapi
- uvicorn[standard]
- pydantic-ai
- mcp[cli]
- python-dotenv
- google-generativeai
- openai

## Development Toolchain

This project uses modern Python development tools for enhanced performance:

- **`uv`**: Ultra-fast Python package manager (10-100x faster than pip)
- **`ruff`**: Lightning-fast linter and formatter (replaces black, isort, flake8, etc.)
- **`mypy`**: Static type checking for enhanced code quality
- **`pytest`**: Comprehensive testing framework with async support

### Development Commands

```bash
# Install dev dependencies
uv sync --all-extras

# Code quality checks
uv run ruff check                    # Lint code
uv run ruff check --fix              # Auto-fix issues
uv run ruff format                   # Format code
uv run mypy document_mcp/            # Type checking

# Run tests
uv run python -m pytest             # All tests
uv run python -m pytest tests/unit/ # Unit tests only
```

## Testing

The MCP server uses a four-tier testing strategy:

1. **Unit Tests**: Mock all dependencies for fast, reliable component testing
2. **Integration Tests**: Real MCP server with mocked AI for tool validation
3. **E2E Tests**: Real MCP server with real AI for complete system validation (runs in CI/CD)
4. **Evaluation Tests**: Performance benchmarking and prompt optimization

Tests cover all MCP tools, error handling, boundary conditions, and multi-step workflows. For more details on the testing strategy, see the [Testing Guidelines](https://github.com/clchinkc/document-mcp/blob/main/tests/README.md) in the main project repository.

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
