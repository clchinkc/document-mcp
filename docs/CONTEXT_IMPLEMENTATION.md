# Phase 4.3: OneContext-Inspired Context Management Implementation

## Overview

Phase 4.3 implements OneContext-inspired context management for the Document MCP system. This system provides persistent session-based context management with memory storage, retrieval, export/import, and lifecycle tracking.

The implementation includes:
- **6 MCP Tools** for memory and context operations
- **Context Directory Structure** with `.context/` namespace
- **Pydantic Models** for type safety and validation
- **Session Lifecycle Management** for multi-session workflows
- **Multiple Export Formats** (JSON, YAML, Markdown)
- **Comprehensive Testing** (45+ test cases)

## Architecture Overview

### Storage Structure

Each document gets a `.context/` directory parallel to chapters:

```
document_name/
├── 01-chapter.md
├── 02-chapter.md
├── .context/                    # Context management directory
│   ├── session.json            # Session metadata (auto-created)
│   ├── goals.md               # Document goals (optional)
│   ├── decisions.md           # Key decisions (optional)
│   ├── blockers.md            # Current blockers (optional)
│   ├── memories/              # Memory entries directory
│   │   ├── goal_key.json
│   │   ├── character_name.json
│   │   └── blocker_issue_1.json
│   └── exports/               # Context exports directory
│       ├── context_2026-02-25.json
│       └── backup.yaml
```

### Core Components

#### 1. Models (document_mcp/models/context.py)

**MemoryEntry**: Key-value memory with metadata
```python
MemoryEntry(
    key: str                          # Unique identifier
    value: Any                        # Stored value (any JSON-serializable)
    stored_at: datetime              # Creation timestamp
    retrieved_at: datetime | None    # Last access timestamp
    tags: list[str]                  # Organization tags
    expires: datetime | None         # Optional expiration
    metadata: dict[str, Any]         # Custom metadata
)
```

**SessionMetadata**: Multi-session tracking
```python
SessionMetadata(
    session_id: str                  # Unique session ID
    document_name: str              # Associated document
    started_at: datetime            # Session start
    last_activity: datetime         # Last operation time
    goals: list[str]               # Session goals
    progress: dict[str, Any]       # Progress tracking
    blockers: list[str]            # Current blockers
    metadata: dict[str, Any]       # Custom metadata
)
```

**ExportStatus & ImportStatus**: Operation results

#### 2. Utilities (document_mcp/utils/context_manager.py)

Core functions for context operations:

```python
# Session management
initialize_session(document_name, session_id=None)
load_session(document_name)
update_session(document_name, session)

# Memory operations
store_memory(document_name, key, value, tags=None, expires=None, metadata=None)
recall_memory(document_name, key, pattern=None)
list_memories(document_name, tags=None)
delete_memory(document_name, key)

# Import/Export
export_context(document_name, export_path, format_type="json")
import_context(document_name, context_file, merge=False)
```

#### 3. MCP Tools (document_mcp/tools/context_tools.py)

**6 Tools Registered with MCP Server:**

1. **store_memory** - Store key-value memories with tags and metadata
2. **recall_memory** - Retrieve and update memory access timestamp
3. **list_memories** - List all memories with optional tag filtering
4. **delete_memory** - Remove memory entry
5. **export_context** - Export to JSON/YAML/Markdown
6. **import_context** - Import from external files with validation

## Usage Guide

### 1. Storing Memories

Store session state, goals, decisions, and context:

```python
# Simple memory
store_memory("novel_draft", "current_chapter", "Chapter 5: Discovery")

# With tags for organization
store_memory(
    "novel_draft",
    "blocker_motivation",
    "Need to strengthen character motivation",
    tags=["blocking", "urgent"]
)

# Complex structured data
store_memory(
    "novel_draft",
    "character_arc_marcus",
    {
        "name": "Marcus Chen",
        "arc": "Detective -> Villain",
        "motivation": "Revenge",
        "keyScenes": ["Betrayal", "Realization", "Redemption"]
    },
    tags=["character", "major"],
    metadata={"priority": "high", "lastReview": "2026-02-25"}
)

# With expiration
store_memory(
    "novel_draft",
    "sprint_goal_week_1",
    "Complete first draft of Act 1",
    expires=datetime.datetime.utcnow() + datetime.timedelta(days=7)
)
```

### 2. Recalling Memories

Retrieve stored context and track usage:

```python
# Recall specific memory
memory = recall_memory("novel_draft", "current_chapter")
if memory:
    print(f"Current chapter: {memory.value}")
    print(f"Last accessed: {memory.retrieved_at}")

# List all memories
all_memories = list_memories("novel_draft")
for mem in all_memories:
    print(f"{mem.key}: {mem.value}")

# List with tag filtering
blockers = list_memories("novel_draft", tags=["blocking"])
decisions = list_memories("novel_draft", tags=["decision"])
```

### 3. Exporting Context

Backup and share context across documents:

```python
# Export to JSON (default, machine-readable)
status = export_context(
    "novel_draft",
    Path(".context/exports/backup.json"),
    format_type="json"
)
if status.success:
    print(f"Exported {status.entry_count} memories")

# Export to YAML (human-friendly)
export_context("novel_draft", Path(".context/backup.yaml"), format_type="yaml")

# Export to Markdown (for review/documentation)
export_context("novel_draft", Path(".context/review.md"), format_type="markdown")
```

### 4. Importing Context

Restore or merge context:

```python
# Import without merge (safe, skips conflicts)
status = import_context("novel_v2", Path(".context/backup.json"), merge=False)
if status.conflicts_detected:
    print(f"Found {status.conflicts_detected} conflicts")
    for conflict in status.conflict_details:
        print(f"  - {conflict['key']}: {conflict['reason']}")

# Import with merge (overwrites existing)
status = import_context("novel_v2", Path(".context/backup.json"), merge=True)
if status.success:
    print(f"Imported {status.entries_imported} memories")
```

### 5. Session Lifecycle

Auto-managed sessions track multi-session workflows:

```python
# Session auto-initializes on first memory store
store_memory("doc", "key", "value")

# Load existing session
session = load_session("doc")
if session:
    print(f"Session {session.session_id} started at {session.started_at}")
    print(f"Goals: {session.goals}")

# Update session with goals and blockers
session.goals = ["Complete draft", "Refine characters"]
session.blockers = ["Character motivation", "Plot hole in Act 2"]
session.progress = {"chapters_written": 3, "revision_passes": 1}
update_session("doc", session)
```

## API Reference

### store_memory

Store a memory entry with optional metadata.

**Parameters:**
- `document_name` (str): Target document
- `key` (str): Unique memory identifier
- `value` (Any): Value to store (JSON-serializable)
- `tags` (list[str], optional): Organization tags
- `expires_days` (int, optional): Days until expiration
- `metadata` (dict, optional): Custom metadata

**Returns:** MemoryEntry

**Example:**
```python
entry = store_memory(
    "novel",
    "pov_character",
    "Marcus Chen",
    tags=["protagonist"],
    metadata={"archetype": "the detective"}
)
```

### recall_memory

Retrieve a memory and update its access timestamp.

**Parameters:**
- `document_name` (str): Source document
- `key` (str): Memory key
- `pattern` (str, optional): Reserved for pattern matching

**Returns:** MemoryEntry | None

**Example:**
```python
memory = recall_memory("novel", "pov_character")
if memory:
    print(memory.value)  # "Marcus Chen"
```

### list_memories

List all memories with optional tag filtering.

**Parameters:**
- `document_name` (str): Source document
- `tags` (list[str], optional): Tags to filter by (OR logic)

**Returns:** list[MemoryEntry]

**Example:**
```python
urgent = list_memories("novel", tags=["urgent", "blocking"])
```

### delete_memory

Remove a memory entry.

**Parameters:**
- `document_name` (str): Source document
- `key` (str): Memory key to delete

**Returns:** OperationStatus

**Example:**
```python
status = delete_memory("novel", "outdated_note")
```

### export_context

Export context to external format.

**Parameters:**
- `document_name` (str): Document to export from
- `export_filename` (str, optional): Output filename
- `format_type` (str): "json", "yaml", or "markdown"

**Returns:** ExportStatus

**Example:**
```python
status = export_context(
    "novel",
    format_type="json",
    export_filename="context_backup.json"
)
```

### import_context

Import context from external file.

**Parameters:**
- `document_name` (str): Target document
- `context_file` (str): Source file path
- `merge` (bool): Merge or replace (default: False)

**Returns:** ImportStatus

**Example:**
```python
status = import_context(
    "novel_v2",
    "context_backup.json",
    merge=True
)
```

## Data Formats

### JSON Export Format

```json
{
  "document_name": "novel_draft",
  "exported_at": "2026-02-25T14:30:00.000000",
  "session": {
    "session_id": "session_abc123",
    "document_name": "novel_draft",
    "started_at": "2026-02-25T10:00:00.000000",
    "last_activity": "2026-02-25T14:30:00.000000",
    "goals": ["Complete draft", "Refine characters"],
    "progress": {"chapters": 5},
    "blockers": ["Motivation issue"],
    "metadata": {}
  },
  "memories": [
    {
      "key": "current_chapter",
      "value": "Chapter 5: The Discovery",
      "stored_at": "2026-02-25T10:15:00.000000",
      "retrieved_at": "2026-02-25T14:30:00.000000",
      "tags": ["progress", "current"],
      "expires": null,
      "metadata": {}
    }
  ],
  "goals": "# Goals\n\n- Complete first draft...",
  "decisions": "# Decisions\n\n- Changed ending...",
  "blockers": "# Blockers\n\n- Character motivation..."
}
```

### YAML Export Format

```yaml
document_name: novel_draft
exported_at: 2026-02-25T14:30:00.000000
session:
  session_id: session_abc123
  document_name: novel_draft
  started_at: 2026-02-25T10:00:00.000000
  goals:
    - Complete draft
    - Refine characters
memories:
  - key: current_chapter
    value: "Chapter 5: The Discovery"
    stored_at: 2026-02-25T10:15:00.000000
    tags: [progress, current]
```

### Markdown Export Format

```markdown
# Context Export: novel_draft

**Exported at:** 2026-02-25T14:30:00.000000

## Session

- **ID:** session_abc123
- **Started:** 2026-02-25T10:00:00.000000
- **Goals:** 2 items
- **Blockers:** 1 items

## Memories

Total: 5 entries

### current_chapter
- **Tags:** progress, current
- **Stored:** 2026-02-25T10:15:00.000000

## Goals

# Goals
...
```

## Implementation Details

### Context Directory Auto-Creation

The `.context/` directory is automatically created when:
- First memory is stored
- Session is initialized
- Import operation begins

### Session Auto-Initialization

A session is automatically created when:
- First `store_memory` is called
- `initialize_session` is explicitly called

Each session has a unique `session_id` (auto-generated unless specified).

### Memory File Naming

Memory keys are converted to safe filenames:
- Special characters (`/`, `\`, `:`, spaces) → `_`
- Filenames limited to 200 characters
- Stored as `{safe_key}.json` in `memories/` directory

### Import Conflict Resolution

When importing without merge (`merge=False`):
- Conflicts are detected for existing keys
- Existing data is preserved
- Conflicts are reported in ImportStatus
- Operation succeeds but documents conflicts

When importing with merge (`merge=True`):
- Existing keys are overwritten
- No conflicts reported
- All entries are imported

### Export Formats

**JSON**
- Machine-readable with full structure
- Preserves all types (lists, dicts, dates as ISO strings)
- Best for programmatic access

**YAML**
- Human-friendly format
- Good for review and editing
- Portable across systems

**Markdown**
- Documentation format
- Best for review and sharing
- Readable without parsing

## Testing

Comprehensive test suite with 45+ test cases:

```bash
# Run all context tests
pytest tests/unit/test_context_tools.py -v

# Run specific test class
pytest tests/unit/test_context_tools.py::TestMemoryStorage -v

# Run specific test
pytest tests/unit/test_context_tools.py::TestMemoryStorage::test_store_memory_creates_file -v
```

Test coverage includes:
- Context directory initialization
- Session lifecycle (create, load, update)
- Memory operations (store, recall, list, delete)
- Edge cases (unicode, special characters, large values)
- Export/import workflows
- Conflict detection and resolution
- Format conversion and validation

## Integration with MCP

All 6 tools are registered with the FastMCP server:

```python
from story_mcp.tools import register_context_tools
from mcp.server import FastMCP

server = FastMCP()
register_context_tools(server)
```

Tools are available immediately for use with:
- Simple Agent
- ReAct Agent
- Direct MCP client access

## Best Practices

### 1. Memory Organization

Use tags consistently for easy retrieval:

```python
# Good: Consistent tag naming
store_memory(doc, "goal_1", value, tags=["goal"])
store_memory(doc, "goal_2", value, tags=["goal"])

# Query by tag
goals = list_memories(doc, tags=["goal"])
```

### 2. Meaningful Keys

Use descriptive, hierarchical keys:

```python
# Good: Self-documenting
store_memory(doc, "character/marcus/motivation", "revenge")
store_memory(doc, "setting/act1/location", "station")

# Less good: Cryptic
store_memory(doc, "c1", "revenge")
store_memory(doc, "s1l1", "station")
```

### 3. Metadata for Context

Use metadata for queryable attributes:

```python
store_memory(
    doc,
    "scene_1",
    "The discovery",
    tags=["scene"],
    metadata={
        "act": 1,
        "importance": "high",
        "status": "draft",
        "word_count": 1500
    }
)
```

### 4. Export Before Major Changes

Export context before significant document changes:

```python
# Backup before rewriting
export_context(doc, Path(".context/pre_rewrite_backup.json"))

# Make changes...

# Can restore if needed
import_context(doc, Path(".context/pre_rewrite_backup.json"))
```

### 5. Regular Cleanup

Remove expired or outdated memories:

```python
# List and delete outdated memories
outdated = list_memories(doc, tags=["temp"])
for mem in outdated:
    if should_delete(mem):
        delete_memory(doc, mem.key)
```

## Zero Breaking Changes

This implementation:
- Adds new `.context/` directory structure (non-intrusive)
- Adds 6 new MCP tools (extends existing tools)
- No changes to existing document structure
- No changes to existing tool behavior
- Fully backward compatible with existing documents

Existing documents can start using context management at any time without migration.

## Future Enhancements

Potential Phase 4.4+ improvements:
- Pattern-based memory search with fuzzy matching
- Memory embedding-based semantic search
- Context versioning with rollback
- Multi-document context aggregation
- Memory expiration automation
- Context templates for common workflows
- Memory access analytics and usage patterns

## Summary

Phase 4.3 delivers a complete OneContext-inspired context management system with:

✅ 4 core memory operations (store, recall, list, delete)
✅ 2 import/export tools with 3 format support
✅ Auto-managed session tracking
✅ Tag-based organization
✅ Metadata support
✅ Conflict detection and resolution
✅ 45+ comprehensive tests
✅ Production-ready implementation
✅ Zero breaking changes

The system is ready for immediate use and integration with existing documents.
