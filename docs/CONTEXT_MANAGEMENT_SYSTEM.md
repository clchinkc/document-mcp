# Context Management System Design (Phase 4.3)

## Executive Summary

This document specifies the design for a **OneContext-inspired persistent memory system** for Document MCP. The system enables agents to maintain cross-session memory across multiple document operations, improving contextual understanding and enabling stateful workflows.

**Core Goal**: Enable agents to store, retrieve, and manage contextual information across sessions with minimal complexity.

**Key Features**:
- Persistent memory storage with JSON serialization
- Session metadata tracking (timestamps, session IDs, operation counts)
- Context export/import for sharing and backup
- Simple, straightforward tool interfaces
- Zero external dependencies (uses only stdlib + existing Pydantic models)

---

## 1. Data Model Design

### 1.1 Memory Entry Structure

Each memory entry is an atomic unit of stored context.

```python
# document_mcp/models/context.py

from __future__ import annotations

import datetime
from typing import Any
from pydantic import BaseModel, Field


class MemoryEntry(BaseModel):
    """Represents a single unit of stored memory/context."""

    # Unique identifier
    memory_id: str = Field(
        ...,
        description="Unique identifier for this memory (UUID)"
    )

    # Semantic categorization
    key: str = Field(
        ...,
        description="Semantic key/namespace (e.g., 'document_state', 'workflow_context', 'user_preference')"
    )

    # Actual content
    value: dict[str, Any] = Field(
        ...,
        description="Arbitrary structured data (JSON-serializable)"
    )

    # Metadata
    scope: str = Field(
        default="global",
        description="Scope: 'global', 'document', 'session' (determines visibility/lifecycle)"
    )

    document_name: str | None = Field(
        default=None,
        description="If scope='document', the target document"
    )

    # Lifecycle
    created_at: datetime.datetime = Field(
        default_factory=datetime.datetime.now,
        description="Creation timestamp"
    )

    updated_at: datetime.datetime = Field(
        default_factory=datetime.datetime.now,
        description="Last update timestamp"
    )

    expires_at: datetime.datetime | None = Field(
        default=None,
        description="Optional expiration timestamp (for temporary memory)"
    )

    # Tracking
    source: str = Field(
        default="manual",
        description="Where memory came from: 'manual', 'agent', 'automatic', 'import'"
    )

    agent_id: str | None = Field(
        default=None,
        description="Which agent stored this memory"
    )

    tags: list[str] = Field(
        default_factory=list,
        description="Optional tags for filtering/search"
    )


class MemoryStore(BaseModel):
    """Complete in-memory representation of all stored memories."""

    memories: list[MemoryEntry] = Field(
        default_factory=list,
        description="All stored memory entries"
    )

    version: int = Field(
        default=1,
        description="Memory store version for future schema evolution"
    )

    last_updated: datetime.datetime = Field(
        default_factory=datetime.datetime.now,
        description="When store was last modified"
    )

    total_memories: int = Field(
        default=0,
        description="Total count of stored memories"
    )
```

### 1.2 Session Metadata Structure

Track session information for context and analytics.

```python
# document_mcp/models/context.py (continued)

class SessionMetadata(BaseModel):
    """Metadata about a session where operations occurred."""

    session_id: str = Field(
        ...,
        description="Unique session identifier"
    )

    created_at: datetime.datetime = Field(
        default_factory=datetime.datetime.now,
        description="Session start time"
    )

    closed_at: datetime.datetime | None = Field(
        default=None,
        description="Session end time (if closed)"
    )

    agent_id: str | None = Field(
        default=None,
        description="Which agent initiated this session"
    )

    operations_count: int = Field(
        default=0,
        description="Number of operations performed in session"
    )

    documents_accessed: list[str] = Field(
        default_factory=list,
        description="List of document names accessed"
    )

    memory_created: int = Field(
        default=0,
        description="Number of memories created in this session"
    )

    memory_recalled: int = Field(
        default=0,
        description="Number of memories recalled in this session"
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary session-specific metadata"
    )


class SessionHistory(BaseModel):
    """History of sessions for audit trail and analytics."""

    sessions: list[SessionMetadata] = Field(
        default_factory=list,
        description="All session records"
    )

    current_session_id: str | None = Field(
        default=None,
        description="ID of current active session (if any)"
    )

    total_sessions: int = Field(
        default=0,
        description="Total sessions recorded"
    )
```

### 1.3 Context Export Structure

Format for exporting/importing context (for sharing and backup).

```python
# document_mcp/models/context.py (continued)

class ContextSnapshot(BaseModel):
    """Complete context snapshot for export/import."""

    # Identification
    snapshot_id: str = Field(
        ...,
        description="Unique snapshot identifier"
    )

    created_at: datetime.datetime = Field(
        default_factory=datetime.datetime.now,
        description="When snapshot was created"
    )

    # Content
    memory_store: MemoryStore = Field(
        ...,
        description="Complete memory store at time of snapshot"
    )

    session_history: SessionHistory = Field(
        default_factory=SessionHistory,
        description="Session history included in snapshot"
    )

    # Metadata
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Snapshot metadata (e.g., 'reason', 'user_comment')"
    )

    version: int = Field(
        default=1,
        description="Snapshot format version"
    )
```

---

## 2. Tool Interface Definitions

### 2.1 store_memory()

**Purpose**: Store a new memory entry or update existing one.

```python
# document_mcp/tools/context_tools.py

@mcp_server.tool()
@log_mcp_call
def store_memory(
    key: str,
    value: dict[str, Any],
    scope: str = "global",
    document_name: str | None = None,
    expires_in_hours: int | None = None,
    tags: list[str] | None = None,
) -> OperationStatus:
    """Store or update a memory entry for cross-session persistence.

    Parameters:
        key (str): Semantic key/namespace for memory (e.g., 'document_structure', 'workflow_progress')
        value (dict): Arbitrary structured data to store (must be JSON-serializable)
        scope (str): Memory scope - 'global' (all sessions), 'document' (specific doc), 'session' (current only)
        document_name (str): Required if scope='document', target document
        expires_in_hours (int): Optional - memory auto-deletes after N hours
        tags (list[str]): Optional tags for filtering/categorization

    Returns:
        OperationStatus:
            success: bool - Whether store operation succeeded
            message: str - Human-readable status
            details: {
                "memory_id": str,
                "key": str,
                "scope": str,
                "created_new": bool,
                "updated_existing": bool,
                "expires_at": datetime | null
            }

    Example:
        ```json
        {
            "key": "book_structure",
            "value": {
                "chapters": ["intro", "main", "conclusion"],
                "total_words": 50000,
                "status": "in_progress"
            },
            "scope": "document",
            "document_name": "my_book",
            "tags": ["structure", "planning"]
        }
        ```

    Implementation Notes:
        - Auto-generates memory_id if new
        - Updates existing memory if key+scope+document_name matches
        - Records source as 'manual' (can be 'agent' if called by agent)
        - Respects expiration: stores expires_at timestamp
    """
```

### 2.2 recall_memory()

**Purpose**: Retrieve stored memories by various filters.

```python
# document_mcp/tools/context_tools.py (continued)

@mcp_server.tool()
@log_mcp_call
def recall_memory(
    key: str | None = None,
    scope: str = "global",
    document_name: str | None = None,
    tags: list[str] | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    """Recall stored memories with flexible filtering.

    Parameters:
        key (str): Optional - specific memory key to retrieve
        scope (str): Filter by scope ('global', 'document', 'session')
        document_name (str): If scope='document', which document
        tags (list[str]): Optional - filter by tags (OR logic - any match)
        limit (int): Max results to return (default 10)

    Returns:
        {
            "success": bool,
            "memories": [
                {
                    "memory_id": str,
                    "key": str,
                    "value": dict,
                    "scope": str,
                    "created_at": datetime,
                    "updated_at": datetime,
                    "expires_at": datetime | null,
                    "tags": list[str]
                }
                ...
            ],
            "total_found": int,
            "expired_removed": int  # Count of auto-deleted expired memories
        }

    Example Queries:
        ```json
        // Retrieve specific memory
        { "key": "book_structure" }

        // All memories for a document
        { "scope": "document", "document_name": "my_book" }

        // All global memories with 'planning' tag
        { "scope": "global", "tags": ["planning"] }

        // All memories (global scope)
        { "limit": 50 }
        ```

    Implementation Notes:
        - Auto-removes expired memories before returning
        - Returns up to 'limit' entries
        - If key specified, returns single entry or empty list
        - Tag filtering uses OR logic (any matching tag passes)
        - Created/updated_at help track memory freshness
    """
```

### 2.3 export_context()

**Purpose**: Export all context to JSON snapshot for sharing/backup.

```python
# document_mcp/tools/context_tools.py (continued)

@mcp_server.tool()
@log_mcp_call
def export_context(
    include_session_history: bool = True,
    scopes: list[str] | None = None,
    document_names: list[str] | None = None,
) -> OperationStatus:
    """Export context snapshot to file for backup or sharing.

    Parameters:
        include_session_history (bool): Whether to include session history (default True)
        scopes (list[str]): Filter - only export these scopes (e.g., ['global', 'document'])
        document_names (list[str]): Filter - only export memories from these documents

    Returns:
        OperationStatus:
            success: bool
            message: str
            details: {
                "snapshot_id": str,
                "file_path": str,
                "exported_memories": int,
                "exported_sessions": int,
                "file_size_bytes": int,
                "created_at": datetime
            }

    Example:
        ```json
        {
            "include_session_history": true,
            "scopes": ["global", "document"],
            "document_names": ["my_book"]
        }
        ```

    Implementation Notes:
        - Saves to .context/snapshots/ directory
        - File naming: snapshot_{timestamp}_{snapshot_id}.json
        - Includes ContextSnapshot with all metadata
        - Creates directory structure if missing
        - Returns absolute file path for user access
    """
```

### 2.4 import_context()

**Purpose**: Import context from JSON snapshot file.

```python
# document_mcp/tools/context_tools.py (continued)

@mcp_server.tool()
@log_mcp_call
def import_context(
    file_path: str,
    merge_mode: str = "add",
    remove_expired: bool = True,
) -> OperationStatus:
    """Import context snapshot from file.

    Parameters:
        file_path (str): Path to snapshot JSON file
        merge_mode (str): How to handle conflicts:
            - 'add': Add new memories, skip existing keys (non-destructive)
            - 'replace': Replace memories with same key (destructive)
            - 'merge': Merge updated_at - newer wins (recommended)
        remove_expired (bool): Auto-remove expired entries after import

    Returns:
        OperationStatus:
            success: bool
            message: str
            details: {
                "snapshot_id": str,
                "imported_memories": int,
                "imported_sessions": int,
                "merge_mode": str,
                "conflicts_skipped": int,
                "expired_removed": int
            }

    Example:
        ```json
        {
            "file_path": ".context/snapshots/snapshot_2025-02-25_abc123.json",
            "merge_mode": "merge"
        }
        ```

    Implementation Notes:
        - Validates snapshot JSON format
        - Respects merge_mode for conflict resolution
        - Auto-removes expired entries if requested
        - Updates session history with import record
        - Returns count of conflicts/merged/skipped items
    """
```

### 2.5 list_memories()

**Purpose**: List available memories with filtering and summary info.

```python
# document_mcp/tools/context_tools.py (continued)

@mcp_server.tool()
@log_mcp_call
def list_memories(
    scope: str = "global",
    document_name: str | None = None,
    include_expired: bool = False,
) -> dict[str, Any]:
    """List stored memories with summary information.

    Parameters:
        scope (str): Filter by scope ('global', 'document', 'session')
        document_name (str): If scope='document', target document
        include_expired (bool): Whether to show expired entries

    Returns:
        {
            "success": bool,
            "memories": [
                {
                    "memory_id": str,
                    "key": str,
                    "scope": str,
                    "document_name": str | null,
                    "created_at": datetime,
                    "updated_at": datetime,
                    "expires_at": datetime | null,
                    "tags": list[str],
                    "value_preview": str  # First 100 chars of stringified value
                }
            ],
            "total_count": int,
            "expired_count": int  # Only if include_expired=True
        }

    Implementation Notes:
        - Doesn't return full value (use recall_memory for that)
        - Shows value_preview for quick scanning
        - Helps understand memory organization
    """
```

### 2.6 delete_memory()

**Purpose**: Delete specific memory entries.

```python
# document_mcp/tools/context_tools.py (continued)

@mcp_server.tool()
@log_mcp_call
def delete_memory(
    memory_id: str | None = None,
    key: str | None = None,
    scope: str = "global",
) -> OperationStatus:
    """Delete stored memory entry(ies).

    Parameters:
        memory_id (str): Delete specific memory by ID (most precise)
        key (str): Delete all memories with this key in given scope (careful!)
        scope (str): Required if using key parameter

    Returns:
        OperationStatus:
            success: bool
            message: str
            details: {
                "deleted_count": int,
                "memory_ids": list[str]
            }

    Example:
        ```json
        // Delete specific memory
        { "memory_id": "uuid-here" }

        // Delete all global memories with key 'draft_notes'
        { "key": "draft_notes", "scope": "global" }
        ```

    Implementation Notes:
        - Either memory_id OR key required (mutual)
        - Deletion is permanent
        - Returns count of deleted items
    """
```

### 2.7 get_context_stats()

**Purpose**: Get statistics about stored context.

```python
# document_mcp/tools/context_tools.py (continued)

@mcp_server.tool()
@log_mcp_call
def get_context_stats() -> dict[str, Any]:
    """Get statistics about stored context.

    Returns:
        {
            "total_memories": int,
            "by_scope": {
                "global": int,
                "document": int,
                "session": int
            },
            "by_source": {
                "manual": int,
                "agent": int,
                "automatic": int,
                "import": int
            },
            "total_sessions": int,
            "current_session_active": bool,
            "memory_storage_bytes": int,
            "oldest_memory": datetime | null,
            "newest_memory": datetime | null,
            "expired_count": int,
            "unique_keys": int
        }

    Implementation Notes:
        - Provides comprehensive overview of context system state
        - Helps with space management and debugging
    """
```

---

## 3. Storage Backend Strategy

### 3.1 Directory Structure

```
.context/                          # Context storage root (parallel to .documents_storage/)
├── memories.json                  # Main memory store (all MemoryEntry objects)
├── sessions.json                  # Session history (SessionHistory)
├── snapshots/                     # Exported snapshots directory
│   ├── snapshot_2025-02-25_abc123.json
│   ├── snapshot_2025-02-25_def456.json
│   └── ...
└── index.json                     # Optional: Quick metadata index
    {
        "version": 1,
        "total_memories": 42,
        "last_updated": "2025-02-25T10:30:00Z",
        "snapshots_count": 3
    }
```

### 3.2 Storage Implementation

```python
# document_mcp/storage/context_storage.py

from pathlib import Path
import json
import datetime
from typing import Any
from document_mcp.models import MemoryEntry, MemoryStore, SessionHistory
from document_mcp.utils.file_operations import DOCS_ROOT_PATH


class ContextStorage:
    """Local filesystem storage for context/memory data."""

    def __init__(self, root_dir: str | None = None):
        """Initialize context storage.

        Args:
            root_dir: Parent directory (default uses parent of DOCS_ROOT_PATH)
        """
        if root_dir:
            self.root = Path(root_dir)
        else:
            # Use parent directory of documents storage
            self.root = Path(DOCS_ROOT_PATH).parent / ".context"

        self.root.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir = self.root / "snapshots"
        self.snapshots_dir.mkdir(exist_ok=True)

        self._memories_file = self.root / "memories.json"
        self._sessions_file = self.root / "sessions.json"
        self._index_file = self.root / "index.json"

    def load_memories(self) -> MemoryStore:
        """Load all memories from storage."""
        if not self._memories_file.exists():
            return MemoryStore()

        with open(self._memories_file, 'r') as f:
            data = json.load(f)
        return MemoryStore(**data)

    def save_memories(self, store: MemoryStore) -> None:
        """Save memories to storage."""
        store.last_updated = datetime.datetime.now()
        store.total_memories = len(store.memories)

        with open(self._memories_file, 'w') as f:
            json.dump(store.model_dump(mode='json'), f, indent=2, default=str)

        self._update_index()

    def load_sessions(self) -> SessionHistory:
        """Load session history from storage."""
        if not self._sessions_file.exists():
            return SessionHistory()

        with open(self._sessions_file, 'r') as f:
            data = json.load(f)
        return SessionHistory(**data)

    def save_sessions(self, history: SessionHistory) -> None:
        """Save session history to storage."""
        history.total_sessions = len(history.sessions)

        with open(self._sessions_file, 'w') as f:
            json.dump(history.model_dump(mode='json'), f, indent=2, default=str)

    def _update_index(self) -> None:
        """Update quick metadata index."""
        memories = self.load_memories()
        snapshots = list(self.snapshots_dir.glob("snapshot_*.json"))

        index = {
            "version": 1,
            "total_memories": len(memories.memories),
            "last_updated": datetime.datetime.now().isoformat(),
            "snapshots_count": len(snapshots)
        }

        with open(self._index_file, 'w') as f:
            json.dump(index, f, indent=2)

    @property
    def context_root(self) -> str:
        """Return root context directory path."""
        return str(self.root)


# Global instance
_context_storage: ContextStorage | None = None

def get_context_storage() -> ContextStorage:
    """Get global context storage instance."""
    global _context_storage
    if _context_storage is None:
        _context_storage = ContextStorage()
    return _context_storage
```

---

## 4. API Design

### 4.1 Core Operations Flow

```
Agent/User Query
    ↓
Query Context System (recall_memory)
    ↓
Get contextual information
    ↓
Perform Document Operations
    ↓
Store Results (store_memory)
    ↓
Next Session Uses Stored Memory
```

### 4.2 Agent Integration Example

```python
# In an agent (e.g., simple_agent or react_agent)

from document_mcp.mcp_client import DocumentMCPClient

client = DocumentMCPClient()

# Recall previous context
context = client.call_tool("recall_memory", {
    "key": "book_structure",
    "scope": "document",
    "document_name": "my_book"
})

if context["memories"]:
    book_structure = context["memories"][0]["value"]
    # Use structure for next operation
else:
    # First time - will analyze and store
    book_structure = analyze_document("my_book")
    client.call_tool("store_memory", {
        "key": "book_structure",
        "value": book_structure,
        "scope": "document",
        "document_name": "my_book",
        "tags": ["structure", "analysis"]
    })
```

### 4.3 Practical Use Cases

**Use Case 1: Workflow Progress Tracking**
```python
# Session 1: Start workflow
client.call_tool("store_memory", {
    "key": "document_editing_workflow",
    "value": {
        "status": "editing_chapters",
        "completed_chapters": ["01-intro"],
        "next_chapter": "02-main",
        "total_chapters": 5
    },
    "scope": "document",
    "document_name": "user_manual"
})

# Session 2: Resume workflow
progress = client.call_tool("recall_memory", {
    "key": "document_editing_workflow",
    "scope": "document",
    "document_name": "user_manual"
})
# Returns saved progress - agent knows where to continue
```

**Use Case 2: Document Analysis Caching**
```python
# Expensive analysis stored for reuse
client.call_tool("store_memory", {
    "key": "document_analysis_cache",
    "value": {
        "themes": ["resilience", "growth"],
        "character_count": 12,
        "plot_complexity": "high",
        "calculated_at": "2025-02-25T10:00:00Z"
    },
    "scope": "document",
    "document_name": "novel_draft",
    "expires_in_hours": 168  # Cache for 1 week
})
```

**Use Case 3: Global Agent State**
```python
# Track agent's current task globally
client.call_tool("store_memory", {
    "key": "agent_current_task",
    "value": {
        "task_id": "task_123",
        "type": "document_review",
        "started_at": "2025-02-25T09:00:00Z",
        "context": {...}
    },
    "scope": "global",
    "tags": ["task", "active"],
    "agent_id": "simple_agent_v1"
})
```

---

## 5. Integration Points with Document MCP Tools

### 5.1 Tool Hook Points

Context tools integrate with existing document tools at key moments:

**After document creation** (optional):
```python
# In create_document tool, optionally:
store_memory({
    "key": f"document_{document_name}_created",
    "value": {"created_at": now, "initial_chapters": 0},
    "scope": "document",
    "document_name": document_name,
    "source": "automatic"
})
```

**Before complex operations** (recommended):
```python
# In react_agent before multi-step operation:
previous_state = recall_memory(key="operation_state", scope="global")
if previous_state:
    # Resume from previous state
else:
    # Start fresh
```

**After document modifications** (optional):
```python
# User can manually store results after editing:
client.call_tool("store_memory", {
    "key": "last_modification_state",
    "value": {
        "operation": "replaced_chapter_content",
        "chapter": "02-main",
        "word_count_delta": 500
    },
    "scope": "document",
    "document_name": doc_name
})
```

### 5.2 No Breaking Changes

Context tools are **completely additive**:
- Existing document operations unaffected
- No modifications to existing tools
- Optional agent integration (not required)
- Backward compatible

---

## 6. Example Usage Patterns

### Pattern 1: Session State Management

```python
# Beginning of session
session_id = str(uuid.uuid4())
memories = client.call_tool("recall_memory", {"scope": "global"})

# Perform operations using contextual info
for memory in memories["memories"]:
    if memory["key"] == "workflow_status":
        continue_workflow(memory["value"])

# End of session
client.call_tool("store_memory", {
    "key": "session_completed",
    "value": {"session_id": session_id, "operations": op_count},
    "scope": "session",
    "expires_in_hours": 24
})
```

### Pattern 2: Memory Export for Sharing

```python
# Export all context
snapshot = client.call_tool("export_context", {
    "include_session_history": True,
    "scopes": ["global", "document"]
})

# File saved to: .context/snapshots/snapshot_2025-02-25_xxx.json
# Share file with collaborator

# Collaborator imports
client.call_tool("import_context", {
    "file_path": "shared_snapshot.json",
    "merge_mode": "merge"
})
```

### Pattern 3: Temporary Analysis Caching

```python
# Store expensive computation
client.call_tool("store_memory", {
    "key": "semantic_analysis",
    "value": {
        "embeddings": [...],  # Large data
        "computed_at": now,
        "document_hash": hash(doc)
    },
    "scope": "document",
    "document_name": "document_x",
    "expires_in_hours": 24,  # Auto-cleanup
    "tags": ["cache", "embeddings"]
})

# Later: reuse without recomputation
cached = client.call_tool("recall_memory", {
    "key": "semantic_analysis",
    "scope": "document",
    "document_name": "document_x"
})

if cached["memories"]:
    embeddings = cached["memories"][0]["value"]["embeddings"]
```

### Pattern 4: Multi-Agent Coordination

```python
# Agent A stores state for Agent B
client.call_tool("store_memory", {
    "key": "task_handoff",
    "value": {
        "from_agent": "simple_agent_v1",
        "to_agent": "react_agent_v2",
        "task_data": {...}
    },
    "scope": "global",
    "tags": ["handoff", "pending"]
})

# Agent B retrieves and processes
handoff = client.call_tool("recall_memory", {
    "tags": ["handoff"],
    "scope": "global"
})

if handoff["memories"]:
    process_task(handoff["memories"][0]["value"]["task_data"])
```

---

## 7. Implementation Complexity

All tools are **straightforward to implement** with low cognitive load:

| Tool | Complexity | Dependencies |
|------|-----------|--------------|
| `store_memory()` | Low | JSON file I/O, UUID generation |
| `recall_memory()` | Low | JSON parsing, filtering, expiration check |
| `export_context()` | Low | JSON serialization, file I/O |
| `import_context()` | Low | JSON deserialization, merge logic |
| `list_memories()` | Low | JSON parsing, filtering |
| `delete_memory()` | Low | JSON file I/O, list manipulation |
| `get_context_stats()` | Low | List aggregation, counting |

**Total LOC estimate**: 400-600 lines for all tools + storage layer.

---

## 8. Security & Reliability

### 8.1 Data Validation

All inputs validated through Pydantic models:
```python
# Automatic validation via Pydantic
MemoryEntry(**user_input)  # Raises ValidationError if invalid
```

### 8.2 File Safety

- JSON format (human-readable, debuggable)
- Atomic writes (write to temp file, then rename)
- UTF-8 encoding with fallback
- Directory creation with proper permissions

### 8.3 Expiration Cleanup

Auto-removes expired entries on recall:
```python
def recall_memory(...):
    # Load all
    memories = storage.load_memories()

    # Filter out expired
    active = [m for m in memories.memories
              if m.expires_at is None or m.expires_at > now]

    # Save cleaned version
    memories.memories = active
    storage.save_memories(memories)

    # Return active only
```

### 8.4 No Authentication Needed

- Local filesystem only
- Users manage directory permissions
- Documents storage security applies equally to context storage

---

## 9. Future Extensions (Not in Phase 4.3)

These are out of scope but design supports them:

1. **Semantic Search**: Index memories by content similarity
2. **Auto-Save**: Automatically persist after key operations
3. **Memory Compression**: Archive old memories
4. **Conflict Resolution**: Merge strategies for collaborative editing
5. **Memory Inheritance**: Child documents inherit parent memories
6. **Observability**: Metrics on memory usage patterns

---

## 10. Deliverables Checklist

- [x] Data model design (MemoryEntry, MemoryStore, SessionMetadata, ContextSnapshot)
- [x] Tool interface definitions (7 tools with full specs)
- [x] Storage backend strategy (.context/ structure + ContextStorage class)
- [x] API design (integration patterns, usage examples)
- [x] Integration points (non-breaking, additive)
- [x] Example usage patterns (4 concrete patterns)
- [x] Implementation guidance (low complexity, straightforward)
- [x] Security & reliability notes
- [x] Future extensions framework

---

## 11. Next Steps

1. **Phase 4.3a**: Implement data models (models/context.py)
2. **Phase 4.3b**: Implement storage layer (storage/context_storage.py)
3. **Phase 4.3c**: Implement tools (tools/context_tools.py)
4. **Phase 4.3d**: Unit tests (tests/unit/test_context_tools.py)
5. **Phase 4.3e**: Integration tests (tests/integration/test_context_integration.py)
6. **Phase 4.3f**: Agent integration examples
7. **Phase 4.3g**: Documentation and examples

---

## Appendix: File Structure Summary

```
document_mcp/
├── models/
│   └── context.py                 # NEW: MemoryEntry, MemoryStore, etc.
├── storage/
│   └── context_storage.py         # NEW: ContextStorage class
├── tools/
│   └── context_tools.py           # NEW: 7 context management tools
│
.context/                          # NEW: Runtime context storage
├── memories.json
├── sessions.json
├── index.json
└── snapshots/
    └── snapshot_*.json
```

