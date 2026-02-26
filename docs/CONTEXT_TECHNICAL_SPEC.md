# Context Management System - Technical Specification

## Document Purpose

This is the technical specification for implementing the OneContext-inspired context management system for Document MCP. It provides detailed implementation guidance for developers implementing Phase 4.3.

---

## 1. Module Organization

### 1.1 File Structure

```
document_mcp/
├── models/
│   ├── __init__.py                    # Export all models
│   ├── context.py                     # NEW: Context models
│   ├── core.py                        # Existing: OperationStatus, etc.
│   ├── documents.py
│   ├── content.py
│   ├── analysis.py
│   └── metadata.py
├── storage/
│   ├── __init__.py
│   ├── base.py                        # Abstract base
│   ├── local.py                       # Existing: LocalStorageBackend
│   ├── gcs.py                         # Existing: GCS backend
│   ├── context_storage.py             # NEW: Context storage
│   └── factory.py
├── tools/
│   ├── __init__.py
│   ├── context_tools.py               # NEW: 7 context tools
│   ├── document_tools.py
│   ├── chapter_tools.py
│   ├── paragraph_tools.py
│   ├── content_tools.py
│   ├── safety_tools.py
│   ├── metadata_tools.py
│   ├── overview_tools.py
│   └── discovery_tools.py
└── doc_tool_server.py                 # Register new tools

tests/
├── unit/
│   ├── test_context_models.py         # NEW
│   ├── test_context_storage.py        # NEW
│   ├── test_context_tools.py          # NEW
│   └── ...existing tests...
└── integration/
    ├── test_context_integration.py    # NEW
    └── ...existing tests...

.context/                              # Runtime - created by system
├── memories.json
├── sessions.json
├── index.json
└── snapshots/
    └── *.json
```

---

## 2. Data Models (document_mcp/models/context.py)

### 2.1 Imports and Setup

```python
from __future__ import annotations

import datetime
import uuid
from typing import Any

from pydantic import BaseModel, Field, field_validator


__all__ = [
    "MemoryEntry",
    "MemoryStore",
    "SessionMetadata",
    "SessionHistory",
    "ContextSnapshot",
]
```

### 2.2 MemoryEntry Model

```python
class MemoryEntry(BaseModel):
    """Single unit of stored context/memory."""

    # Identity
    memory_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier"
    )

    # Content
    key: str = Field(..., min_length=1, max_length=255)
    value: dict[str, Any]

    # Scope & Targeting
    scope: str = Field(
        default="global",
        pattern="^(global|document|session)$"
    )
    document_name: str | None = None

    # Lifecycle
    created_at: datetime.datetime = Field(
        default_factory=datetime.datetime.now
    )
    updated_at: datetime.datetime = Field(
        default_factory=datetime.datetime.now
    )
    expires_at: datetime.datetime | None = None

    # Metadata
    source: str = Field(
        default="manual",
        pattern="^(manual|agent|automatic|import)$"
    )
    agent_id: str | None = None
    tags: list[str] = Field(default_factory=list, max_length=10)

    @field_validator("document_name")
    @classmethod
    def validate_document_required_for_scope(cls, v, info):
        """Ensure document_name set when scope='document'."""
        if info.data.get("scope") == "document" and not v:
            raise ValueError("document_name required when scope='document'")
        return v

    def is_expired(self) -> bool:
        """Check if memory has expired."""
        if self.expires_at is None:
            return False
        return datetime.datetime.now() > self.expires_at

    def get_value_preview(self, max_chars: int = 100) -> str:
        """Get string preview of value."""
        import json
        try:
            value_str = json.dumps(self.value)
            if len(value_str) > max_chars:
                return value_str[:max_chars] + "..."
            return value_str
        except Exception:
            return str(self.value)[:max_chars]


class MemoryStore(BaseModel):
    """Container for all memories."""

    memories: list[MemoryEntry] = Field(default_factory=list)
    version: int = 1
    last_updated: datetime.datetime = Field(
        default_factory=datetime.datetime.now
    )
    total_memories: int = 0

    def add_or_update(
        self,
        key: str,
        value: dict[str, Any],
        scope: str = "global",
        document_name: str | None = None,
    ) -> MemoryEntry:
        """Add new or update existing memory by key+scope+document."""
        # Find existing
        for entry in self.memories:
            if (entry.key == key and
                entry.scope == scope and
                entry.document_name == document_name):
                # Update
                entry.value = value
                entry.updated_at = datetime.datetime.now()
                return entry

        # Create new
        entry = MemoryEntry(
            key=key,
            value=value,
            scope=scope,
            document_name=document_name
        )
        self.memories.append(entry)
        return entry

    def find_by_key(self, key: str) -> MemoryEntry | None:
        """Find first entry with given key."""
        for entry in self.memories:
            if entry.key == key and not entry.is_expired():
                return entry
        return None

    def find_by_scope(self, scope: str) -> list[MemoryEntry]:
        """Find all non-expired entries with given scope."""
        return [e for e in self.memories
                if e.scope == scope and not e.is_expired()]

    def find_by_document(self, document_name: str) -> list[MemoryEntry]:
        """Find all non-expired entries for document."""
        return [e for e in self.memories
                if e.scope == "document" and
                e.document_name == document_name and
                not e.is_expired()]

    def find_by_tags(self, tags: list[str]) -> list[MemoryEntry]:
        """Find all non-expired entries matching any tag (OR logic)."""
        return [e for e in self.memories
                if any(tag in e.tags for tag in tags)
                and not e.is_expired()]

    def remove_expired(self) -> int:
        """Remove expired entries, return count removed."""
        original_len = len(self.memories)
        self.memories = [e for e in self.memories if not e.is_expired()]
        return original_len - len(self.memories)

    def delete_by_id(self, memory_id: str) -> bool:
        """Delete entry by ID, return success."""
        original_len = len(self.memories)
        self.memories = [e for e in self.memories
                        if e.memory_id != memory_id]
        return len(self.memories) < original_len

    def delete_by_key(self, key: str, scope: str) -> int:
        """Delete all entries with key in scope, return count."""
        original_len = len(self.memories)
        self.memories = [e for e in self.memories
                        if not (e.key == key and e.scope == scope)]
        return original_len - len(self.memories)
```

### 2.3 SessionMetadata and SessionHistory

```python
class SessionMetadata(BaseModel):
    """Information about a single session."""

    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime.datetime = Field(
        default_factory=datetime.datetime.now
    )
    closed_at: datetime.datetime | None = None
    agent_id: str | None = None
    operations_count: int = 0
    documents_accessed: list[str] = Field(default_factory=list)
    memory_created: int = 0
    memory_recalled: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)

    def duration_seconds(self) -> int | None:
        """Duration in seconds, or None if still open."""
        if self.closed_at is None:
            return None
        return int((self.closed_at - self.created_at).total_seconds())


class SessionHistory(BaseModel):
    """All recorded sessions."""

    sessions: list[SessionMetadata] = Field(default_factory=list)
    current_session_id: str | None = None
    total_sessions: int = 0

    def new_session(self, agent_id: str | None = None) -> SessionMetadata:
        """Create and track new session."""
        session = SessionMetadata(agent_id=agent_id)
        self.sessions.append(session)
        self.current_session_id = session.session_id
        return session

    def close_session(self, session_id: str) -> bool:
        """Close session, return success."""
        for session in self.sessions:
            if session.session_id == session_id:
                session.closed_at = datetime.datetime.now()
                if self.current_session_id == session_id:
                    self.current_session_id = None
                return True
        return False
```

### 2.4 ContextSnapshot

```python
class ContextSnapshot(BaseModel):
    """Complete context snapshot for export/import."""

    snapshot_id: str = Field(
        default_factory=lambda: f"snap_{datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
    )
    created_at: datetime.datetime = Field(
        default_factory=datetime.datetime.now
    )
    memory_store: MemoryStore
    session_history: SessionHistory = Field(
        default_factory=SessionHistory
    )
    metadata: dict[str, Any] = Field(default_factory=dict)
    version: int = 1
```

---

## 3. Storage Layer (document_mcp/storage/context_storage.py)

### 3.1 ContextStorage Class

```python
import json
from pathlib import Path
from typing import Any

from document_mcp.models import MemoryStore, SessionHistory, ContextSnapshot
from document_mcp.utils.file_operations import DOCS_ROOT_PATH


class ContextStorage:
    """Local filesystem context storage.

    Manages persistence of memory entries and session history to JSON files.
    """

    def __init__(self, root_dir: str | None = None):
        """Initialize storage.

        Args:
            root_dir: Root directory for context storage.
                     Defaults to parent of DOCS_ROOT_PATH/.context
        """
        if root_dir:
            self.root = Path(root_dir).resolve()
        else:
            parent = Path(DOCS_ROOT_PATH).parent
            self.root = parent / ".context"

        # Create directories
        self.root.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir = self.root / "snapshots"
        self.snapshots_dir.mkdir(exist_ok=True)

        # File paths
        self._memories_file = self.root / "memories.json"
        self._sessions_file = self.root / "sessions.json"
        self._index_file = self.root / "index.json"

    @property
    def context_root(self) -> str:
        """Return root context directory path."""
        return str(self.root)

    # Load operations
    def load_memories(self) -> MemoryStore:
        """Load memory store from disk."""
        if not self._memories_file.exists():
            return MemoryStore()

        try:
            with open(self._memories_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return MemoryStore(**data)
        except Exception as e:
            print(f"Error loading memories: {e}")
            return MemoryStore()

    def load_sessions(self) -> SessionHistory:
        """Load session history from disk."""
        if not self._sessions_file.exists():
            return SessionHistory()

        try:
            with open(self._sessions_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return SessionHistory(**data)
        except Exception as e:
            print(f"Error loading sessions: {e}")
            return SessionHistory()

    # Save operations
    def save_memories(self, store: MemoryStore) -> None:
        """Save memory store to disk."""
        store.last_updated = datetime.datetime.now()
        store.total_memories = len(store.memories)

        # Write to temp file first (atomicity)
        temp_file = self._memories_file.with_suffix('.tmp')
        try:
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(
                    store.model_dump(mode='json'),
                    f,
                    indent=2,
                    default=str
                )
            # Atomic rename
            temp_file.replace(self._memories_file)
        except Exception as e:
            temp_file.unlink(missing_ok=True)
            raise e

        self._update_index()

    def save_sessions(self, history: SessionHistory) -> None:
        """Save session history to disk."""
        history.total_sessions = len(history.sessions)

        temp_file = self._sessions_file.with_suffix('.tmp')
        try:
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(
                    history.model_dump(mode='json'),
                    f,
                    indent=2,
                    default=str
                )
            temp_file.replace(self._sessions_file)
        except Exception as e:
            temp_file.unlink(missing_ok=True)
            raise e

    # Snapshot operations
    def save_snapshot(self, snapshot: ContextSnapshot) -> str:
        """Save snapshot to file, return file path."""
        filename = f"{snapshot.snapshot_id}.json"
        filepath = self.snapshots_dir / filename

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(
                    snapshot.model_dump(mode='json'),
                    f,
                    indent=2,
                    default=str
                )
            return str(filepath)
        except Exception as e:
            raise IOError(f"Failed to save snapshot: {e}")

    def load_snapshot(self, filepath: str) -> ContextSnapshot:
        """Load snapshot from file."""
        path = Path(filepath)

        if not path.exists():
            raise FileNotFoundError(f"Snapshot not found: {filepath}")

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return ContextSnapshot(**data)
        except Exception as e:
            raise IOError(f"Failed to load snapshot: {e}")

    def list_snapshots(self) -> list[str]:
        """List all snapshot files."""
        return sorted([
            str(p) for p in self.snapshots_dir.glob("snapshot_*.json")
        ], reverse=True)  # Newest first

    # Index operations
    def _update_index(self) -> None:
        """Update quick metadata index."""
        memories = self.load_memories()
        snapshots = list(self.snapshots_dir.glob("snapshot_*.json"))

        index = {
            "version": 1,
            "total_memories": len(memories.memories),
            "last_updated": datetime.datetime.now().isoformat(),
            "snapshots_count": len(snapshots),
            "memories_by_scope": {
                scope: len(memories.find_by_scope(scope))
                for scope in ["global", "document", "session"]
            }
        }

        with open(self._index_file, 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2)

    # Utility
    def get_storage_stats(self) -> dict[str, Any]:
        """Get storage statistics."""
        memories = self.load_memories()
        memories.remove_expired()

        total_size = sum([
            p.stat().st_size
            for p in [self._memories_file, self._sessions_file]
            if p.exists()
        ])

        return {
            "total_memories": len(memories.memories),
            "storage_bytes": total_size,
            "snapshots_count": len(list(self.snapshots_dir.glob("*.json")))
        }


# Global singleton
_context_storage: ContextStorage | None = None


def get_context_storage() -> ContextStorage:
    """Get or create global context storage instance."""
    global _context_storage
    if _context_storage is None:
        _context_storage = ContextStorage()
    return _context_storage
```

---

## 4. Tools Implementation (document_mcp/tools/context_tools.py)

### 4.1 Tool Registration Function

```python
"""Context management tools for persistent cross-session memory."""

import uuid
import datetime
from typing import Any

from mcp.server import FastMCP

from ..logger_config import log_mcp_call
from ..models import MemoryEntry, MemoryStore, ContextSnapshot, OperationStatus
from ..storage.context_storage import get_context_storage


def register_context_tools(mcp_server: FastMCP) -> None:
    """Register all context management tools with MCP server."""

    storage = get_context_storage()

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
        """Store or update memory for cross-session persistence."""

        # Validate inputs
        if not key or len(key) > 255:
            return OperationStatus(
                success=False,
                message="Key must be 1-255 characters"
            )

        if scope not in ["global", "document", "session"]:
            return OperationStatus(
                success=False,
                message="Scope must be 'global', 'document', or 'session'"
            )

        if scope == "document" and not document_name:
            return OperationStatus(
                success=False,
                message="document_name required when scope='document'"
            )

        try:
            # Load current store
            store = storage.load_memories()

            # Calculate expiration
            expires_at = None
            if expires_in_hours:
                expires_at = datetime.datetime.now() + \
                    datetime.timedelta(hours=expires_in_hours)

            # Add or update
            is_new = not any(
                e.key == key and
                e.scope == scope and
                e.document_name == document_name
                for e in store.memories
            )

            entry = MemoryEntry(
                key=key,
                value=value,
                scope=scope,
                document_name=document_name,
                expires_at=expires_at,
                tags=tags or [],
            )

            # Add to store
            existing_idx = None
            for i, e in enumerate(store.memories):
                if (e.key == key and
                    e.scope == scope and
                    e.document_name == document_name):
                    existing_idx = i
                    break

            if existing_idx is not None:
                store.memories[existing_idx] = entry
            else:
                store.memories.append(entry)

            # Save
            storage.save_memories(store)

            return OperationStatus(
                success=True,
                message=f"Memory '{key}' stored successfully",
                details={
                    "memory_id": entry.memory_id,
                    "key": key,
                    "scope": scope,
                    "created_new": is_new,
                    "expires_at": expires_at.isoformat() if expires_at else None
                }
            )

        except Exception as e:
            return OperationStatus(
                success=False,
                message=f"Failed to store memory: {str(e)}"
            )

    # Additional tools follow similar pattern...
    # (recall_memory, export_context, import_context, etc.)
```

### 4.2 Tool Registration in Main Server

In `document_mcp/doc_tool_server.py`, add:

```python
from .tools import register_context_tools

# In setup function/initialization:
register_context_tools(mcp_server)
```

---

## 5. Testing Strategy

### 5.1 Unit Tests (test_context_models.py)

```python
import pytest
from document_mcp.models import MemoryEntry, MemoryStore


class TestMemoryEntry:
    def test_memory_entry_creation(self):
        entry = MemoryEntry(
            key="test",
            value={"data": "value"}
        )
        assert entry.key == "test"
        assert not entry.is_expired()

    def test_expiration_validation(self):
        """Memory with past expiration is expired."""
        past = datetime.datetime.now() - datetime.timedelta(hours=1)
        entry = MemoryEntry(
            key="test",
            value={},
            expires_at=past
        )
        assert entry.is_expired()

    def test_scope_document_requires_name(self):
        """Scope='document' requires document_name."""
        with pytest.raises(ValueError):
            MemoryEntry(
                key="test",
                value={},
                scope="document"
            )


class TestMemoryStore:
    def test_add_or_update(self):
        store = MemoryStore()
        entry = store.add_or_update("key1", {"val": 1})
        assert len(store.memories) == 1

        # Update
        entry2 = store.add_or_update("key1", {"val": 2})
        assert len(store.memories) == 1
        assert entry2.value == {"val": 2}

    def test_find_by_tags(self):
        store = MemoryStore()
        store.add_or_update("k1", {}, tags=["tag1", "tag2"])
        store.add_or_update("k2", {}, tags=["tag2", "tag3"])

        # Find by tag1 (OR logic)
        result = store.find_by_tags(["tag1"])
        assert len(result) == 1

        # Find by tag2 or tag3
        result = store.find_by_tags(["tag2", "tag3"])
        assert len(result) == 2

    def test_remove_expired(self):
        store = MemoryStore()
        store.add_or_update("k1", {})
        past = datetime.datetime.now() - datetime.timedelta(hours=1)
        store.add_or_update("k2", {}, expires_at=past)

        removed = store.remove_expired()
        assert removed == 1
        assert len(store.memories) == 1
```

### 5.2 Integration Tests (test_context_integration.py)

```python
@pytest.mark.asyncio
async def test_store_and_recall_via_mcp(mcp_client):
    """Test store_memory → recall_memory round trip via MCP."""

    # Store
    response = await mcp_client.call_tool("store_memory", {
        "key": "test_memory",
        "value": {"data": "test"},
        "scope": "global"
    })
    assert response["success"]

    # Recall
    response = await mcp_client.call_tool("recall_memory", {
        "key": "test_memory"
    })
    assert response["success"]
    assert len(response["memories"]) == 1
    assert response["memories"][0]["value"]["data"] == "test"
```

---

## 6. Error Handling

All tools should follow this pattern:

```python
try:
    # Validate inputs
    if invalid:
        return OperationStatus(
            success=False,
            message="Clear error message"
        )

    # Perform operation
    result = operation()

    return OperationStatus(
        success=True,
        message="Operation successful",
        details={...}
    )

except FileNotFoundError:
    return OperationStatus(
        success=False,
        message=f"File not found: {path}"
    )
except json.JSONDecodeError:
    return OperationStatus(
        success=False,
        message="Invalid JSON format in stored data"
    )
except Exception as e:
    return OperationStatus(
        success=False,
        message=f"Unexpected error: {str(e)}"
    )
```

---

## 7. Implementation Checklist

### Data Models
- [ ] MemoryEntry with validation
- [ ] MemoryStore with helper methods
- [ ] SessionMetadata and SessionHistory
- [ ] ContextSnapshot
- [ ] All models in __all__ exports
- [ ] Pydantic validators for cross-field validation

### Storage Layer
- [ ] ContextStorage class
- [ ] Load/save for memories
- [ ] Load/save for sessions
- [ ] Snapshot save/load
- [ ] Index updates
- [ ] Atomic file operations
- [ ] UTF-8 encoding
- [ ] Error handling

### Tools
- [ ] store_memory()
- [ ] recall_memory()
- [ ] export_context()
- [ ] import_context()
- [ ] list_memories()
- [ ] delete_memory()
- [ ] get_context_stats()
- [ ] MCP registration
- [ ] @log_mcp_call decorators

### Testing
- [ ] Unit tests for models (85%+ coverage)
- [ ] Unit tests for storage (95%+ coverage)
- [ ] Unit tests for tools (90%+ coverage)
- [ ] Integration tests with MCP
- [ ] Type checking passes (mypy)
- [ ] Linting passes (ruff)

### Integration
- [ ] Register tools in doc_tool_server.py
- [ ] Update __init__.py exports
- [ ] Add to tools/__init__.py
- [ ] Test with both Simple and ReAct agents

---

## 8. Performance Considerations

- **Memory Loading**: Entire store loads into memory (~< 10MB typical)
- **Expiration Cleanup**: Runs on every recall (O(n) but acceptable)
- **Search**: Linear scan by key/scope/tags (acceptable for typical store size)
- **File I/O**: Atomic writes to temp file
- **Serialization**: JSON (human-readable, slight overhead vs binary)

### Optimization if needed
- Implement indexing by key for O(1) lookup
- Background cleanup task (out of scope for Phase 4.3)
- Lazy loading of snapshots

---

## 9. Security Notes

- All inputs validated via Pydantic
- File paths resolved with `.resolve()` to prevent traversal
- JSON serialization (safe from code injection)
- No authentication (relies on filesystem permissions)
- No encryption (use environment secrets for sensitive data)

---

## 10. Dependencies

**Required** (already in project):
- pydantic
- typing (stdlib)
- pathlib (stdlib)
- json (stdlib)
- uuid (stdlib)
- datetime (stdlib)

**Not required**:
- No external database
- No external caching library
- No encryption library

---

## 11. Version Compatibility

- Python 3.10+ (uses `|` union syntax)
- Pydantic v2.x
- Compatible with existing Document MCP codebase

---

## Quick Reference: Adding New Tool

To add a new context tool:

1. Create function with `@mcp_server.tool()` and `@log_mcp_call` decorators
2. Validate inputs
3. Load context: `storage = get_context_storage(); store = storage.load_memories()`
4. Perform operation
5. Save context: `storage.save_memories(store)`
6. Return OperationStatus with details
7. Add unit tests
8. Add integration tests

Example:
```python
@mcp_server.tool()
@log_mcp_call
def new_tool(param: str) -> OperationStatus:
    """Docstring with full description."""
    try:
        # Validate
        if not param:
            return OperationStatus(success=False, message="Invalid input")

        # Load
        storage = get_context_storage()
        store = storage.load_memories()

        # Operate
        result = do_something(store, param)

        # Save
        storage.save_memories(store)

        return OperationStatus(
            success=True,
            message="Success",
            details=result
        )
    except Exception as e:
        return OperationStatus(success=False, message=str(e))
```

---

## References

- Main design: `docs/CONTEXT_MANAGEMENT_SYSTEM.md`
- Quick reference: `docs/CONTEXT_QUICK_REFERENCE.md`
- Implementation roadmap: `docs/CONTEXT_IMPLEMENTATION_ROADMAP.md`

