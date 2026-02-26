# Context Management System - Architecture Diagrams

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Document MCP System                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌────────────────────────────────────────────────────────┐    │
│  │              Agent (Simple or ReAct)                   │    │
│  │  - Executes user queries                              │    │
│  │  - Calls MCP tools                                    │    │
│  │  - Stores/recalls context across sessions             │    │
│  └────────────────────┬─────────────────────────────────┘    │
│                       │                                       │
│                       │ MCP Tool Calls                        │
│                       ▼                                       │
│  ┌────────────────────────────────────────────────────────┐    │
│  │         MCP Server (FastMCP)                           │    │
│  │  ┌──────────────────────────────────────────────────┐  │    │
│  │  │ Tool Categories:                                │  │    │
│  │  │                                                  │  │    │
│  │  │ • Document Tools (6)   ─┐                       │  │    │
│  │  │ • Chapter Tools (4)    │                        │  │    │
│  │  │ • Paragraph Tools (4)  │ Existing Tools         │  │    │
│  │  │ • Content Tools (6)    │ (28 total)             │  │    │
│  │  │ • Safety Tools (3)     │                        │  │    │
│  │  │ • Metadata Tools (3)   │                        │  │    │
│  │  │ • Overview Tools (1)   │                        │  │    │
│  │  │ • Discovery Tools (1) ─┤                        │  │    │
│  │  │                        │                        │  │    │
│  │  │ • Context Tools (7)   ─┐ NEW in Phase 4.3      │  │    │
│  │  │  - store_memory()     │                        │  │    │
│  │  │  - recall_memory()    │ Context Management     │  │    │
│  │  │  - export_context()   │ (7 new tools)          │  │    │
│  │  │  - import_context()   │                        │  │    │
│  │  │  - list_memories()    │                        │  │    │
│  │  │  - delete_memory()    │                        │  │    │
│  │  │  - get_context_stats()┘                        │  │    │
│  │  └──────────────────────────────────────────────────┘  │    │
│  │                      │                                  │    │
│  │                      │ Reads/Writes                     │    │
│  │                      ▼                                  │    │
│  │  ┌──────────────────────────────────────────────────┐  │    │
│  │  │         Storage Layer                           │  │    │
│  │  │  ┌────────────────────────────────────────────┐ │  │    │
│  │  │  │ ContextStorage (context_storage.py)       │ │  │    │
│  │  │  │  - load_memories()                        │ │  │    │
│  │  │  │  - save_memories()                        │ │  │    │
│  │  │  │  - load_sessions()                        │ │  │    │
│  │  │  │  - save_sessions()                        │ │  │    │
│  │  │  │  - save_snapshot()                        │ │  │    │
│  │  │  │  - load_snapshot()                        │ │  │    │
│  │  │  └────────────────────────────────────────────┘ │  │    │
│  │  └──────────────────────────────────────────────────┘  │    │
│  │                      │                                  │    │
│  │                      │ JSON I/O                         │    │
│  │                      ▼                                  │    │
│  │  ┌──────────────────────────────────────────────────┐  │    │
│  │  │         Local Filesystem                        │  │    │
│  │  │  .context/                                      │  │    │
│  │  │    ├── memories.json                            │  │    │
│  │  │    ├── sessions.json                            │  │    │
│  │  │    ├── index.json                               │  │    │
│  │  │    └── snapshots/                               │  │    │
│  │  │        └── snapshot_*.json                      │  │    │
│  │  │                                                  │  │    │
│  │  │  .documents_storage/  (existing)               │  │    │
│  │  │    └── [document dirs]                         │  │    │
│  │  └──────────────────────────────────────────────────┘  │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Model Hierarchy

```
ContextSnapshot (export/import unit)
    │
    ├── MemoryStore
    │   └── memories: list[MemoryEntry]
    │       ├── key: str (semantic identifier)
    │       ├── value: dict[str, Any]
    │       ├── scope: "global" | "document" | "session"
    │       ├── document_name: str | None
    │       ├── expires_at: datetime | None
    │       ├── tags: list[str]
    │       ├── created_at: datetime
    │       ├── updated_at: datetime
    │       ├── source: "manual" | "agent" | "automatic" | "import"
    │       └── agent_id: str | None
    │
    └── SessionHistory
        └── sessions: list[SessionMetadata]
            ├── session_id: str
            ├── created_at: datetime
            ├── closed_at: datetime | None
            ├── agent_id: str | None
            ├── operations_count: int
            ├── documents_accessed: list[str]
            ├── memory_created: int
            └── memory_recalled: int
```

---

## Tool Workflow - Store and Recall

```
Session 1:
┌──────────────────────┐
│ Agent/User starts    │
├──────────────────────┤
│ Performs operations  │
│                      │
│ Wants to save state  │
├──────────────────────┤
│ call store_memory({  │
│   key: "state",      │
│   value: {...},      │
│   scope: "global"    │
│ })                   │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────────────┐
│ ContextStorage.load_memories │
│ (read current .json)         │
└──────┬───────────────────────┘
       │
       ▼
┌──────────────────────────────┐
│ Create/Update MemoryEntry    │
└──────┬───────────────────────┘
       │
       ▼
┌──────────────────────────────┐
│ ContextStorage.save_memories │
│ (write to .context/JSON)     │
└──────┬───────────────────────┘
       │
       ▼
   ╔═════════╗
   ║ SUCCESS ║ Memory persisted
   ╚═════════╝


Session 2 (new session):
┌──────────────────────────────┐
│ Agent starts                 │
│                              │
│ Wants to recall context      │
├──────────────────────────────┤
│ call recall_memory({         │
│   key: "state",              │
│   scope: "global"            │
│ })                           │
└──────┬───────────────────────┘
       │
       ▼
┌──────────────────────────────┐
│ ContextStorage.load_memories │
│ (read from .context/JSON)    │
└──────┬───────────────────────┘
       │
       ▼
┌──────────────────────────────┐
│ Filter:                      │
│ - Remove expired entries     │
│ - Match key + scope          │
│ - Apply other filters        │
└──────┬───────────────────────┘
       │
       ▼
┌──────────────────────────────┐
│ Return matched memories      │
└──────┬───────────────────────┘
       │
       ▼
   ╔════════════════════════╗
   ║ Agent has context from ║
   ║ previous session       ║
   ╚════════════════════════╝
```

---

## Scope Boundaries

```
Global Scope
┌─────────────────────────────────────────────┐
│ Visible to: All agents, All sessions        │
│ Lifetime: Permanent (unless expires_at set) │
│ Across Sessions: YES                        │
│                                             │
│ Examples:                                   │
│ - Agent preferences                        │
│ - System configuration                     │
│ - Shared task state                        │
│ - Multi-agent coordination data             │
└─────────────────────────────────────────────┘

Document Scope
┌─────────────────────────────────────────────┐
│ Visible to: All agents, for one document    │
│ Lifetime: For lifetime of document          │
│ Across Sessions: YES                        │
│                                             │
│ Examples:                                   │
│ - Document structure analysis               │
│ - Workflow progress per document            │
│ - Document-specific embeddings cache        │
│ - Editorial metadata                        │
└─────────────────────────────────────────────┘

Session Scope
┌─────────────────────────────────────────────┐
│ Visible to: Current session only            │
│ Lifetime: Duration of session               │
│ Across Sessions: NO                         │
│                                             │
│ Examples:                                   │
│ - Temporary request state                   │
│ - Current operation progress                │
│ - Session-specific caches                   │
│ - Temporary computation results             │
└─────────────────────────────────────────────┘
```

---

## Export/Import Pipeline

```
Source System          Export                  File              Import           Target System
┌──────────────┐       ┌────────────────────┐  ┌──────────────┐  ┌─────────────┐  ┌──────────────┐
│ Memory Store │──────→│ ContextSnapshot    │─→│   JSON File  │─→│   Merge     │─→│ Memory Store │
│              │       │ (with filters)     │  │              │  │   (modes)   │  │              │
├──────────────┤       │  - scope filtering │  │ snapshot.json│  ├─────────────┤  ├──────────────┤
│ Session      │──────→│  - document filter │─→│              │─→│ • add       │─→│ Session      │
│ History      │       │  - include sessions│  │              │  │ • replace   │  │ History      │
└──────────────┘       └────────────────────┘  │              │  │ • merge     │  └──────────────┘
                                               │  (portable)  │  └─────────────┘
                                               │  (shareable) │
                                               │  (backupable)│
                                               └──────────────┘
                                                      │
                                                      │ Can be:
                                                      ├─ Emailed
                                                      ├─ Version controlled
                                                      ├─ Shared with colleagues
                                                      ├─ Stored in cloud
                                                      └─ Merged with other snapshots
```

---

## Tool Call Sequence Diagram

### Store → Recall → Export → Import

```
Agent                 MCP Server              ContextStorage       Filesystem
  │                        │                         │                  │
  ├─ store_memory() ───────→│                         │                  │
  │                         │─ load_memories() ──────→│                  │
  │                         │                         │─ read JSON ─────→│
  │                         │                         │←─ MemoryStore ──│
  │                         │─ add_or_update() ──────→│                  │
  │                         │← updated MemoryStore ──│                  │
  │                         │─ save_memories() ──────→│                  │
  │                         │                         │─ write JSON ────→│
  │←─ OperationStatus ──────┤                         │←─ OK ───────────│
  │                         │                         │                  │
  │                         │                         │                  │
  ├─ recall_memory() ──────→│                         │                  │
  │                         │─ load_memories() ──────→│                  │
  │                         │                         │─ read JSON ─────→│
  │                         │                         │←─ MemoryStore ──│
  │                         │─ filter/remove_expired()→│                  │
  │                         │← filtered memories ────│                  │
  │←─ memories list ────────┤                         │                  │
  │                         │                         │                  │
  │                         │                         │                  │
  ├─ export_context() ─────→│                         │                  │
  │                         │─ load_memories() ──────→│                  │
  │                         │                         │─ read JSON ─────→│
  │                         │─ create ContextSnapshot│                  │
  │                         │─ save_snapshot() ──────→│                  │
  │                         │                         │─ write snapshot→│
  │←─ snapshot file path ───┤                         │←─ OK ──────────│
  │                         │                         │                  │
  │                         │                         │                  │
  ├─ import_context() ─────→│                         │                  │
  │  (with file path)       │─ load_snapshot() ──────→│                  │
  │                         │                         │─ read JSON ─────→│
  │                         │                         │←─ ContextSnapshot│
  │                         │─ load_memories() ──────→│                  │
  │                         │                         │─ read JSON ─────→│
  │                         │─ merge() ──────────────→│                  │
  │                         │← merged store ────────│                  │
  │                         │─ save_memories() ──────→│                  │
  │                         │                         │─ write JSON ────→│
  │←─ import stats ─────────┤                         │←─ OK ──────────│
```

---

## File Structure with Example Paths

```
Project Root
└── /Users/clchinkc/Documents/GitHub/document-mcp/
    ├── .documents_storage/              (Existing: Documents)
    │   ├── my_book/
    │   │   ├── 01-intro.md
    │   │   ├── 02-main.md
    │   │   ├── .snapshots/
    │   │   └── .embeddings/
    │   └── user_guide/
    │
    ├── .context/                        (NEW: Context Storage)
    │   ├── memories.json                (All MemoryEntry objects)
    │   │   [
    │   │     {
    │   │       "memory_id": "a1b2c3...",
    │   │       "key": "book_structure",
    │   │       "value": {...},
    │   │       "scope": "document",
    │   │       "document_name": "my_book",
    │   │       "created_at": "2025-02-25T...",
    │   │       "updated_at": "2025-02-25T...",
    │   │       "expires_at": null,
    │   │       "tags": ["structure", "planning"]
    │   │     },
    │   │     {...more entries...}
    │   │   ]
    │   │
    │   ├── sessions.json                (SessionHistory)
    │   │   {
    │   │     "sessions": [
    │   │       {
    │   │         "session_id": "sess123",
    │   │         "created_at": "2025-02-25T...",
    │   │         "agent_id": "simple_agent",
    │   │         "operations_count": 5,
    │   │         ...
    │   │       }
    │   │     ],
    │   │     "current_session_id": "sess123"
    │   │   }
    │   │
    │   ├── index.json                   (Quick metadata)
    │   │   {
    │   │     "version": 1,
    │   │     "total_memories": 42,
    │   │     "last_updated": "2025-02-25T...",
    │   │     "snapshots_count": 3
    │   │   }
    │   │
    │   └── snapshots/                   (Exported snapshots)
    │       ├── snap_2025-02-25_abc123.json
    │       ├── snap_2025-02-25_def456.json
    │       └── snap_2025-02-25_ghi789.json
    │
    ├── document_mcp/                    (MCP Server)
    │   ├── models/
    │   │   ├── core.py                  (Existing)
    │   │   ├── documents.py             (Existing)
    │   │   ├── content.py               (Existing)
    │   │   └── context.py               (NEW - Phase 4.3)
    │   │
    │   ├── storage/
    │   │   ├── base.py                  (Existing)
    │   │   ├── local.py                 (Existing)
    │   │   └── context_storage.py       (NEW - Phase 4.3)
    │   │
    │   ├── tools/
    │   │   ├── document_tools.py        (Existing)
    │   │   ├── chapter_tools.py         (Existing)
    │   │   ├── content_tools.py         (Existing)
    │   │   ├── context_tools.py         (NEW - Phase 4.3)
    │   │   └── ...
    │   │
    │   └── doc_tool_server.py           (Register new tools)
    │
    └── tests/
        ├── unit/
        │   ├── test_context_models.py        (NEW)
        │   ├── test_context_storage.py       (NEW)
        │   └── test_context_tools.py         (NEW)
        │
        └── integration/
            └── test_context_integration.py   (NEW)
```

---

## State Transitions

```
Initial State
    │
    ├─→ store_memory("key1", {...})
    │       │
    │       ▼
    │    MemoryStore {
    │      memories: [MemoryEntry(key="key1")]
    │    }
    │
    ├─→ store_memory("key2", {...}, expires_in_hours=24)
    │       │
    │       ▼
    │    MemoryStore {
    │      memories: [
    │        MemoryEntry(key="key1"),
    │        MemoryEntry(key="key2", expires_at=tomorrow)
    │      ]
    │    }
    │
    ├─→ recall_memory() [auto-cleanup]
    │       │
    │       ▼
    │    Auto-removes expired entries:
    │    MemoryStore {
    │      memories: [
    │        MemoryEntry(key="key1")  ✓
    │                      (key2 expired)
    │      ]
    │    }
    │
    ├─→ export_context()
    │       │
    │       ▼
    │    snapshot.json saved
    │    [can be shared/backed up]
    │
    └─→ import_context(snapshot.json)
            │
            ▼
        Merge with existing:
        MemoryStore {
          memories: [
            MemoryEntry(key="key1", updated=original),
            MemoryEntry(key="key3", updated=from_snapshot)
          ]
        }
```

---

## Integration Points with Existing Tools

```
Existing Tools (28)          ←→  Context Tools (7 NEW)
                                    │
├─ Document Tools (6)          store_memory() ──→ Save state
├─ Chapter Tools (4)           recall_memory() ← Get state
├─ Paragraph Tools (4)         export_context() → Backup
├─ Content Tools (6)           import_context() ← Restore
├─ Safety Tools (3)            list_memories() → Overview
├─ Metadata Tools (3)          delete_memory() → Cleanup
├─ Overview Tools (1)          get_context_stats() → Stats
└─ Discovery Tools (1)


Optional Integration Patterns:

1. Automatic Snapshots (Future)
   Document Create → auto store_memory with doc metadata

2. Agent State Management
   Agent → store_memory after operation
   Agent → recall_memory at start

3. Workflow Coordination
   Tool A → store_memory with results
   Tool B → recall_memory for inputs

4. Analytics & Auditing
   Every operation → store to context
   export_context() → analyze patterns
```

---

## Error Handling Flow

```
Tool Call
    │
    ├─→ Input Validation (Pydantic)
    │       │
    │       ├─ INVALID → Return OperationStatus(success=False, message="...")
    │       └─ VALID ↓
    │
    ├─→ Load from Storage
    │       │
    │       ├─ FILE NOT FOUND → Return empty store (create fresh)
    │       ├─ PARSE ERROR → Return OperationStatus(success=False)
    │       └─ SUCCESS ↓
    │
    ├─→ Process Operation
    │       │
    │       ├─ LOGIC ERROR → Return OperationStatus(success=False)
    │       └─ SUCCESS ↓
    │
    ├─→ Save to Storage
    │       │
    │       ├─ WRITE ERROR → Return OperationStatus(success=False)
    │       └─ SUCCESS ↓
    │
    └─→ Return OperationStatus(success=True, details={...})


All errors caught at top level:
    try:
        # Full operation
    except Exception:
        return OperationStatus(success=False, message=str(e))
```

---

## Typical Session Flow

```
Session 1 (Day 1)
┌─────────────────────────────────────┐
│ Agent: simple_agent_v1              │
│ Time: 09:00 AM                      │
├─────────────────────────────────────┤
│ 1. User: "Edit chapter 2"           │
│    → recall_memory(key="state")     │
│    ← (empty, first time)            │
│                                     │
│ 2. Perform edits...                 │
│                                     │
│ 3. store_memory({                   │
│      key: "workflow_state",         │
│      value: {                       │
│        status: "paused",            │
│        next_chapter: "03",          │
│        progress: "50%"              │
│      },                             │
│      scope: "document",             │
│      document_name: "my_book"       │
│    })                               │
│                                     │
│ 4. Session ends ✓                   │
└─────────────────────────────────────┘

Days Later...

Session 2 (Day 5)
┌─────────────────────────────────────┐
│ Agent: simple_agent_v1              │
│ Time: 02:30 PM                      │
├─────────────────────────────────────┤
│ 1. User: "Resume editing my_book"   │
│    → recall_memory(key="workflow...")│
│    ← {status: "paused", next: "03"}  │
│                                     │
│ 2. Agent understands context!       │
│    Resume from chapter 03...        │
│                                     │
│ 3. Perform more edits...            │
│                                     │
│ 4. update_memory with new progress  │
│                                     │
│ 5. Session ends ✓                   │
└─────────────────────────────────────┘

Result: Seamless multi-session workflow
```

---

## Design Summary

The Context Management System provides:

**Persistence**: Store data once, use across sessions
**Flexibility**: Three scopes, arbitrary JSON values, tagging
**Safety**: Validation, atomic writes, auto-cleanup
**Simplicity**: 7 straightforward tools, JSON storage, zero external deps
**Integration**: Non-breaking, completely additive to existing system

**Total**: ~7 tools, ~3-4 models, ~600 LOC implementation

