# Context Management System - Quick Reference

## Overview

The Context Management System provides persistent, cross-session memory for Document MCP agents. Store contextual information once, use it across multiple sessions.

**Directory**: `.context/` (parallel to `.documents_storage/`)

---

## 7 Core Tools

### 1. store_memory()

**Store context for later retrieval**

```python
{
    "key": "memory_name",           # Required: semantic key
    "value": {...},                 # Required: any JSON-serializable data
    "scope": "global",              # Optional: 'global', 'document', 'session' (default: 'global')
    "document_name": "my_book",     # Required if scope='document'
    "expires_in_hours": 24,         # Optional: auto-delete after N hours
    "tags": ["tag1", "tag2"]        # Optional: for filtering
}
```

**Returns**: `{ success, message, details: { memory_id, created_new, expires_at } }`

---

### 2. recall_memory()

**Retrieve stored memories with filtering**

```python
{
    "key": "memory_name",           # Optional: specific memory
    "scope": "global",              # Optional: filter by scope
    "document_name": "my_book",     # Optional: if scope='document'
    "tags": ["tag1"],               # Optional: filter by tags (OR logic)
    "limit": 10                     # Optional: max results (default: 10)
}
```

**Returns**: `{ success, memories: [...], total_found, expired_removed }`

---

### 3. export_context()

**Export all context to JSON file for backup/sharing**

```python
{
    "include_session_history": true,        # Optional: include sessions
    "scopes": ["global", "document"],       # Optional: filter scopes
    "document_names": ["my_book"]           # Optional: filter documents
}
```

**Returns**: `{ success, details: { snapshot_id, file_path, file_size_bytes } }`

**File saved to**: `.context/snapshots/snapshot_2025-02-25_abc123.json`

---

### 4. import_context()

**Import context from JSON file**

```python
{
    "file_path": ".context/snapshots/snapshot_2025-02-25_abc123.json",
    "merge_mode": "merge",          # 'add' (skip existing), 'replace', 'merge' (newer wins)
    "remove_expired": true          # Optional: auto-delete expired entries
}
```

**Returns**: `{ success, details: { imported_memories, merge_mode, conflicts_skipped } }`

---

### 5. list_memories()

**List memories with summary info**

```python
{
    "scope": "global",              # Optional: filter by scope
    "document_name": "my_book",     # Optional: if scope='document'
    "include_expired": false        # Optional: show expired entries
}
```

**Returns**: `{ success, memories: [ { memory_id, key, tags, created_at, value_preview } ] }`

---

### 6. delete_memory()

**Delete specific memories**

```python
// By ID (most precise)
{ "memory_id": "uuid-abc123" }

// By key in scope (careful - deletes all matching!)
{ "key": "memory_name", "scope": "global" }
```

**Returns**: `{ success, details: { deleted_count, memory_ids } }`

---

### 7. get_context_stats()

**Get statistics about stored context**

No parameters required.

**Returns**:
```python
{
    "total_memories": 42,
    "by_scope": { "global": 20, "document": 15, "session": 7 },
    "by_source": { "manual": 25, "agent": 10, "automatic": 5, "import": 2 },
    "total_sessions": 8,
    "memory_storage_bytes": 245000,
    "oldest_memory": "2025-02-20T10:00:00Z",
    "newest_memory": "2025-02-25T15:30:00Z",
    "expired_count": 3,
    "unique_keys": 18
}
```

---

## Quick Start Examples

### Example 1: Store and Retrieve

```python
# Store analysis for later use
store_memory({
    "key": "book_analysis",
    "value": {
        "themes": ["resilience", "growth"],
        "characters": 12,
        "word_count": 50000
    },
    "scope": "document",
    "document_name": "my_novel",
    "tags": ["analysis", "structure"]
})

# Retrieve same analysis in another session
result = recall_memory({
    "key": "book_analysis",
    "scope": "document",
    "document_name": "my_novel"
})

analysis = result["memories"][0]["value"]  # Use the data
```

### Example 2: Workflow State Tracking

```python
# Session 1: Start workflow
store_memory({
    "key": "editing_workflow",
    "value": {
        "status": "in_progress",
        "completed": ["01-intro.md"],
        "next": "02-main.md",
        "progress": "20%"
    },
    "scope": "document",
    "document_name": "user_guide"
})

# Session 2: Resume workflow
state = recall_memory({
    "key": "editing_workflow",
    "scope": "document",
    "document_name": "user_guide"
})

if state["memories"]:
    workflow = state["memories"][0]["value"]
    print(f"Continue from: {workflow['next']}")
```

### Example 3: Share Context

```python
# Export all context
export_context({
    "include_session_history": true
})
# File: .context/snapshots/snapshot_2025-02-25_xyz.json

# Share file with colleague, they import it
import_context({
    "file_path": "snapshot_shared_by_colleague.json",
    "merge_mode": "merge"
})

# Now both have same context
```

### Example 4: Temporary Cache

```python
# Store expensive computation with expiration
store_memory({
    "key": "embeddings_cache",
    "value": {"vectors": [...]},  # Large data
    "expires_in_hours": 24,
    "tags": ["cache"]
})

# Next session - reuse if still valid
cache = recall_memory({
    "key": "embeddings_cache",
    "tags": ["cache"]
})

if cache["memories"]:
    vectors = cache["memories"][0]["value"]["vectors"]
else:
    vectors = recompute_embeddings()  # No longer cached
```

### Example 5: Filter by Tags

```python
# Store multiple tagged memories
store_memory({
    "key": "config_1",
    "value": {...},
    "tags": ["configuration", "production"]
})

store_memory({
    "key": "config_2",
    "value": {...},
    "tags": ["configuration", "staging"]
})

# Retrieve all configuration memories
configs = recall_memory({
    "tags": ["configuration"],
    "limit": 50
})

# Returns all memories with 'configuration' tag (2 in this case)
```

---

## Scope Explanation

| Scope | Visibility | Use Case |
|-------|-----------|----------|
| **global** | All agents, all sessions | Agent state, shared preferences, configuration |
| **document** | Specific document only | Document analysis, workflow state, metadata |
| **session** | Current session only | Temporary state, request-scoped data |

---

## Storage Structure

```
.context/
├── memories.json              # All memory entries
├── sessions.json              # Session history
├── index.json                 # Quick metadata index
└── snapshots/
    ├── snapshot_2025-02-25_abc123.json
    ├── snapshot_2025-02-25_def456.json
    └── ...
```

**File sizes**: Minimal JSON overhead, typical memory entries < 1KB each.

---

## Common Patterns

### Pattern: Session State

Save state at end of session, retrieve at start of next session.

```python
# Start
previous = recall_memory({"scope": "global", "tags": ["session-state"]})

# Work...

# End
store_memory({
    "key": f"session_{session_id}",
    "value": {...current_state...},
    "tags": ["session-state"],
    "expires_in_hours": 7 * 24  # Keep for 1 week
})
```

### Pattern: Caching

Store expensive computations with expiration.

```python
# Try cache first
cached = recall_memory({"key": "expensive_result", "tags": ["cache"]})

if cached["memories"]:
    result = cached["memories"][0]["value"]
else:
    # Compute and cache
    result = expensive_function()
    store_memory({
        "key": "expensive_result",
        "value": result,
        "expires_in_hours": 24,
        "tags": ["cache"]
    })
```

### Pattern: Multi-Agent Coordination

Agent A stores for Agent B.

```python
# Agent A
store_memory({
    "key": "task_for_agent_b",
    "value": {...task_data...},
    "scope": "global",
    "tags": ["handoff"]
})

# Agent B
task = recall_memory({
    "tags": ["handoff"],
    "scope": "global"
})
```

### Pattern: Analytics

Track operational metrics.

```python
store_memory({
    "key": "operation_metrics",
    "value": {
        "operation": "bulk_edit",
        "duration_ms": 1250,
        "items_processed": 50,
        "timestamp": "2025-02-25T10:30:00Z"
    },
    "tags": ["metrics", "analytics"]
})

# Later: analyze
all_metrics = recall_memory({
    "tags": ["metrics"],
    "limit": 100
})
```

---

## Tips & Best Practices

### DO

- ✅ Use semantic keys: `book_structure`, `workflow_progress`, `cache_embeddings`
- ✅ Tag memories for easier filtering: `["analysis", "cache", "draft"]`
- ✅ Use appropriate scope: `document` for per-doc state, `global` for shared
- ✅ Set expiration for temporary data: `expires_in_hours=24`
- ✅ Export regularly for backup: `export_context()`

### DON'T

- ❌ Store passwords or secrets (use environment variables)
- ❌ Store massive data structures (keep memories < 1MB typically)
- ❌ Assume memories persist forever (set expiration for temporary data)
- ❌ Use global scope for document-specific data (use `document` scope)
- ❌ Delete without confirming (deletion is permanent)

---

## Debugging

### Check what's stored

```python
get_context_stats()
# Shows: total memories, by scope, by source, storage size, etc.

list_memories({"scope": "global"})
# Shows: all global memories with previews
```

### Export and inspect

```python
export_context({})
# File: .context/snapshots/snapshot_2025-02-25_xxx.json

# Open file manually to inspect JSON
cat .context/snapshots/snapshot_2025-02-25_xxx.json | jq
```

### Delete problematic memory

```python
# By ID (precise)
delete_memory({"memory_id": "uuid-here"})

# By key (all in scope)
delete_memory({"key": "bad_memory", "scope": "global"})
```

---

## Integration with Agents

### Simple Agent

```python
from document_mcp.mcp_client import DocumentMCPClient

client = DocumentMCPClient()

# Check context before operation
context = client.call_tool("recall_memory", {
    "key": "book_state",
    "scope": "document",
    "document_name": "my_book"
})

if context["memories"]:
    state = context["memories"][0]["value"]
    # Use state to inform operation
```

### ReAct Agent

Context tools work like any other tools - agent can use them in reasoning loop.

```
Thought: I need to understand the document structure
Action: recall_memory with key="doc_structure"
Observation: Retrieved previous analysis
Thought: Now I can proceed with editing...
```

---

## Limits & Performance

| Aspect | Limit | Notes |
|--------|-------|-------|
| Memory size | ~10 MB | Per memory entry, but keep smaller for performance |
| Total context | No limit | Limited only by disk space |
| Query speed | < 10ms | For typical recalls (up to 10 results) |
| Export speed | < 1 sec | For typical context stores (< 1000 memories) |
| Expiration check | Automatic | Runs on every recall |

---

## File Examples

### Memory Entry

```json
{
    "memory_id": "a1b2c3d4-e5f6-4a8b-9c0d-1e2f3a4b5c6d",
    "key": "book_structure",
    "value": {
        "chapters": ["intro", "main", "conclusion"],
        "word_count": 50000
    },
    "scope": "document",
    "document_name": "my_book",
    "created_at": "2025-02-25T10:00:00",
    "updated_at": "2025-02-25T10:00:00",
    "expires_at": null,
    "source": "manual",
    "agent_id": null,
    "tags": ["structure", "planning"]
}
```

### Snapshot Metadata

```json
{
    "snapshot_id": "snap_2025-02-25_123abc",
    "created_at": "2025-02-25T15:30:00",
    "memory_store": {...},
    "session_history": {...},
    "metadata": {
        "reason": "manual_export",
        "user_comment": "Backup before major edit"
    },
    "version": 1
}
```

---

## Next Steps

1. **List current memories**: `get_context_stats()`
2. **Store your first memory**: `store_memory({...})`
3. **Retrieve it**: `recall_memory({key: "..."})`
4. **Export for backup**: `export_context()`

See full design in `docs/CONTEXT_MANAGEMENT_SYSTEM.md`

