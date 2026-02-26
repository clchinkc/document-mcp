# Phase 4.3: OneContext-Inspired Context Management System

## Overview

Phase 4.3 adds **persistent cross-session memory** to Document MCP, enabling agents to store and retrieve contextual information across multiple sessions.

**Status**: Design Complete - Ready for Implementation
**Timeline**: 2-3 development sessions (~4.5 hours)
**Complexity**: Low - Straightforward JSON-based storage
**Breaking Changes**: None - Completely additive

---

## Deliverables

### 1. Design Documents (COMPLETE)

#### Main Architecture Document
**File**: `docs/CONTEXT_MANAGEMENT_SYSTEM.md`

Complete design specification including:
- Data model design (MemoryEntry, MemoryStore, SessionMetadata, ContextSnapshot)
- 7 tool interface definitions with full specs
- Storage backend strategy (.context/ directory structure)
- API design and integration patterns
- Security & reliability considerations
- Future extensions framework

#### Technical Specification
**File**: `docs/CONTEXT_TECHNICAL_SPEC.md`

Developer-focused implementation guide:
- Module organization and file structure
- Detailed code patterns for each component
- Error handling strategies
- Testing requirements
- Performance considerations
- Quick reference for adding new tools

#### Implementation Roadmap
**File**: `docs/CONTEXT_IMPLEMENTATION_ROADMAP.md`

Step-by-step implementation plan:
- 10 milestones with specific tasks
- Time estimates per milestone (~4.5 hours total)
- Testing strategy for each milestone
- Task dependencies and critical path
- Success criteria checklist

#### Quick Reference Guide
**File**: `docs/CONTEXT_QUICK_REFERENCE.md`

User-facing documentation:
- All 7 tools with examples
- Common usage patterns (4 practical patterns)
- Scope explanation and storage structure
- Tips & best practices
- Debugging guide
- File examples with sample JSON

---

## 7 Core Tools

### 1. store_memory()
**Store or update** a memory entry for later retrieval.

```python
store_memory(
    key="book_structure",
    value={"chapters": [...], "word_count": 50000},
    scope="document",
    document_name="my_book",
    expires_in_hours=None,
    tags=["structure", "planning"]
)
```

### 2. recall_memory()
**Retrieve** stored memories with flexible filtering.

```python
recall_memory(
    key="book_structure",      # Optional: specific key
    scope="document",          # Optional: filter by scope
    document_name="my_book",   # Optional: document filter
    tags=["planning"],         # Optional: tag filter
    limit=10                   # Optional: max results
)
```

### 3. export_context()
**Export** all context to JSON file for backup/sharing.

```python
export_context(
    include_session_history=True,
    scopes=["global", "document"],
    document_names=["my_book"]
)
# Saves to: .context/snapshots/snapshot_2025-02-25_abc123.json
```

### 4. import_context()
**Import** context from JSON file with merge strategies.

```python
import_context(
    file_path=".context/snapshots/snapshot_2025-02-25_abc123.json",
    merge_mode="merge",        # 'add', 'replace', 'merge'
    remove_expired=True
)
```

### 5. list_memories()
**List** available memories with summary information.

```python
list_memories(
    scope="global",
    document_name=None,
    include_expired=False
)
```

### 6. delete_memory()
**Delete** specific memory entries by ID or key.

```python
delete_memory(
    memory_id="uuid-here"      # Precise deletion by ID
    # OR
    # key="memory_name", scope="global"  # Delete all with key in scope
)
```

### 7. get_context_stats()
**Get** statistics about stored context.

```python
get_context_stats()
# Returns: total, by scope, by source, storage size, etc.
```

---

## Data Models

### MemoryEntry
Represents a single stored memory.

```python
{
    "memory_id": str,              # UUID
    "key": str,                    # Semantic identifier
    "value": dict,                 # Arbitrary JSON data
    "scope": str,                  # "global", "document", "session"
    "document_name": str | None,   # Required if scope="document"
    "created_at": datetime,        # Auto-generated
    "updated_at": datetime,        # Auto-updated
    "expires_at": datetime | None, # Optional expiration
    "source": str,                 # "manual", "agent", "automatic", "import"
    "agent_id": str | None,        # Which agent created it
    "tags": list[str]              # For filtering
}
```

### MemoryStore
Contains all memory entries with helper methods.

```python
{
    "memories": list[MemoryEntry],
    "version": int,
    "last_updated": datetime,
    "total_memories": int
}
```

### SessionMetadata & SessionHistory
Track sessions for audit and analytics.

```python
SessionMetadata {
    "session_id": str,
    "created_at": datetime,
    "closed_at": datetime | None,
    "agent_id": str | None,
    "operations_count": int,
    "documents_accessed": list[str],
    "memory_created": int,
    "memory_recalled": int
}

SessionHistory {
    "sessions": list[SessionMetadata],
    "current_session_id": str | None,
    "total_sessions": int
}
```

### ContextSnapshot
Complete context snapshot for export/import.

```python
{
    "snapshot_id": str,
    "created_at": datetime,
    "memory_store": MemoryStore,
    "session_history": SessionHistory,
    "metadata": dict,
    "version": int
}
```

---

## Storage Structure

```
.context/                              # Context root (parallel to .documents_storage/)
├── memories.json                      # All memory entries
├── sessions.json                      # Session history
├── index.json                         # Quick metadata index
└── snapshots/                         # Exported snapshots
    ├── snapshot_2025-02-25_abc123.json
    ├── snapshot_2025-02-25_def456.json
    └── ...
```

**Features**:
- JSON format (human-readable, debuggable)
- Atomic writes (write to temp file, then rename)
- UTF-8 encoding
- Auto-creates directory structure

---

## Integration Points

### With Existing Document Tools
Context tools are **completely additive**:
- No modifications to existing tools
- No breaking changes
- Optional agent integration
- Can be used independently

### With Agents
Both Simple and ReAct agents can use context tools:

```python
# Recall previous context
context = client.call_tool("recall_memory", {
    "key": "workflow_state",
    "scope": "document",
    "document_name": "my_book"
})

# Use contextual info...

# Store results for next session
client.call_tool("store_memory", {
    "key": "workflow_state",
    "value": {...},
    "scope": "document",
    "document_name": "my_book"
})
```

---

## Usage Patterns

### Pattern 1: Workflow State Tracking
Save progress, resume in next session.

```python
# Session 1: Start editing workflow
store_memory({
    "key": "editing_workflow",
    "value": {
        "status": "in_progress",
        "completed": ["01-intro"],
        "next": "02-main"
    },
    "scope": "document",
    "document_name": "my_book"
})

# Session 2: Continue workflow
state = recall_memory({
    "key": "editing_workflow",
    "scope": "document",
    "document_name": "my_book"
})
```

### Pattern 2: Analysis Caching
Store expensive computations for reuse.

```python
# Cache semantic analysis
store_memory({
    "key": "semantic_analysis",
    "value": {...computed_embeddings...},
    "expires_in_hours": 24,
    "tags": ["cache", "embeddings"]
})

# Reuse cache (if still fresh)
cached = recall_memory({
    "key": "semantic_analysis",
    "tags": ["cache"]
})
```

### Pattern 3: Context Sharing
Export and share context between users.

```python
# Export all context
export_context({
    "include_session_history": true
})

# Share file with colleague
# Colleague imports it
import_context({
    "file_path": "colleague_snapshot.json",
    "merge_mode": "merge"
})
```

### Pattern 4: Multi-Agent Coordination
Agent A stores for Agent B.

```python
# Agent A: Store task for Agent B
store_memory({
    "key": "task_for_b",
    "value": {...},
    "scope": "global",
    "tags": ["handoff"]
})

# Agent B: Retrieve task
task = recall_memory({
    "tags": ["handoff"],
    "scope": "global"
})
```

---

## Implementation Timeline

### Phase 4.3a: Data Models (30 mins)
- Create `document_mcp/models/context.py`
- Define all Pydantic models
- Type validation and validators

### Phase 4.3b: Storage Layer (30 mins)
- Create `document_mcp/storage/context_storage.py`
- Implement ContextStorage class
- File I/O with atomicity

### Phase 4.3c: Core Tools (1.5 hours)
- Create `document_mcp/tools/context_tools.py`
- Implement all 7 tools
- MCP registration

### Phase 4.3d: Unit Tests (45 mins)
- Models validation tests
- Storage I/O tests
- Tool functionality tests
- 85%+ coverage target

### Phase 4.3e: Integration Tests (30 mins)
- MCP server integration
- Tool chaining (store → recall)
- Round-trip testing

### Phase 4.3f: Agent Integration (45 mins)
- Example scripts
- Documentation
- Usage patterns

### Phase 4.3g: Final Polish (30 mins)
- Documentation completion
- Performance validation
- Release prep

**Total: ~4.5 hours**

---

## Key Characteristics

### Simplicity
- **7 straightforward tools** - no complex logic
- **JSON storage** - human-readable, debuggable
- **~400-600 LOC** total implementation
- **Zero external dependencies** - uses only stdlib + Pydantic

### Safety
- **Atomic writes** - temp file then rename
- **Validation** - all inputs via Pydantic
- **Expiration cleanup** - auto-removes stale entries
- **Non-destructive** - merge modes preserve data

### Flexibility
- **Three scopes**: global (all agents), document (specific doc), session (temporary)
- **Arbitrary JSON values** - store any structured data
- **Tagging system** - organize memories for filtering
- **Expiration support** - auto-cleanup for temporary data

### Extensibility
- Design supports future features:
  - Semantic search by content similarity
  - Auto-save on key operations
  - Memory compression for old entries
  - Collaborative conflict resolution

---

## Success Criteria

- [x] All 7 tools fully specified
- [x] Data models defined
- [x] Storage strategy clear
- [x] API design complete
- [x] Integration points identified
- [x] Usage patterns documented
- [x] Implementation roadmap ready
- [x] No breaking changes
- [x] Low complexity design

**Implementation Criteria** (to verify after coding):
- [ ] 85%+ test coverage
- [ ] All tests passing
- [ ] Type checking clean (mypy)
- [ ] Linting clean (ruff)
- [ ] Agent integration working
- [ ] Performance acceptable (< 10ms per operation)

---

## Files in This Delivery

| File | Purpose | Size |
|------|---------|------|
| `docs/CONTEXT_MANAGEMENT_SYSTEM.md` | Main architecture design | ~3500 lines |
| `docs/CONTEXT_TECHNICAL_SPEC.md` | Implementation details | ~1500 lines |
| `docs/CONTEXT_IMPLEMENTATION_ROADMAP.md` | Step-by-step implementation | ~400 lines |
| `docs/CONTEXT_QUICK_REFERENCE.md` | User/developer quick guide | ~700 lines |
| `docs/PHASE_4_3_SUMMARY.md` | This summary | ~400 lines |

**Total Design Documentation**: ~6500 lines

---

## Next Steps

### For Architecture Review
1. Review `docs/CONTEXT_MANAGEMENT_SYSTEM.md` for design correctness
2. Review `docs/CONTEXT_TECHNICAL_SPEC.md` for implementation approach
3. Verify integration points don't break existing functionality
4. Check scope boundaries (global/document/session)

### For Implementation
1. Start with Milestone 1 (Data Models) from roadmap
2. Follow implementation roadmap sequentially
3. Run tests after each milestone
4. Reference technical spec for code patterns
5. Use quick reference for tool behavior

### For Validation
1. Test all 7 tools with provided examples
2. Verify storage persistence across sessions
3. Test import/export round-trip
4. Validate with both Simple and ReAct agents
5. Performance check (expect < 10ms per operation)

---

## Questions & Clarifications

### Scope Choice
- **global**: Use for agent state, shared preferences, configuration
- **document**: Use for per-document analysis, workflow state
- **session**: Use for temporary request-scoped data

### When to Use Expiration
- Cached computations: 24 hours
- Temporary workflow state: 7 days
- Permanent data: Don't set expiration (None)

### Merge Modes for Import
- **add**: Skip existing keys (non-destructive, safe)
- **replace**: Overwrite all (destructive)
- **merge**: Newer timestamp wins (recommended for collaboration)

### Storage Size Expectations
- Typical memory entry: < 1 KB
- 1000 memories: < 1 MB
- No hard limits - limited only by disk space

---

## Appendix: File Locations

**Design documents created in this phase**:
```
/Users/clchinkc/Documents/GitHub/document-mcp/docs/
├── CONTEXT_MANAGEMENT_SYSTEM.md         # Main architecture (3500 lines)
├── CONTEXT_TECHNICAL_SPEC.md            # Implementation guide (1500 lines)
├── CONTEXT_IMPLEMENTATION_ROADMAP.md    # Step-by-step plan (400 lines)
├── CONTEXT_QUICK_REFERENCE.md           # Quick guide (700 lines)
└── PHASE_4_3_SUMMARY.md                 # This summary (400 lines)
```

**Implementation will create**:
```
document_mcp/
├── models/
│   └── context.py                       # NEW: Data models
├── storage/
│   └── context_storage.py               # NEW: Storage layer
├── tools/
│   └── context_tools.py                 # NEW: All 7 tools
└── doc_tool_server.py                   # MODIFIED: Register tools

tests/
├── unit/
│   ├── test_context_models.py           # NEW
│   ├── test_context_storage.py          # NEW
│   └── test_context_tools.py            # NEW
└── integration/
    └── test_context_integration.py      # NEW

.context/                                # NEW: Runtime storage
├── memories.json
├── sessions.json
├── index.json
└── snapshots/
```

---

## Contact & Questions

For questions about the design:
- Review the detailed design docs (linked above)
- Check technical spec for implementation details
- Reference quick reference for tool usage
- See implementation roadmap for timeline

**Design is ready for implementation**.

