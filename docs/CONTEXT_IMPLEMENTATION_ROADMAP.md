# Context Management System - Implementation Roadmap

## Phase 4.3: One-Context Persistent Memory System

**Timeline**: Estimated 2-3 development sessions
**Scope**: Add persistent cross-session memory to Document MCP
**Complexity**: Low - Straightforward JSON-based storage with no external dependencies

---

## Milestones

### Milestone 1: Data Models (Session 1a - 30 mins)
**Goal**: Define all Pydantic models for context system

**Tasks**:
- [ ] Create `document_mcp/models/context.py`
- [ ] Implement `MemoryEntry` model
- [ ] Implement `MemoryStore` model
- [ ] Implement `SessionMetadata` model
- [ ] Implement `SessionHistory` model
- [ ] Implement `ContextSnapshot` model
- [ ] Add to `document_mcp/models/__init__.py` exports
- [ ] Type hints verified with mypy

**Testing**:
- [ ] Unit tests for model validation
- [ ] Test Pydantic serialization/deserialization
- [ ] Test datetime handling

**Files Modified**:
- `/Users/clchinkc/Documents/GitHub/document-mcp/document_mcp/models/context.py` (NEW)
- `/Users/clchinkc/Documents/GitHub/document-mcp/document_mcp/models/__init__.py`

---

### Milestone 2: Storage Layer (Session 1b - 30 mins)
**Goal**: Implement persistent JSON-based storage

**Tasks**:
- [ ] Create `document_mcp/storage/context_storage.py`
- [ ] Implement `ContextStorage` class with methods:
  - [ ] `load_memories()` - Load MemoryStore from JSON
  - [ ] `save_memories()` - Save MemoryStore to JSON
  - [ ] `load_sessions()` - Load SessionHistory from JSON
  - [ ] `save_sessions()` - Save SessionHistory to JSON
  - [ ] `_update_index()` - Update quick metadata index
- [ ] Create directory structure:
  - [ ] `.context/` root directory
  - [ ] `.context/snapshots/` for exported snapshots
- [ ] Implement `get_context_storage()` singleton
- [ ] Error handling for file I/O

**Testing**:
- [ ] Unit tests for storage operations
- [ ] Test file I/O with various data sizes
- [ ] Test directory creation

**Files Modified**:
- `/Users/clchinkc/Documents/GitHub/document-mcp/document_mcp/storage/context_storage.py` (NEW)

---

### Milestone 3: Core Tools - Memory Operations (Session 2 - 1 hour)
**Goal**: Implement store and recall tools

**Tasks**:
- [ ] Create `document_mcp/tools/context_tools.py`
- [ ] Implement `store_memory()` tool:
  - [ ] Create new memory entries
  - [ ] Update existing memories (by key+scope+document)
  - [ ] Handle expiration timestamps
  - [ ] UUID generation
  - [ ] Timestamp tracking
- [ ] Implement `recall_memory()` tool:
  - [ ] Filter by key
  - [ ] Filter by scope
  - [ ] Filter by document_name
  - [ ] Filter by tags
  - [ ] Auto-remove expired entries
  - [ ] Limit results
- [ ] Register tools with MCP server in `doc_tool_server.py`
- [ ] Add logging with `@log_mcp_call` decorator

**Testing**:
- [ ] Unit tests for store_memory
- [ ] Unit tests for recall_memory
- [ ] Test expiration cleanup
- [ ] Test filtering logic

**Files Modified**:
- `/Users/clchinkc/Documents/GitHub/document-mcp/document_mcp/tools/context_tools.py` (NEW)
- `/Users/clchinkc/Documents/GitHub/document-mcp/document_mcp/doc_tool_server.py` (register tools)
- `/Users/clchinkc/Documents/GitHub/document-mcp/document_mcp/tools/__init__.py` (register function)

---

### Milestone 4: Export/Import Tools (Session 2 - 30 mins)
**Goal**: Implement snapshot export and import

**Tasks**:
- [ ] Implement `export_context()` tool:
  - [ ] Filter by scope
  - [ ] Filter by document names
  - [ ] Include/exclude session history
  - [ ] Generate snapshot ID
  - [ ] Write to `.context/snapshots/snapshot_*.json`
  - [ ] Return file path and metadata
- [ ] Implement `import_context()` tool:
  - [ ] Validate snapshot JSON format
  - [ ] Implement merge modes: 'add', 'replace', 'merge'
  - [ ] Auto-remove expired entries
  - [ ] Handle conflicts based on merge_mode
  - [ ] Return import statistics
- [ ] Error handling for file operations
- [ ] Path validation and safety

**Testing**:
- [ ] Unit tests for export_context
- [ ] Unit tests for import_context
- [ ] Test all merge modes
- [ ] Test invalid file handling

**Files Modified**:
- `/Users/clchinkc/Documents/GitHub/document-mcp/document_mcp/tools/context_tools.py` (add functions)

---

### Milestone 5: Utility Tools (Session 2 - 20 mins)
**Goal**: Implement list, delete, and stats tools

**Tasks**:
- [ ] Implement `list_memories()` tool:
  - [ ] Filter by scope
  - [ ] Filter by document_name
  - [ ] Show metadata (ID, key, tags, timestamps)
  - [ ] Generate value preview (first 100 chars)
  - [ ] Show expiration info
- [ ] Implement `delete_memory()` tool:
  - [ ] Delete by memory_id
  - [ ] Delete by key (with scope)
  - [ ] Return deleted count
- [ ] Implement `get_context_stats()` tool:
  - [ ] Total memory count
  - [ ] Count by scope
  - [ ] Count by source
  - [ ] Session statistics
  - [ ] Storage size
  - [ ] Age information (oldest/newest)

**Testing**:
- [ ] Unit tests for each tool
- [ ] Test filtering and counting logic

**Files Modified**:
- `/Users/clchinkc/Documents/GitHub/document-mcp/document_mcp/tools/context_tools.py` (add functions)

---

### Milestone 6: Unit Tests (Session 3a - 45 mins)
**Goal**: Comprehensive unit test coverage

**Tasks**:
- [ ] Create `tests/unit/test_context_models.py`:
  - [ ] Test MemoryEntry validation
  - [ ] Test MemoryStore operations
  - [ ] Test SessionMetadata
  - [ ] Test ContextSnapshot
  - [ ] Test datetime handling
  - [ ] Test Pydantic serialization
- [ ] Create `tests/unit/test_context_storage.py`:
  - [ ] Test load/save operations
  - [ ] Test directory structure creation
  - [ ] Test file I/O errors
  - [ ] Test singleton instance
- [ ] Create `tests/unit/test_context_tools.py`:
  - [ ] Test store_memory with various inputs
  - [ ] Test recall_memory filtering
  - [ ] Test expiration logic
  - [ ] Test export/import round-trip
  - [ ] Test merge modes
  - [ ] Test delete_memory
  - [ ] Test stats calculation
- [ ] Run mypy type checking
- [ ] Achieve > 85% coverage for context modules

**Files Created**:
- `/Users/clchinkc/Documents/GitHub/document-mcp/tests/unit/test_context_models.py`
- `/Users/clchinkc/Documents/GitHub/document-mcp/tests/unit/test_context_storage.py`
- `/Users/clchinkc/Documents/GitHub/document-mcp/tests/unit/test_context_tools.py`

---

### Milestone 7: Integration Tests (Session 3b - 30 mins)
**Goal**: Test context tools with real MCP server

**Tasks**:
- [ ] Create `tests/integration/test_context_integration.py`:
  - [ ] Test agent calling context tools via MCP
  - [ ] Test store_memory → recall_memory round trip
  - [ ] Test export_context → import_context round trip
  - [ ] Test concurrent operations
  - [ ] Test with real document operations
- [ ] Test with both Simple and ReAct agents
- [ ] Verify MCP tool registration

**Files Created**:
- `/Users/clchinkc/Documents/GitHub/document-mcp/tests/integration/test_context_integration.py`

---

### Milestone 8: Agent Integration Examples (Session 4a - 45 mins)
**Goal**: Demonstrate usage patterns in agents

**Tasks**:
- [ ] Create `CONTEXT_USAGE_EXAMPLES.md`:
  - [ ] Pattern 1: Workflow state tracking
  - [ ] Pattern 2: Analysis caching
  - [ ] Pattern 3: Session resumption
  - [ ] Pattern 4: Multi-agent coordination
- [ ] Update `src/agents/simple_agent/prompts.py`:
  - [ ] Add context tool descriptions to system prompt
  - [ ] Document when to use context tools
- [ ] Update `src/agents/react_agent/prompts.py`:
  - [ ] Add context tool descriptions
- [ ] Create example scripts:
  - [ ] `scripts/examples/context_workflow_example.py`
  - [ ] `scripts/examples/context_sharing_example.py`
- [ ] Update `CLAUDE.md` with context system info

**Files Created/Modified**:
- `docs/CONTEXT_USAGE_EXAMPLES.md` (NEW)
- `scripts/examples/context_workflow_example.py` (NEW)
- `scripts/examples/context_sharing_example.py` (NEW)
- `src/agents/shared/tool_descriptions.py` (add context tools)
- `CLAUDE.md` (update)

---

### Milestone 9: Documentation & Polish (Session 4b - 30 mins)
**Goal**: Complete documentation and prepare for release

**Tasks**:
- [ ] Update `document_mcp/README.md`:
  - [ ] Add context system section
  - [ ] Link to CONTEXT_MANAGEMENT_SYSTEM.md
  - [ ] Show quick start example
- [ ] Update main `README.md`:
  - [ ] Document context management feature
  - [ ] Add to feature table
- [ ] Create quick reference card:
  - [ ] `docs/CONTEXT_QUICK_REFERENCE.md`
  - [ ] Tool summary with examples
- [ ] Update `CHANGELOG.md`:
  - [ ] Document Phase 4.3 additions
- [ ] Run full test suite:
  - [ ] All tests pass
  - [ ] Coverage > 85%
  - [ ] Type checking passes
  - [ ] Linting passes
- [ ] Code review checklist

**Files Modified**:
- `document_mcp/README.md`
- `README.md`
- `CHANGELOG.md`
- `docs/CONTEXT_QUICK_REFERENCE.md` (NEW)

---

### Milestone 10: Final Validation & Release (Session 5 - 30 mins)
**Goal**: Verify everything works end-to-end

**Tasks**:
- [ ] Run all tests: `uv run pytest`
- [ ] Check type safety: `uv run mypy document_mcp/`
- [ ] Check linting: `uv run ruff check --fix`
- [ ] Run E2E workflow test
- [ ] Manual testing with real agents
- [ ] Performance check (memory/disk usage)
- [ ] Create release commit
- [ ] Tag version (e.g., v0.1.0-context)

**Verification**:
- [ ] All unit tests pass (> 85% coverage)
- [ ] All integration tests pass
- [ ] Type checking clean
- [ ] Linting clean
- [ ] Documentation complete
- [ ] Example scripts work

---

## Task Dependencies

```
Milestone 1 (Models)
    ↓
Milestone 2 (Storage)
    ↓
Milestone 3 (Core Tools) ← Milestone 4 (Export/Import)
    ↓
Milestone 5 (Utility Tools)
    ↓
Milestone 6 (Unit Tests) ← Milestone 7 (Integration Tests)
    ↓
Milestone 8 (Agent Examples) ← Milestone 9 (Documentation)
    ↓
Milestone 10 (Release)
```

---

## Testing Strategy

### Unit Tests (Milestones 6)
- Isolated component testing
- No external dependencies
- 100% coverage for models and storage
- 95%+ coverage for tools

### Integration Tests (Milestone 7)
- Real MCP server interaction
- Agent communication
- File I/O operations
- Full workflows

### Manual Testing
- Agent examples execution
- Export/import round trips
- Multi-session workflows
- Performance validation

---

## Time Estimate

| Phase | Time | Complexity |
|-------|------|-----------|
| Data Models | 30 mins | Low |
| Storage Layer | 30 mins | Low |
| Core Tools (Store/Recall) | 1 hour | Low |
| Export/Import Tools | 30 mins | Low |
| Utility Tools | 20 mins | Low |
| Unit Tests | 45 mins | Medium |
| Integration Tests | 30 mins | Medium |
| Agent Integration | 45 mins | Medium |
| Documentation | 30 mins | Low |
| Final Validation | 30 mins | Low |
| **Total** | **~4.5 hours** | **Low-Medium** |

---

## Success Criteria

- [x] All 7 core tools fully implemented
- [x] ≥ 85% test coverage for context modules
- [x] All tests passing (unit + integration)
- [x] Type safety: mypy passes without errors
- [x] Code quality: ruff linting passes
- [x] Documentation complete and examples working
- [x] Agent integration demonstrated
- [x] No breaking changes to existing tools
- [x] Performance acceptable (< 10ms per operation)

---

## Notes

- **Storage Format**: JSON for human readability and debuggability
- **No External Dependencies**: Uses only stdlib + existing Pydantic
- **Backward Compatible**: Completely additive to Document MCP
- **Extensible Design**: Supports future semantic search, auto-save, etc.
- **Agent-Optional**: Not required for existing workflows

---

## Related Documentation

- **Architecture**: `docs/CONTEXT_MANAGEMENT_SYSTEM.md`
- **Usage Examples**: `docs/CONTEXT_USAGE_EXAMPLES.md` (to be created)
- **Quick Reference**: `docs/CONTEXT_QUICK_REFERENCE.md` (to be created)
- **Phase 4 Overview**: Check project roadmap

