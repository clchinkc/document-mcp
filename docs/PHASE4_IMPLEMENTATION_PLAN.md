# Phase 4 Implementation Plan

**Status**: Comprehensive analysis and planning (February 25, 2026)

This document synthesizes the Phase 4 implementation strategy across all work streams.

---

## Work Stream Summary

### WS1: MCP Standards Compliance (outputSchema)
**Owner**: Architecture team
**Effort**: Medium (2-3 days)
**Impact**: High - Enables strict response validation in clients

**Key Tasks**:
1. Audit current tools to identify top 10 most-used by frequency
2. Generate JSON Schema from Pydantic models
3. Add `outputSchema` parameter to tool decorators
4. Test schema validation with mock clients
5. Document schema patterns in MCP Design Patterns guide

**Success Criteria**:
- All top 10 tools have outputSchema
- Schema validation passes in integration tests
- Clients can read and validate responses

---

### WS2: OneContext-Inspired Memory Tools
**Owner**: Backend architecture team
**Effort**: Low (1 week)
**Impact**: High - Enables cross-session learning

**Phase 2a: Quick Wins (Days 1-2)**
- [ ] `store_memory(key, value, metadata?)` - Store arbitrary agent state
- [ ] `recall_memory(key)` - Retrieve stored memory
- [ ] `export_context()` - Package current session state
- [ ] `import_context(package)` - Import saved context
- [ ] Session metadata in `.context/session.json`

**Phase 2b: Medium Effort (Days 3-7)**
- [ ] Git-backed snapshot history (replace `.snapshots/`)
- [ ] Document relationship graph
- [ ] Context URI scheme (`context://session-id/summary`)

**Storage Design**:
```
document_name/
├── .context/
│   ├── session.json          # Current session metadata
│   ├── goals.md              # Session goals/objectives
│   └── memories/             # Key-value memory store
│       └── {key}.json        # Stored memory entries
├── .git/                      # Git repository (Phase 2b)
└── [existing document files]
```

---

### WS3: Claude Code/Desktop Integration Verification
**Owner**: QA team
**Effort**: Low (1-2 days)
**Impact**: Critical - Production readiness

**Verification Checklist**:
- [ ] Binary installation via pip
- [ ] PATH configuration for discovery
- [ ] Stdio communication verification
- [ ] Tool list retrieval and parsing
- [ ] Simple tool execution (list_documents)
- [ ] Error handling and recovery
- [ ] Claude Code integration test
- [ ] Claude Desktop config validation

**Test Matrix**:
| Platform | Installation | Discovery | Execution |
|----------|--------------|-----------|-----------|
| Claude Code | ✓ | ✓ | ✓ |
| Claude Desktop | ✓ | ✓ | ✓ |
| Manual CLI | ✓ | ✓ | ✓ |

---

### WS4: Git-Backed Version History
**Owner**: Backend architecture team
**Effort**: Medium (2-4 weeks)
**Impact**: Very High - Production-grade versioning

**Design Principles**:
1. Each document directory is a Git repo
2. Automatic commits on every edit
3. User attribution in commits
4. Full history traversal across sessions

**Migration Path**:
- Phase 1: Git initialization on document creation
- Phase 2: Migrate existing `.snapshots/` to Git
- Phase 3: Retire legacy snapshot system

**Tool Integration**:
- `manage_snapshots()` → Git branch/tag operations
- `check_content_status()` → `git status` equivalent
- `diff_content()` → `git diff` equivalent
- New: `get_version_history()` → Log traversal
- New: `checkout_version(hash)` → Version switching

---

### WS5: Rename to Story MCP
**Owner**: Release engineering
**Effort**: High (2-3 weeks)
**Impact**: Strategic - Brand realignment

**Rename Checklist**:

| Task | Component | Effort | Risk |
|------|-----------|--------|------|
| GitHub repo | document-mcp → story-mcp | Low | Low |
| PyPI package | document-mcp → story-mcp | Low | Medium |
| Package module | document_mcp → story_mcp | High | High |
| CLI command | document-mcp → story-mcp | Low | Low |
| Class names | DocumentMCP → StoryMCP | Medium | Low |
| All imports | document_mcp → story_mcp | High | High |
| Documentation | 100+ references | High | Low |
| Cloud Run service | Update OAuth config | Medium | Low |
| Backward compatibility | Legacy package alias | Medium | Low |

**Implementation Strategy**:
1. Create `story-mcp` as new repo (fork from current)
2. Perform complete rename in new repo
3. Maintain `document-mcp` as deprecated alias with redirect
4. PyPI package points to new repo after 6-month deprecation period

**Backward Compatibility**:
```python
# Old import still works
from document_mcp import tools  # Imports story_mcp with deprecation warning

# New import recommended
from story_mcp import tools
```

---

## Implementation Sequencing

### Timeline: 6-8 Weeks (Recommended)

**Week 1-2: Foundation (WS1 + WS3)**
- [ ] Add outputSchema to top 10 tools
- [ ] Complete Claude Code/Desktop verification
- [ ] Validate production readiness

**Week 2-3: Quick Wins (WS2a)**
- [ ] Implement memory tools
- [ ] Add session metadata storage
- [ ] Test memory persistence

**Week 3-4: Medium Effort (WS2b)**
- [ ] Git-backed version history design complete
- [ ] Document relationship graph MVP

**Week 4-6: Git Integration (WS4)**
- [ ] Git initialization on document creation
- [ ] Tool updates for Git operations
- [ ] Migration utilities for existing snapshots

**Week 6-8: Rename & Release (WS5)**
- [ ] Complete rename in new repository
- [ ] Backward compatibility testing
- [ ] PyPI publication
- [ ] Cloud Run deployment
- [ ] Production migration plan

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|-----------|
| Git integration breaks snapshots | Low | High | Comprehensive migration testing |
| Rename breaks existing users | Medium | Medium | 6-month deprecation period |
| OutputSchema incompatibility | Low | Low | Fallback to plain text schema |
| Memory tool data loss | Low | High | Backup/export functionality |
| Claude Desktop config issues | Low | Medium | Detailed troubleshooting guide |

---

## Success Criteria

**Phase 4 Complete When**:
- ✅ Claude Code/Desktop integration verified and documented
- ✅ outputSchema implemented on all high-use tools
- ✅ Memory tools functional with persistence
- ✅ Git-backed version history operational
- ✅ Rename plan approved and execution started
- ✅ All tests passing (unit + integration + E2E)
- ✅ Documentation updated

---

## References

- [TODO.md Phase 4.1 - Claude Verification](./TODO.md#41-claude-code--claude-desktop-verification)
- [TODO.md Phase 4.2 - MCP Standards](./TODO.md#42-mcp-2025-06-18-standards-compliance)
- [TODO.md Phase 4.3 - Context Management](./TODO.md#43-onecontext-inspired-context-management)
- [TODO.md Phase 4.4 - Story MCP Rename](./TODO.md#44-rename-to-story-mcp)
- [Architectural Analysis](./ARCHITECTURAL_ANALYSIS.md)
- [MCP Design Patterns](./MCP_DESIGN_PATTERNS.md)
