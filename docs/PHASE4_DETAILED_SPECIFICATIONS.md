# Phase 4: Detailed Technical Specifications

**Last Updated**: February 25, 2026
**Status**: Specifications & Architecture Documentation Complete

This document provides detailed technical specifications for Phase 4 implementation across all work streams.

---

## 4.1: Claude Code & Claude Desktop Integration Verification

### Overview
Comprehensive verification and integration testing for Document MCP with Claude Code and Claude Desktop clients.

### Integration Paths

#### Path A: Claude Code (Recommended for Development)
**Setup Steps**:
```bash
# 1. Install the package
pip install document-mcp

# 2. Add to Claude Code (automatic setup)
claude mcp add document-mcp -s local -- document-mcp stdio

# 3. Verify installation
claude mcp list
# Expected output: document-mcp: ... - ✓ Connected

# 4. Test in Claude Code with queries
"list_documents"
"create a document called 'test-book'"
```

**Verification Checklist**:
- [ ] Package installs cleanly
- [ ] Binary location findable via `which document-mcp`
- [ ] MCP server starts via stdio
- [ ] Tool discovery returns all 28 tools
- [ ] Sample operations succeed (list, create, read)

#### Path B: Claude Desktop (Production Use)
**Configuration File**: `~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "document-mcp": {
      "command": "document-mcp",
      "args": ["stdio"],
      "env": {
        "DOCUMENT_ROOT_DIR": "~/.document_storage"
      }
    }
  }
}
```

**If Binary Not in PATH**:
```json
{
  "mcpServers": {
    "document-mcp": {
      "command": "/full/path/to/document-mcp",
      "args": ["stdio"],
      "env": {}
    }
  }
}
```

### Test Scenarios

#### Scenario 1: Basic Connectivity
```
Test: Server startup and tool discovery
Expected: All 28 tools listed without errors
Tool categories verified:
  - Document (6)
  - Chapter (4)
  - Paragraph (4)
  - Content (6)
  - Metadata (3)
  - Safety (3)
  - Overview (1)
  - Discovery (1)
```

#### Scenario 2: Document Creation & Manipulation
```
Test: Create document → Add chapters → Read content
Commands:
  1. create_document(document_name="integration_test")
  2. create_chapter(document_name, chapter_name="01-intro.md", content="...")
  3. read_content(document_name, scope="document")
Expected: All operations succeed, file system reflects changes
```

#### Scenario 3: Safety & Versioning
```
Test: Snapshot creation and content status checking
Commands:
  1. Modify chapter content via replace_text
  2. check_content_status to verify modification
  3. manage_snapshots to create version snapshot
Expected: Snapshots created, can be listed and compared
```

### Success Criteria
- ✅ Claude Code integration working with simple queries
- ✅ Claude Desktop config properly recognized
- ✅ All 28 tools callable and responsive
- ✅ Document CRUD operations complete successfully
- ✅ Error messages clear and actionable
- ✅ No crashes or timeout issues

### Troubleshooting Matrix

| Issue | Cause | Solution |
|-------|-------|----------|
| "spawn document-mcp ENOENT" | Binary not in PATH | Use full path in config |
| "Connection timeout" | Server startup slow | Increase timeout, check disk space |
| "Module not found: document_mcp" | Package not installed | `pip install document-mcp` |
| Permission denied | Binary not executable | `chmod +x /path/to/binary` |
| Tools not discovered | MCP server not running | Check stdio transport, restart Claude |

---

## 4.2: MCP 2025-06-18 Standards Compliance

### OutputSchema Implementation

#### Current Gap Analysis
- **Status**: No outputSchema on any tools
- **Impact**: Clients cannot validate responses against schema
- **Priority**: P1 (High Value, Medium Effort)

#### Implementation Strategy

**Phase 1: Identify Top 10 Tools by Usage**
```
Frequency-based ranking (estimated):
1. list_documents (highest use)
2. create_document
3. read_content
4. create_chapter
5. find_text
6. get_statistics
7. read_metadata
8. list_chapters
9. manage_snapshots
10. get_document_outline
```

**Phase 2: Generate JSON Schemas**
From Pydantic models:
- DocumentInfo → DocumentListSchema
- ChapterContent → ChapterContentSchema
- PaginatedContent → PaginationSchema
- StatisticsReport → StatsSchema
- MetadataResponse → MetadataSchema

**Phase 3: Add to Tool Decorators**
```python
# Current
@mcp_server.tool()
def list_documents() -> list[DocumentInfo]:
    ...

# Target
@mcp_server.tool()
@mcp_server.output_schema({
    "type": "array",
    "items": {
        "$ref": "#/definitions/DocumentInfo"
    }
})
def list_documents() -> list[DocumentInfo]:
    ...
```

#### Pydantic Schema Mapping
```
DocumentInfo
├── document_name: str
├── chapter_count: int
├── total_word_count: int
└── last_modified: datetime

ChapterContent
├── document_name: str
├── chapter_name: str
├── content: str
├── word_count: int
├── paragraph_count: int
└── last_modified: datetime

PaginatedContent
├── content: str
├── document_name: str
├── scope: str
├── chapter_name: str | None
├── paragraph_index: int | None
└── pagination: PaginationInfo
```

#### Testing Approach
1. Unit tests: Schema validity for each model
2. Integration tests: Tool response matches schema
3. E2E tests: Claude Desktop validates responses

---

## 4.3: OneContext-Inspired Context Management

### Architecture Design

#### Data Model: Session Memory Storage
```
document_name/
├── .context/                      # NEW: Session context directory
│   ├── session.json               # Current session metadata
│   ├── goals.md                   # Session objectives
│   ├── memories/                  # Key-value memory store
│   │   └── {memory_key}.json     # Individual memory entries
│   ├── decisions.md               # Recorded decisions
│   └── blockers.md                # Known issues/blockers
```

#### Memory Entry Structure
```json
{
  "key": "feature_x_optimization",
  "value": {
    "query": "How to optimize feature X?",
    "response": "Use caching strategy Y",
    "timestamp": "2026-02-25T16:00:00Z",
    "tags": ["optimization", "performance"],
    "session_id": "sess_xyz",
    "expires": null
  }
}
```

### Tools to Implement

#### Quick Wins (Phase 3a: 1-2 weeks)

**Tool 1: store_memory()**
```
Input:
  - key: str (memory identifier)
  - value: Any (data to store)
  - metadata?: dict (tags, expiry, etc.)

Output:
  - OperationStatus (success, message)

Storage:
  - .context/memories/{key}.json
  - Indexed for fast retrieval
```

**Tool 2: recall_memory()**
```
Input:
  - key: str (memory key to retrieve)
  - pattern?: str (regex match)

Output:
  - memory_value: Any
  - retrieved_at: datetime
  - stored_at: datetime
  - tags: list[str]

Behavior:
  - Exact match lookup
  - Regex pattern matching for related memories
```

**Tool 3: export_context()**
```
Input:
  - format: str ("json" | "yaml" | "markdown")
  - include_session?: bool
  - include_goals?: bool
  - include_decisions?: bool

Output:
  - exported_file_path: str
  - file_size: int
  - entry_count: int
  - format_used: str

Effect:
  - Creates .context/export_{timestamp}.{ext}
  - Portable context package for sharing
```

**Tool 4: import_context()**
```
Input:
  - context_package: str (path to export file)
  - merge?: bool (merge vs overwrite)
  - validate?: bool (validate before importing)

Output:
  - OperationStatus
  - entries_imported: int
  - conflicts_detected: int

Behavior:
  - Load session state from package
  - Validate JSON/YAML format
  - Merge with existing or overwrite
```

**Tool 5: Session Metadata Storage**
```
File: .context/session.json

Structure:
{
  "session_id": "uuid",
  "started_at": "2026-02-25T16:00:00Z",
  "goals": ["complete Phase 4", "..."],
  "progress": {
    "tasks_completed": 5,
    "tasks_in_progress": 3,
    "current_focus": "outputSchema implementation"
  },
  "blockers": [
    {
      "blocker": "FastMCP schema support unclear",
      "severity": "low",
      "action_needed": "Verify docs"
    }
  ]
}
```

#### Medium Effort (Phase 3b: 2-4 weeks)

**Git-Backed Snapshots**
- Initialize Git repo in document directory on creation
- Atomic commits on every edit with user attribution
- Full history traversal via `get_version_history()`
- Branch support for experimental changes

**Document Relationship Graph**
- Semantic linking between documents
- Dependency tracking
- Context-aware operations

---

## 4.4: Git-Backed Version History

### Architecture Design

#### Current State
- Timestamp-based `.snapshots/` directories
- No cross-session history
- Manual snapshot management

#### Target State
- Git repository per document
- Automatic commits on all edits
- Full version history with diffs
- User attribution in commits

### Implementation Phases

**Phase 1: Git Initialization**
- Auto-initialize `.git/` on `create_document()`
- Configure user email/name from MCP context
- Create initial commit with document structure

**Phase 2: Tool Migration**
- `manage_snapshots()` → Git operations
  - Create: `git commit -am "snapshot"`
  - List: `git log --oneline`
  - Restore: `git checkout <hash>`

- `check_content_status()` → `git status` equivalent
  - Show modified files
  - Indicate tracked/untracked

- `diff_content()` → `git diff` with enhanced output
  - Character-level diffs
  - Statistical summary

**Phase 3: New Tools**
- `get_version_history(document_name, limit=10)` → Get commit log
- `checkout_version(document_name, version_hash)` → Restore version
- `compare_versions(doc, v1_hash, v2_hash)` → Detailed diff

### Storage Structure
```
document_name/
├── .git/                    # Git repository (created on document init)
│   ├── objects/
│   ├── refs/
│   ├── HEAD
│   └── config
├── 01-chapter.md
├── 02-chapter.md
├── summaries/
├── metadata/
├── .snapshots/             # LEGACY: Can be migrated or retired
├── .embeddings/
└── .context/
```

### Commit Strategy
```
Commit message format:
{operation} {scope}: {description}
{user}

Examples:
- "edit chapter: updated introduction narrative"
- "add paragraph: new section on performance"
- "batch operation: restructure chapters 3-5"

Commits include:
- User attribution
- Timestamp
- Operation type
- Scope (document/chapter/paragraph)
- Diff statistics
```

---

## 4.5: Story MCP Rename

### Strategic Rationale
- Current name "Document MCP" is generic
- "Story MCP" positions as AI-native story writing platform
- Aligns with vision of story-specific features (character tracking, plot arcs)
- Maintains backward compatibility with alias

### Rename Components

| Component | Current | Target | Effort | Risk |
|-----------|---------|--------|--------|------|
| GitHub repo | document-mcp | story-mcp | Low | Low |
| PyPI package | document-mcp | story-mcp | Low | Medium |
| Module name | document_mcp | story_mcp | High | High |
| CLI command | document-mcp | story-mcp | Low | Low |
| Classes | DocumentMCP | StoryMCP | Medium | Low |
| Imports | `from document_mcp` | `from story_mcp` | High | High |

### Backward Compatibility Strategy
- PyPI: Keep `document-mcp` as deprecated alias package
- Python: `from document_mcp import *` → internal redirect to `story_mcp`
- CLI: Create wrapper script for old command name
- 6-month deprecation period with warnings

### Implementation Approach
1. Create new GitHub repository: `story-mcp`
2. Fork and perform complete rename in new repo
3. Keep `document-mcp` as read-only archive
4. Publish new `story-mcp` package to PyPI
5. Maintain compatibility shim for 6 months

---

## Summary & Next Steps

### What's Complete
✅ Phase 4 architecture and specifications defined
✅ Implementation roadmap established
✅ Technical details for all work streams documented
✅ Integration patterns for Claude Code/Desktop specified
✅ Schema mapping strategy for MCP compliance
✅ Context management system designed
✅ Git integration architecture planned
✅ Rename strategy and execution plan created

### Recommended Execution Order
1. **Week 1-2**: Claude Code/Desktop verification + MCP outputSchema
2. **Week 2-3**: Memory tools quick wins
3. **Week 3-4**: Medium-effort features
4. **Week 4-6**: Git integration
5. **Week 6-8**: Rename execution + release

### Success Metrics
- All 28 tools have outputSchema by week 2
- Memory tools functional with persistence by week 3
- Git-backed version history operational by week 6
- Story MCP published to PyPI by week 8
- Zero breaking changes for existing users

---

## References

- [Phase 4 Implementation Plan](./PHASE4_IMPLEMENTATION_PLAN.md)
- [TODO.md Phase 4 Section](./TODO.md#phase-4-mcp-standards-compliance--context-enhancement)
- [MCP Design Patterns Guide](./MCP_DESIGN_PATTERNS.md)
- [Architectural Analysis](./ARCHITECTURAL_ANALYSIS.md)
