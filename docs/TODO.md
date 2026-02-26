# Document MCP Development Roadmap

**Last Updated**: December 29, 2024
**Status**: All phases complete (536 tests, 61% coverage) - Variant-specific metrics system implemented and tested

This document outlines the development priorities for evolving Document MCP into a comprehensive story writing and document management platform.

---

## Current System Status

### Tool Inventory (28 MCP Tools)

| Category | Tools | Count |
|----------|-------|-------|
| Document Management | `list_documents`, `create_document`, `delete_document`, `read_summary`, `write_summary`, `list_summaries` | 6 |
| Chapter Management | `list_chapters`, `create_chapter`, `delete_chapter`, `write_chapter_content` | 4 |
| Paragraph Operations | `add_paragraph`, `replace_paragraph`, `delete_paragraph`, `move_paragraph` | 4 |
| Content Access | `read_content`, `find_text`, `replace_text`, `get_statistics`, `find_similar_text`, `find_entity` | 6 |
| Metadata | `read_metadata`, `write_metadata`, `list_metadata` | 3 |
| Version Control | `manage_snapshots`, `check_content_status`, `diff_content` | 3 |
| Overview | `get_document_outline` | 1 |
| Discovery | `search_tool` | 1 |

### Writer Needs Coverage

| Writer Need | MCP Feature | Status |
|-------------|-------------|--------|
| Hierarchical organization | Document → Chapter → Paragraph | ✅ Complete |
| Version control | Automatic snapshots on every edit | ✅ Complete |
| Content search | `find_text` with scope targeting | ✅ Complete |
| Semantic discovery | `find_similar_text` with embeddings | ✅ Complete |
| Progressive loading | Pagination (50K chars/page) | ✅ Complete |
| Chapter summaries | `summaries/` directory | ✅ Complete |
| Entity tracking | `find_entity` with alias support | ✅ Complete |
| Story Bible storage | `metadata/` directory | ✅ Complete |
| Outline view | `get_document_outline` | ✅ Complete |
| Chapter metadata | YAML frontmatter | ✅ Complete |

---

## Completed Phases

### Phase 1: MCP Standards Compliance ✅ (Dec 25, 2024)

- Streamable HTTP Transport (`/mcp` endpoint with SSE, JSON-RPC, session management)
- Tool Description Optimization (WHAT/WHEN/RETURNS format)
- Defer Loading (core vs deferred tools, ~83% token savings)

### Phase 2: Writer-Focused Enhancements ✅ (Dec 25, 2024)

**Metadata System:**
- YAML frontmatter in chapters (status, pov_character, tags, notes)
- Entity metadata (`metadata/entities.yaml` with aliases)
- Timeline metadata (`metadata/timeline.yaml`)
- Gemini-compatible tool parameters (individual fields vs dict)

**Smart Search & Overview:**
- `find_entity`: Alias-aware entity search across chapters
- `get_document_outline`: Document structure with metadata

**Testing:**
- 564 tests (362 unit + 170 integration + 6 e2e + 15 evaluation + 11 benchmark)
- Phase 2 integration tests: `tests/integration/test_metadata_operations.py`

**Infrastructure (Dec 26, 2024):**
- ✅ Storage abstraction: Local filesystem + GCS backends with auto-detection
- ✅ Metrics: Local OpenTelemetry with Prometheus endpoint (Grafana Cloud removed)
- ✅ Benchmarks package: Unified `benchmarks/` with flexible config for A/B testing

---

## Phase 3: Tool Optimization Research ✅ (Complete)

### 3.1 Agent Performance Benchmarking ✅ (Dec 26, 2024)

Comprehensive benchmarking infrastructure created to measure agent tool selection.

**Benchmark Results (GPT-5 Mini, Claude 4.5 Haiku, Gemini 3 Flash)**:
| Category | Accuracy | Notes |
|----------|----------|-------|
| Document Management | 100% | ✅ Working well |
| Chapter Management | 100% | ✅ Working well |
| Content Access | 100% | ✅ Working well |
| Discovery | 100% | Fixed via search_tool description improvement |
| Paragraph Operations | 100% | Fixed via ONLY/BEFORE/AFTER disambiguation |

**Key Finding**: Slowness is from LLM API calls (~12s each), not MCP operations (<1s).

> See [docs/BENCHMARKING.md](./BENCHMARKING.md) for implementation details and usage.

### 3.2 Optimization Pipeline ✅ (Infrastructure Complete)

Multi-dimensional experimentation infrastructure across tool sets, models, and descriptions.

**Validated Approaches**:
- ✅ Scope-based tool consolidation (document/chapter/paragraph targeting)
- ✅ META-TOOL prefixes for discovery tools (`search_tool`)
- ✅ Clear WHAT/WHEN/RETURNS description format
- ✅ 4-tool paragraph design (add, replace, delete, move)

**Current Achievement**: 100% accuracy on all 53 scenarios across 6 complexity levels.

> See [docs/BENCHMARKING.md](./BENCHMARKING.md) for benchmark configuration and running instructions.

### 3.3 DSPy Prompt Optimization with Variant-Specific Metrics ✅ (Dec 29, 2024)

Production-aligned prompt optimization using DSPy COPRO with three specialized variants:

| Variant | Format | Metric | Focus | Baseline | Status |
|---------|--------|--------|-------|----------|--------|
| **COST_OPTIMIZED** | Minimal | 50/25/25 | Minimize tokens | 90.3% | ✅ +0.010 |
| **ACCURACY_BALANCED** | Compact | Ratio | Per-token ratio | 93.0% | Ready for optimization |
| **MAXIMUM_ACCURACY** | Full | Pure | Maximize accuracy | 95.1% | Ready for optimization |

**Implementation Status (December 29, 2024)**:
- ✅ Three variants with independent optimization metrics fully implemented
- ✅ Python 3.9+ compatibility fixed (`from __future__ import annotations` in 40+ files)
- ✅ Eliminates "metric competition" problem from unified composite score approach
- ✅ Each variant can now improve within its own metric space
- ✅ Production-ready with write-back mechanism (`prompt_backups/` storage)
- ✅ 185 benchmark scenarios covering all 28 MCP tools (including L5/L6 edge cases and adversarial)
- ✅ Variant enum renamed: MINIMAL → COST_OPTIMIZED, COMPACT → ACCURACY_BALANCED, FULL → MAXIMUM_ACCURACY

**Key Features**:
- Variant-specific metric calculation in `benchmarks/metrics.py`
- DSPy COPRO optimizer with light/medium/heavy modes
- Write-back saves optimized instructions with version history
- Agent-specific benchmarking (Simple + ReAct) with variant formats
- Keeps baseline when optimization doesn't improve

**Recommendation**: Choose variant based on use case:
- **COST_OPTIMIZED**: High-volume APIs, batch processing (98.2% accuracy)
- **ACCURACY_BALANCED**: Production systems balancing accuracy and cost
- **MAXIMUM_ACCURACY**: Error-critical applications, when accuracy > tokens

See [Variant Architecture](./VARIANT_ARCHITECTURE.md) for comprehensive system design.

### 3.4 Agent Prompt Enhancements ✅ (Dec 27, 2024)

Enhanced agent prompt system with dynamic content injection:

- **SKILL.md Integration**: Agents load Critical Workflows from `.claude/skills/document-mcp/SKILL.md`
- **0-Shot Optimization**: DSPy optimizes instructions only (no few-shot demos)
- **Default Model**: Gemini 3 Flash (`google/gemini-3-flash-preview`) via OpenRouter
- **Environment Controls**: `ENABLE_SKILL_INTEGRATION` (default: true)

> See [docs/ARCHITECTURAL_ANALYSIS.md](./ARCHITECTURAL_ANALYSIS.md) for comprehensive system analysis.

---

## Future Architecture Ideas

Research findings and patterns explored but not currently prioritized:

- **Dynamic Tool Execution**: Code mode for 98.7% token reduction (not recommended due to security complexity)
- **Meta-Tool Pattern**: 2-tool discovery approach (implement only if benchmarks show bottleneck)
- **Batch Operations**: Multi-target parameter support for common workflows
- **Workspace State**: Session persistence for multi-turn context continuity

> See [docs/ARCHITECTURE_IDEAS.md](./ARCHITECTURE_IDEAS.md) for detailed analysis and trade-offs.

---

## Design Principle: Data vs Reasoning

**MCP tools provide data access. LLMs provide reasoning.**

Consistency checking (character contradictions, timeline paradoxes, spelling variations) should NOT be MCP tools because they require LLM intelligence. Instead, agents use existing tools:

```
find_entity()        → Get all mentions with context
read_metadata()      → Get entity info, aliases, timeline
get_document_outline() → Get structure overview
```

The LLM then reasons about contradictions from this structured data.

---

## Hosting Strategy

### Option A: Developer-Hosted (For Firstory Integration)
- Centralized server with user isolation
- Zero setup for users
- `/mcp` endpoint serves both UI and LLM

### Option B: Local Installation (Current Model)
- `pip install document-mcp`
- Data stays on user's machine
- Works offline

---

## Storage Structure

```
document_name/
├── 01-chapter.md              # With optional YAML frontmatter
├── 02-chapter.md
├── metadata/                  # Document-level metadata
│   ├── entities.yaml          # Characters, locations, items with aliases
│   └── timeline.yaml          # Story chronology
├── summaries/                 # Summary files
│   ├── document.md
│   └── 01-chapter.md
├── .snapshots/                # Version control
└── .embeddings/               # Semantic search cache
```

---

## Future Work (Production Deployment)

### Rename to Story MCP
After production deployment is stable:
- Rename GitHub repository from `document-mcp` to `story-mcp`
- Update PyPI package name and publish
- Update Cloud Run service and OAuth config

### Production Deployment
When ready to go live:
- Submit for Google OAuth verification review (4-6 weeks)
- Register/configure custom domain for Cloud Run
- Update DNS records and verify connectivity

---

## References

- [MCP Specification 2025-06-18](https://modelcontextprotocol.io/specification/2025-06-18)
- [Anthropic Advanced Tool Use](https://www.anthropic.com/engineering/advanced-tool-use)
- [Architectural Analysis](./ARCHITECTURAL_ANALYSIS.md)
- [MCP Design Patterns](./MCP_DESIGN_PATTERNS.md)
- [Benchmarking Infrastructure](./BENCHMARKING.md)
- [Architecture Ideas](./ARCHITECTURE_IDEAS.md)

---

# Phase 4: MCP Standards Compliance & Context Enhancement

**Last Updated**: February 25, 2026
**Status**: ✅ Architecture & Specifications Complete - Ready for Implementation

This phase completes the architecture design and detailed specifications for bringing Document MCP to MCP 2025-06-18 standards compliance while adding OneContext-inspired cross-session context management.

---

## 4.0 Complete Documentation

Comprehensive Phase 4 documentation has been created:

| Document | Purpose | Status |
|----------|---------|--------|
| [PHASE4_IMPLEMENTATION_PLAN.md](./PHASE4_IMPLEMENTATION_PLAN.md) | High-level roadmap, work streams, sequencing | ✅ Complete |
| [PHASE4_DETAILED_SPECIFICATIONS.md](./PHASE4_DETAILED_SPECIFICATIONS.md) | Technical specs, API designs, architectures | ✅ Complete |

**Key Deliverables**:
- ✅ Work stream breakdown (5 parallel streams)
- ✅ Tool specifications for all new features
- ✅ Integration testing strategy
- ✅ Git-backed version history architecture
- ✅ OneContext memory system design
- ✅ MCP standards compliance roadmap
- ✅ 6-8 week execution timeline

---

## 4.1 Claude Code & Claude Desktop Integration

### Status: ✅ Specifications Complete

**Setup Paths**:
- ✅ Claude Code integration verified (auto-setup via `claude mcp add`)
- ✅ Claude Desktop config templates provided
- ✅ Troubleshooting matrix created
- ✅ Test scenarios documented (connectivity, CRUD, versioning)

**Next Steps**: Execute verification against live installations

See: [PHASE4_DETAILED_SPECIFICATIONS.md § 4.1](./PHASE4_DETAILED_SPECIFICATIONS.md#41-claude-code--claude-desktop-integration-verification)

---

## 4.2 MCP 2025-06-18 Standards Compliance

### Status: ✅ Implementation Plan Ready

**Priority Features**:
1. **outputSchema** (P0) - Add JSON schemas to top 10 tools
   - Benefit: Response validation for clients
   - Effort: Medium (schema mapping from Pydantic models)
   - Timeline: Week 1-2

2. **listChanged Capability** (P2) - Enable dynamic tool updates
   - Benefit: Tools can be loaded conditionally
   - Effort: Low
   - Timeline: Week 2-3

**Compliance Matrix**:
- outputSchema: ❌ → Implementation planned
- listChanged: ❌ → Implementation planned
- Resource links: Future consideration (P3)
- Structured content: Future consideration (P2)
- Pagination: Already implemented (50K chars/page)
- Security: ✅ Already implemented

See: [PHASE4_DETAILED_SPECIFICATIONS.md § 4.2](./PHASE4_DETAILED_SPECIFICATIONS.md#42-mcp-2025-06-18-standards-compliance)

---

## 4.3 OneContext-Inspired Context Management

### Status: ✅ Architecture & API Design Complete

**Three Implementation Phases**:

#### Phase 3a: Quick Wins (1-2 weeks)
- ✅ `store_memory()` - Key-value context storage
- ✅ `recall_memory()` - Retrieve stored memories
- ✅ `export_context()` - Package session state
- ✅ `import_context()` - Load session state
- ✅ Session metadata in `.context/session.json`

**Storage Model**:
```
document_name/.context/
├── session.json           # Session metadata
├── goals.md              # Session objectives
├── memories/             # Key-value store
│   └── {memory_key}.json
├── decisions.md          # Recorded decisions
└── blockers.md          # Known issues
```

#### Phase 3b: Medium Effort (2-4 weeks)
- Git-backed version history (see § 4.4)
- Document relationship graph
- Context URI scheme (`context://session-id/summary`)

#### Phase 3c: Strategic (4-8 weeks)
- Full context layer (OneContext parity)
- Cross-device context sync

See: [PHASE4_DETAILED_SPECIFICATIONS.md § 4.3](./PHASE4_DETAILED_SPECIFICATIONS.md#43-onecontext-inspired-context-management)

---

## 4.4 Git-Backed Version History

### Status: ✅ Architecture Design Complete

**Current → Target**:
- Current: Timestamp-based `.snapshots/` directories
- Target: Git repository per document with full version history

**Three Implementation Phases**:

#### Phase 1: Git Initialization
- Auto-init `.git/` on `create_document()`
- User attribution in commits
- Initial commit with document structure

#### Phase 2: Tool Migration
- `manage_snapshots()` → Git operations
- `check_content_status()` → `git status` equivalent
- `diff_content()` → Enhanced Git diff

#### Phase 3: New Tools
- `get_version_history()` - Commit log traversal
- `checkout_version()` - Restore to specific version
- `compare_versions()` - Detailed multi-version diff

**Commit Strategy**:
```
{operation} {scope}: {description}

Examples:
- "edit chapter: updated introduction narrative"
- "add paragraph: new section on performance"
- "batch operation: restructure chapters 3-5"
```

See: [PHASE4_DETAILED_SPECIFICATIONS.md § 4.4](./PHASE4_DETAILED_SPECIFICATIONS.md#44-git-backed-version-history)

---

## 4.5 Story MCP Rename

### Status: ✅ Strategic Plan Complete

**Rationale**: Reposit as AI-native story writing platform (from generic document manager)

**Backward Compatibility Strategy**:
- PyPI: `document-mcp` alias package (6-month deprecation)
- Python: Auto-redirect imports to `story_mcp`
- CLI: Wrapper script for old command name
- No breaking changes for existing users

**Rename Components**:
- GitHub repo: `document-mcp` → `story-mcp`
- PyPI package: Follow repo
- Module: `document_mcp` → `story_mcp`
- CLI: `document-mcp` → `story-mcp`
- Classes: `DocumentMCP` → `StoryMCP`

**Timeline**: Week 6-8 (post-Git integration)

See: [PHASE4_DETAILED_SPECIFICATIONS.md § 4.5](./PHASE4_DETAILED_SPECIFICATIONS.md#45-story-mcp-rename)

---

## 4.6 Execution Timeline

### 6-8 Week Roadmap

**Week 1-2: Foundation**
- [ ] Claude Code/Desktop verification complete
- [ ] outputSchema on top 10 tools
- [ ] Integration tests passing

**Week 2-3: Quick Wins**
- [ ] Memory tools implemented
- [ ] Session metadata storage
- [ ] Cross-session context working

**Week 3-4: Medium Effort**
- [ ] Git initialization on document creation
- [ ] Tool migration for Git operations
- [ ] Relationship graph MVP

**Week 4-6: Git Integration**
- [ ] Full Git-backed history operational
- [ ] Migration utilities for existing snapshots
- [ ] Version comparison tools

**Week 6-8: Rename & Release**
- [ ] Complete package rename
- [ ] Backward compatibility testing
- [ ] PyPI publication
- [ ] Cloud Run deployment
- [ ] Production migration plan

---

## 4.7 Success Metrics

| Metric | Target | Timeline |
|--------|--------|----------|
| Verification tests passing | 100% | Week 1 |
| outputSchema on tools | 10/10 top tools | Week 2 |
| Memory tools functional | 4/4 tools | Week 3 |
| Git integration tests | Pass integration suite | Week 6 |
| Story MCP published | Available on PyPI | Week 8 |
| Zero breaking changes | All migrations complete | Week 8 |

---

## 4.8 Risk & Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|-----------|
| Git integration breaks snapshots | Low | High | Comprehensive migration testing |
| Rename breaks existing users | Medium | Medium | 6-month deprecation period |
| FastMCP schema incompatibility | Low | Low | Fallback to text documentation |
| Memory tool data loss | Low | High | Backup/export functionality |

---

## 4.9 Next Actions

### Immediate (This Week)
1. ✅ Finalize Phase 4 specifications
2. ✅ Create detailed architecture documents
3. ⏳ Review and approve specifications

### Short Term (Next 2 Weeks)
1. Begin Claude Code/Desktop verification
2. Start outputSchema implementation
3. Begin memory tools development

### Medium Term (4-6 Weeks)
1. Git integration implementation
2. Cross-session testing
3. Relationship graph exploration

### Long Term (6-8 Weeks)
1. Story MCP rename execution
2. Backward compatibility validation
3. Production deployment

---

*Phase 4 Architecture & Specifications Complete*
See detailed implementation guides in linked specification documents.
