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
