# docs/ — Navigation Index

**Last updated**: March 22, 2026 | **Project**: story-mcp v0.0.5

All documentation is organized into five areas. Start with the section matching your goal.

---

## 1. Project Status & Roadmap

| File | Purpose |
|------|---------|
| [TODO.md](./TODO.md) | Full development history (Phases 1–4.7) and next actions |
| [PHASE5_PLAN.md](./PHASE5_PLAN.md) | v0.0.6 feature plan (45-50 tools, April–June 2026) |

**Current state** (March 22, 2026):
- **37 MCP tools** across 10 categories
- **469 unit tests** + **185 integration tests** — all passing
- Package: `story-mcp` v0.0.5 (PyPI) | Module: `story_mcp`
- Completion reports: [PHASE4_FINAL_COMPLETION_REPORT.md](../PHASE4_FINAL_COMPLETION_REPORT.md), [PHASE4.6_COMPLETION_REPORT.md](../PHASE4.6_COMPLETION_REPORT.md)

---

## 2. Architecture & Design

| File | Purpose |
|------|---------|
| [MCP_DESIGN_PATTERNS.md](./MCP_DESIGN_PATTERNS.md) | Core MCP patterns: pagination, safety, resources, tool design |
| [ARCHITECTURAL_ANALYSIS.md](./ARCHITECTURAL_ANALYSIS.md) | System architecture deep-dive and component analysis |
| [ARCHITECTURE_IDEAS.md](./ARCHITECTURE_IDEAS.md) | Research findings and deferred architecture ideas |
| [FASTMCP_INTEGRATION_PATTERNS.md](./FASTMCP_INTEGRATION_PATTERNS.md) | FastMCP-specific integration patterns and gotchas |
| [CONTEXT_MANAGEMENT_SYSTEM.md](./CONTEXT_MANAGEMENT_SYSTEM.md) | Phase 4.3 context/memory system design reference |
| [GIT_BACKED_VERSION_HISTORY.md](./GIT_BACKED_VERSION_HISTORY.md) | Phase 4.4 git-backed version history design reference |
| [OUTPUTSCHEMA_QUICK_START.md](./OUTPUTSCHEMA_QUICK_START.md) | MCP 2025-06-18 outputSchema quick-start reference |

---

## 3. Operations & Deployment

| File | Purpose |
|------|---------|
| [DEPLOYMENT.md](./DEPLOYMENT.md) | Production deployment guide (stdio, SSE, Cloud Run) |
| [CLAUDE_INTEGRATION_TESTING.md](./CLAUDE_INTEGRATION_TESTING.md) | Claude Code/Desktop integration setup and testing |
| [FLAKY_TEST_DETECTION.md](./FLAKY_TEST_DETECTION.md) | CI flaky test detection system documentation |
| [manual_testing.md](./manual_testing.md) | Manual testing workflows and complete E2E examples |
| [TOOL_TEST_PROMPT.md](./TOOL_TEST_PROMPT.md) | Test prompt for verifying all 37 MCP tools with Claude |

---

## 4. Migration & Deprecation

| File | Purpose |
|------|---------|
| [STORY_MCP_MIGRATION.md](./STORY_MCP_MIGRATION.md) | Migration guide: `document_mcp` → `story_mcp` |
| [DEPRECATION_NOTICE.md](./DEPRECATION_NOTICE.md) | Deprecation timeline and backward compatibility policy |

---

## 5. Benchmarking & Optimization

| File | Purpose |
|------|---------|
| [BENCHMARKING.md](./BENCHMARKING.md) | Benchmark infrastructure, A/B testing, usage guide |
| [VARIANT_ARCHITECTURE.md](./VARIANT_ARCHITECTURE.md) | Three-variant prompt optimization system (Dec 2024) |
| [VARIANT_RESULTS.md](./VARIANT_RESULTS.md) | Benchmark results for all 3 prompt variants |

---

## Root-level files

| File | Purpose |
|------|---------|
| [../README.md](../README.md) | Project README (installation, quick start) |
| [../AGENTS.md](../AGENTS.md) | Agent usage guide (Simple + ReAct agents) |
| [../CLAUDE.md](../CLAUDE.md) | Claude Code guidance (architecture, commands, patterns) |
| [../PHASE4_FINAL_COMPLETION_REPORT.md](../PHASE4_FINAL_COMPLETION_REPORT.md) | Phase 4 completion report (v0.0.5) |
| [../PHASE4.6_COMPLETION_REPORT.md](../PHASE4.6_COMPLETION_REPORT.md) | Phase 4.6 flakiness fixes completion report |
