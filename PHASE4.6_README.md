# Phase 4.6 Planning Complete

**Date**: February 26, 2026
**Status**: 🎯 Ready for Execution
**Created By**: Claude Code Assessment Agent

---

## Quick Start

### For Executives
📄 **Read**: `PHASE4.6_QUICK_PLAN.txt` (2-minute overview)

### For Project Managers
📊 **Read**: `PHASE4.6_ASSESSMENT.md` (blockers, risks, timeline)

### For Developers
🛠️ **Read**: `PHASE4.6_EXECUTION_PLAN.md` (detailed implementation guide)

---

## Phase 4.6 Overview

**Objective**: Fix integration test flakiness and add automated flaky test detection to CI

**Duration**: 3-4 days
**Effort**: 1 sprint
**Risk**: Low
**Blockers**: None

### Key Improvements

| What | Current | After Phase 4.6 |
|------|---------|-----------------|
| Integration test pass rate | ~51% | ≥95% |
| Flaky test detection | ❌ Manual | ✅ Automated |
| Test isolation | Poor | Excellent |
| CI flakiness reporting | None | Full automation |

---

## Document Guide

### 1. PHASE4.6_QUICK_PLAN.txt (5 min read)
**For**: Executives, stakeholders, quick briefing
**Contains**:
- 2-3 line objective
- 3 key work items (numbered)
- Dependencies and blockers
- Success metrics
- Implementation timeline
- Next steps

**Use when**: You need a quick briefing or need to present to stakeholders

---

### 2. PHASE4.6_ASSESSMENT.md (15 min read)
**For**: Project managers, decision makers
**Contains**:
- Pre-execution assessment
- Codex CLI tooling assessment (not found, deferred)
- Blocker analysis (none critical)
- Test infrastructure current state
- 11 detailed assessment sections
- Risk matrix
- Production impact analysis
- Final recommendation

**Use when**:
- Making execution decisions
- Understanding risks and dependencies
- Validating feasibility
- Stakeholder reporting

---

### 3. PHASE4.6_EXECUTION_PLAN.md (30 min read)
**For**: Developers, technical leads
**Contains**:
- Executive summary
- 4 detailed work items with code examples
- Dependencies and prerequisites
- Success metrics
- Day-by-day implementation timeline
- Specific files to create/modify
- Verification checklist
- References

**Use when**:
- Starting development
- Understanding technical details
- Getting code examples
- Planning sprint tasks
- Allocating developer resources

---

## What This Phase Addresses

### Root Cause: Subprocess Environment Variable Inheritance

**The Problem**:
```python
# Integration tests set environment:
os.environ["DOCUMENT_ROOT_DIR"] = str(temp_docs_root)

# But MCPServerStdio subprocess doesn't inherit it:
server = MCPServerStdio(
    command=sys.executable,
    args=["-m", "story_mcp.doc_tool_server", "stdio"]
    # ❌ Missing: env=os.environ.copy()
)

# Result: Tools create files in wrong location, tests fail ~49% of the time
```

**The Solution**:
```python
# Explicitly pass environment to subprocess:
env = os.environ.copy()
env["DOCUMENT_ROOT_DIR"] = str(temp_docs_root)

server = MCPServerStdio(
    command=sys.executable,
    args=["-m", "story_mcp.doc_tool_server", "stdio"],
    env=env  # ✅ Explicitly pass environment
)
```

---

## Key Facts

### Current State (Phase 4 Complete)
- ✅ 469/469 unit tests passing (100%)
- ⚠️ ~39/79 integration/E2E tests flaky (~49%)
- ❌ Codex CLI tooling not in codebase
- ❌ No automated flaky test detection in CI

### After Phase 4.6
- ✅ 469/469 unit tests passing (maintained)
- ✅ ≥95% integration test pass rate
- ✅ <5% flakiness rate in 5-run suites
- ✅ Automated flaky test detection in CI
- ✅ Better test isolation
- ℹ️ Codex CLI deferred to Phase 5

### Why Phase 4.6 Doesn't Block v0.0.5 Release
1. Integration flakiness is **pre-existing** (noted in Phase 4 analysis)
2. All **production code** validated by unit tests
3. Flakiness is **test infrastructure**, not code defect
4. **v0.0.5 is production-ready** regardless

---

## Success Criteria

### Must Pass
- ✅ All 469 unit tests pass
- ✅ Integration test suite passes 5 consecutive times
- ✅ Flakiness rate < 5%
- ✅ CI detects and reports flaky tests

### Should Achieve
- ✅ Documentation updated
- ✅ Phase 4.6 completion report created
- ✅ Git commit with all changes

### Not in Scope (Deferred to Phase 5)
- Codex CLI integration
- Cross-platform (Windows) test support
- E2E test reliability (external API stability)

---

## Effort Breakdown

| Task | Days | Complexity | Confidence |
|------|------|-----------|-----------|
| Fix env var inheritance | 1-1.5 | Easy | 95% |
| Improve test isolation | 1-1.5 | Moderate | 90% |
| Add flaky detection to CI | 0.5-1 | Moderate | 95% |
| Validation & docs | 0.5-1 | Easy | 90% |
| **TOTAL** | **3-4** | **Low-Moderate** | **92%** |

---

## New Files to Create

```
scripts/development/ci/
├── flaky_test_reporter.py    # Parse XML, identify flaky tests
├── check_flakiness.py         # Enforce flakiness policy in CI
└── __init__.py
```

## Files to Modify

```
tests/conftest.py                      # Enhanced MCP server fixture
tests/shared/test_environment.py       # Add subprocess support class
tests/integration/test_agents_stdio.py # Fix env inheritance
.github/workflows/python-test.yml      # Add flaky test 3x runs
CLAUDE.md                              # Document improvements
```

---

## Risks & Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|-----------|
| Subprocess env fragility | Low | Medium | Explicit env dict passing |
| Fixture cleanup hangs | Very Low | High | Async best practices, pytest-timeout |
| False positives in detection | Medium | Low | Tunable threshold, whitelist |
| CI time increase | Low | Low | Parallelization in Phase 5 |

**Overall Risk Level**: 🟢 Low

---

## Dependencies

### New Python Package
```bash
pip install pytest-rerunfailures
```

### Existing Dependencies
- All available in current `pyproject.toml`
- No breaking version requirements
- Compatible with Python 3.13

### System Requirements
- Linux/macOS (primary)
- ~50MB disk space
- No external APIs required

---

## Timeline

```
Day 1 (1-1.5 days)
├─ Fix environment variable inheritance
└─ Run integration tests - validate improvement

Day 1-2 (1-1.5 days)
├─ Improve test isolation
└─ Add subprocess cleanup tracking

Day 2-3 (0.5-1 days)
├─ Create flaky_test_reporter.py
├─ Create check_flakiness.py
└─ Update CI workflow

Day 3-4 (0.5-1 days)
├─ Run tests 5 consecutive times
├─ Verify < 5% flakiness
├─ Update documentation
└─ Create completion report
```

---

## Next Steps

### Before Starting
1. Review `PHASE4.6_EXECUTION_PLAN.md`
2. Check dependencies (`pytest-rerunfailures`)
3. Understand root causes in Phase 4 flakiness analysis

### During Execution
1. Create branch: `git checkout -b phase-4.6-test-infrastructure`
2. Execute work items in order (environment → isolation → detection)
3. Commit frequently with clear messages
4. Test locally before pushing to CI

### After Completion
1. Run full test suite 5 consecutive times
2. Verify metrics meet targets
3. Create Phase 4.6 completion report
4. Create PR with all changes
5. Merge to main after review
6. Tag v0.0.6 with improvements noted

---

## Related Documents

### Phase 4 Context
- **PHASE4_FLAKINESS_ANALYSIS.md** - Root cause analysis
- **PHASE4_FINAL_COMPLETION_REPORT.md** - Phase 4 deliverables
- **PHASE4_SESSION_SUMMARY.md** - Session notes

### Architecture
- **CLAUDE.md** - Project instructions (update with Phase 4.6 improvements)
- **docs/MCP_DESIGN_PATTERNS.md** - MCP integration patterns
- **README.md** - Project overview

### Current CI/CD
- **.github/workflows/python-test.yml** - CI configuration
- **pytest.ini** - Pytest configuration
- **pyproject.toml** - Dependencies and build config

---

## Key Contacts

| Role | Contact | Area |
|------|---------|------|
| Project Lead | - | Overall coordination |
| Test Infrastructure | - | Phase 4.6 execution |
| CI/CD | - | Workflow updates |
| Release Manager | - | v0.0.6 deployment |

---

## Approval Status

- [x] Assessment complete
- [x] Execution plan documented
- [x] Blockers identified (none critical)
- [x] Timeline validated
- [x] Risk assessment complete
- [ ] Approval to proceed (awaiting)

---

## Version History

| Date | Version | Status | Changes |
|------|---------|--------|---------|
| 2026-02-26 | 1.0 | Final | Initial assessment complete |

---

## Questions?

Refer to the appropriate document:
- **"Is this blocking v0.0.5?"** → See PHASE4.6_ASSESSMENT.md section 10
- **"How long will this take?"** → See PHASE4.6_QUICK_PLAN.txt (timeline)
- **"What's the root cause?"** → See PHASE4.6_EXECUTION_PLAN.md (1.0)
- **"How do I implement this?"** → See PHASE4.6_EXECUTION_PLAN.md (detailed)

---

**Status**: 🎯 READY FOR EXECUTION
**Confidence**: 95% (well-researched, clear scope)
**Next Action**: Review execution plan and begin implementation
