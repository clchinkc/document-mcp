# Phase 4.6: Test Infrastructure & Execution Plan

**Date**: February 26, 2026
**Status**: Planning
**Effort Estimate**: 3-4 days (implementation + validation)

---

## Phase 4.6 Objectives

1. **Fix integration test flakiness** - Resolve subprocess environment variable inheritance issues
2. **Improve test isolation** - Per-test process management and cleanup
3. **Add flaky test detection to CI** - Automated identification and reporting of unstable tests

---

## Executive Summary

**Current State** (from Phase 4 completion):
- ✅ 469/469 unit tests passing (100% pass rate)
- ⚠️ ~39/79 integration/E2E tests flaky (~49% flakiness rate)
- ❌ Codex CLI tooling not yet established
- ❌ Flaky test detection in CI not implemented

**Root Cause**: Subprocess environment variable inheritance - `MCPServerStdio` subprocess does not properly receive parent test environment (`DOCUMENT_ROOT_DIR`, API keys, etc.)

**Non-Blocking**: Flakiness is environmental (test infrastructure), not code-related. All production logic validated by unit tests.

---

## Key Work Items

### 1. Fix Subprocess Environment Variable Inheritance (1-1.5 days)

**Problem**: Tests set `DOCUMENT_ROOT_DIR` but `MCPServerStdio` subprocess doesn't inherit it.

**Solution Options**:
- **A (Recommended)**: Explicitly pass env dict to MCPServerStdio
  ```python
  # Current: MCPServerStdio(command=...) - inherits parent env poorly
  # Fixed: Pass explicit env dict
  import os
  env = os.environ.copy()
  env["DOCUMENT_ROOT_DIR"] = str(temp_docs_root)
  server = MCPServerStdio(
      command=sys.executable,
      args=["-m", "story_mcp.doc_tool_server", "stdio"],
      env=env,  # Explicitly pass environment
      timeout=60.0
  )
  ```

- **B (Alternative)**: Configure via API instead of subprocess
  - Less viable - would require major refactoring

**Implementation**:
- Update `tests/integration/test_agents_stdio.py` - MCP server fixture
- Update `tests/shared/test_environment.py` - EnvironmentManager for subprocess support
- Update all MCPServerStdio instantiations in integration tests

**Tests to Add**:
- `test_mcp_server_inherits_environment_variables()` - Verify env dict is passed
- `test_document_root_dir_respected_in_subprocess()` - End-to-end verification

**Success Criteria**:
- All integration tests pass when run individually AND in suite
- Flakiness rate drops to < 5%

---

### 2. Improve Test Isolation (1-1.5 days)

**Problem**: Tests leave state (open files, processes, env vars) affecting subsequent tests.

**Solution**:
- **Enhance EnvironmentManager** with context managers for subprocess cleanup
- **Add process tracking** to fixture - ensure all MCPServerStdio instances cleaned up
- **Implement proper async cleanup** in pytest fixtures

**Key Changes**:

**`tests/shared/test_environment.py`**:
```python
class SubprocessEnvironment:
    """Manages subprocess creation with proper cleanup."""

    def __init__(self, env_dict: dict[str, str]):
        self.env_dict = env_dict
        self.processes: list = []

    def create_server(self, command, args, timeout):
        """Create MCPServerStdio with tracked cleanup."""
        server = MCPServerStdio(
            command=command,
            args=args,
            env=self.env_dict,
            timeout=timeout
        )
        self.processes.append(server)
        return server

    def cleanup(self):
        """Ensure all processes terminated."""
        for proc in self.processes:
            if proc and hasattr(proc, '_stdin'):
                try:
                    proc._stdin.close()
                except:
                    pass
```

**`conftest.py` fixture enhancement**:
```python
@pytest.fixture
async def mcp_server_managed(temp_docs_root, test_env_manager):
    """MCP server with managed cleanup and environment."""
    env = os.environ.copy()
    env["DOCUMENT_ROOT_DIR"] = str(temp_docs_root)

    server = MCPServerStdio(
        command=sys.executable,
        args=["-m", "story_mcp.doc_tool_server", "stdio"],
        env=env,
        timeout=60.0
    )

    yield server

    # Explicit cleanup
    try:
        if hasattr(server, '_stdin') and server._stdin:
            server._stdin.close()
        if hasattr(server, '_stdout') and server._stdout:
            server._stdout.close()
    except:
        pass
```

**Tests to Add**:
- `test_fixture_cleanup_on_exception()` - Cleanup happens even if test fails
- `test_multiple_servers_isolated()` - Multiple servers don't interfere

---

### 3. Add Flaky Test Detection to CI (0.5-1 days)

**Problem**: Flaky tests pass/fail randomly, hard to detect in single CI run.

**Solution**: Rerun flaky tests in CI with reporting.

**Implementation**:

**`.github/workflows/python-test.yml` changes**:
```yaml
- name: Run integration tests with flaky detection
  if: matrix.group == 'tests'
  run: |
    # Run integration tests 3 times, collect flakiness
    uv run pytest \
      tests/integration/ \
      --tb=short \
      -v \
      --count=3 \
      --max-count=3 \
      -m "not e2e" \
      --junit-xml=test-results.xml

- name: Generate flaky test report
  if: matrix.group == 'tests' && failure()
  run: |
    python3 scripts/development/ci/flaky_test_reporter.py \
      test-results.xml \
      --output flaky-tests.json

    # Fail only if critical tests are flaky (not marked as flaky)
    python3 scripts/development/ci/check_flakiness.py \
      flaky-tests.json \
      --critical-only
```

**New Scripts to Create**:

**`scripts/development/ci/flaky_test_reporter.py`**:
```python
#!/usr/bin/env python3
"""Generate flaky test report from pytest runs."""

import json
import sys
from pathlib import Path
import xml.etree.ElementTree as ET
from collections import defaultdict

def analyze_junit_xml(junit_path):
    """Analyze JUnit XML for flaky tests."""
    tree = ET.parse(junit_path)
    root = tree.getroot()

    flaky_tests = defaultdict(lambda: {"passed": 0, "failed": 0, "errors": []})

    for testcase in root.findall(".//testcase"):
        test_name = f"{testcase.get('classname')}::{testcase.get('name')}"

        if testcase.find("failure") is not None:
            flaky_tests[test_name]["failed"] += 1
            flaky_tests[test_name]["errors"].append(testcase.find("failure").text)
        elif testcase.find("error") is not None:
            flaky_tests[test_name]["failed"] += 1
            flaky_tests[test_name]["errors"].append(testcase.find("error").text)
        else:
            flaky_tests[test_name]["passed"] += 1

    return dict(flaky_tests)

def identify_flaky(results, threshold=0.5):
    """Identify tests flakier than threshold."""
    flaky = {}
    for test_name, stats in results.items():
        total = stats["passed"] + stats["failed"]
        failure_rate = stats["failed"] / total if total > 0 else 0
        if failure_rate > threshold:
            flaky[test_name] = {
                "failure_rate": failure_rate,
                "passed": stats["passed"],
                "failed": stats["failed"],
                "errors": stats["errors"][:1]  # First error only
            }
    return flaky

if __name__ == "__main__":
    junit_path = Path(sys.argv[1])
    results = analyze_junit_xml(junit_path)
    flaky = identify_flaky(results)

    output_path = Path(sys.argv[3])  # --output
    output_path.write_text(json.dumps(flaky, indent=2))
    print(f"Found {len(flaky)} flaky tests")
```

**`scripts/development/ci/check_flakiness.py`**:
```python
#!/usr/bin/env python3
"""Check flakiness and fail if critical tests are flaky."""

import json
import sys
from pathlib import Path

# Tests that are known flaky (acceptable)
KNOWN_FLAKY = {
    "tests/integration/test_agents_mcp_integration.py",
    # ... others marked with @pytest.mark.flaky
}

if __name__ == "__main__":
    flaky_path = Path(sys.argv[1])
    flaky = json.loads(flaky_path.read_text())

    critical_flaky = {
        k: v for k, v in flaky.items()
        if not any(known in k for known in KNOWN_FLAKY)
    }

    if critical_flaky:
        print(f"CRITICAL: {len(critical_flaky)} tests are flaky but not marked as such")
        for test, stats in critical_flaky.items():
            print(f"  - {test}: {stats['failure_rate']*100:.0f}% failure rate")
        sys.exit(1)
    else:
        print(f"✅ All flaky tests are marked with @pytest.mark.flaky")
        sys.exit(0)
```

**Mark Known Flaky Tests** (`tests/integration/test_agents_mcp_integration.py`, etc.):
```python
@pytest.mark.flaky(reruns=3, reruns_delay=2)
class TestAgentsMCPIntegration:
    """Tests known to be flaky due to subprocess environment issues."""
```

---

### 4. Codex CLI Integration (0.5-1 days)

**Note**: No existing Codex CLI tooling found in codebase. This is a placeholder for future integration if needed.

**What This Would Entail** (if/when required):
- Create `scripts/development/codex/` directory structure
- Build Codex-compatible task runner
- Wire test execution through Codex CLI
- Generate Codex-compatible output format

**Current Status**: Not blocking v0.0.5 release. Can be deferred to Phase 5.

---

## Dependencies & Prerequisites

### Required Dependencies
- **pytest-rerunfailures** - For `--reruns` support
  ```bash
  pip install pytest-rerunfailures
  ```
- **pytest-repeat** - For `--count` support (already in dev dependencies)

### Environment Requirements
- Python 3.13 (matches CI)
- All existing dev dependencies from `pyproject.toml`

### System Requirements
- Unix-like environment (tests use subprocess)
- ~50MB free disk space (temp files during tests)

---

## Blockers & Constraints

**None Identified** - Phase 4.6 is a quality improvement with no dependency on other work.

**Risks**:
- **Test environment complexity** - Subprocess environment handling is fragile
- **Cross-platform compatibility** - Process management differs on Windows (low risk - Linux/macOS only for now)
- **CI resource constraints** - Running tests 3x increases CI time by ~3 minutes

---

## Success Metrics

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Integration test pass rate (single run) | ~51% | ≥95% | 🎯 |
| Flaky test suite (5 consecutive runs) | ~49% fail rate | <5% | 🎯 |
| CI time for core tests | ~90s | <100s | ✅ |
| Flaky test detection in CI | ❌ None | ✅ Automated | 🎯 |
| Code coverage (unit tests) | 85%+ | ≥85% | ✅ |
| Test isolation score | Poor | Excellent | 🎯 |

---

## Implementation Timeline

### Day 1: Fix Environment Variable Inheritance (1-1.5 days)
- [ ] Update `MCPServerStdio` fixture to explicitly pass env dict
- [ ] Enhance `EnvironmentManager` for subprocess support
- [ ] Add 2-3 verification tests
- [ ] Run integration test suite - verify pass rate improves

### Day 1-2: Improve Test Isolation (1-1.5 days)
- [ ] Add subprocess cleanup to fixtures
- [ ] Add process tracking to test environment
- [ ] Implement proper async cleanup
- [ ] Add isolation verification tests

### Day 2-3: Add Flaky Test Detection (0.5-1 days)
- [ ] Create `scripts/development/ci/flaky_test_reporter.py`
- [ ] Create `scripts/development/ci/check_flakiness.py`
- [ ] Update CI workflow to run integration tests 3x
- [ ] Mark known-flaky tests with `@pytest.mark.flaky()`
- [ ] Verify CI detects and reports flakiness

### Day 3-4: Validation & Documentation (0.5-1 days)
- [ ] Run full test suite 5 consecutive times
- [ ] Verify flakiness < 5%
- [ ] Update CLAUDE.md with CI improvements
- [ ] Create Phase 4.6 completion report

---

## Files to Create/Modify

### Create New Files
```
scripts/development/ci/
├── flaky_test_reporter.py      # Analyze flaky tests from XML
├── check_flakiness.py           # Enforce flakiness policy
└── __init__.py

tests/shared/
└── (enhancement to test_environment.py)
```

### Modify Files
```
tests/conftest.py                 # Enhanced MCP server fixture
tests/shared/test_environment.py   # Add subprocess support
.github/workflows/python-test.yml  # Add flaky test detection
tests/integration/test_agents_stdio.py  # Fix env inheritance
tests/integration/test_agents_mcp_integration.py  # Mark as flaky
tests/integration/pytest_markers.py  # Update markers
CLAUDE.md                          # Document improvements
```

---

## Verification Checklist

- [ ] All 469 unit tests pass
- [ ] Integration test suite passes 5 consecutive times
- [ ] Flakiness rate < 5%
- [ ] CI workflow detects and reports flaky tests
- [ ] Flaky tests marked with `@pytest.mark.flaky()` pass when rerun
- [ ] Documentation updated in CLAUDE.md
- [ ] Phase 4.6 completion report created
- [ ] Git commit with all changes

---

## Deferred to Phase 5+

- **Codex CLI integration** - No existing tooling, deferred until required
- **Cross-platform test support** - Currently Unix-only, can expand later
- **E2E test reliability** - Depends on external API stability (separate from Phase 4.6)
- **Performance test infrastructure** - Can be added to `tests/evaluation/`

---

## References

- **Phase 4 Flakiness Analysis**: `PHASE4_FLAKINESS_ANALYSIS.md`
- **Current CI Configuration**: `.github/workflows/python-test.yml`
- **Test Environment**: `tests/shared/test_environment.py`
- **Integration Tests**: `tests/integration/test_agents_stdio.py`

---

## Next Steps

1. **Review this plan** with team
2. **Assign execution** (if applicable)
3. **Create branch** from `main` for Phase 4.6 work
4. **Execute** in order: Environment Vars → Isolation → Flaky Detection → Validation
5. **Create PR** with all changes and Phase 4.6 completion report
6. **Merge** to `main` for v0.0.6 release (or earlier if blocking)

---

**Status**: 🎯 Ready for execution
