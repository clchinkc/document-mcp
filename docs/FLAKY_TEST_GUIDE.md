# Flaky Test Detection Guide

This document explains how the flaky test detection system works and how to use it for local development and CI/CD.

## Overview

Flaky tests are tests that fail intermittently—sometimes passing and sometimes failing on the same code. These are problematic because they erode confidence in the test suite and make it hard to identify real issues.

The document-mcp project includes a two-part flaky test detection system:

1. **CI/CD Workflow** (`.github/workflows/flaky-test-detection.yml`) - Automatically runs on every push/PR
2. **Local Development Tool** (`scripts/development/flaky_test_detector.py`) - Run manually during development

## How It Works

### Classification

Tests are classified into three categories after multiple runs:

- **Stable (Passed)**: Passes in all runs
- **Stable (Failed)**: Fails in all runs
- **Flaky**: Sometimes passes, sometimes fails

### Flakiness Metric

Failure rate = (Failed Runs / Total Runs) × 100%

For example, in 5 test runs:
- Test passes 4 times, fails 1 time = 20% failure rate = FLAKY
- Test passes 5 times = 0% failure rate = STABLE
- Test fails 5 times = 100% failure rate = STABLE (consistently failing)

## Using the CI/CD Workflow

### Automatic Detection

The flaky test detection workflow runs automatically on:
- Every push to `main` or `develop` branches
- Every pull request
- Nightly at 2 AM UTC (catches time-based flakiness)

### Viewing Results

#### In GitHub Actions

1. Go to your repository
2. Click "Actions" tab
3. Select "Flaky Test Detection" workflow
4. Click the specific run

The workflow generates:
- **Test run logs** (5 individual test executions)
- **JSON report** - Machine-readable results for integration
- **Markdown report** - Human-readable summary

#### In Pull Requests

The workflow automatically comments on PRs with the flaky test report:

```
## 🔍 Flaky Test Detection Report

**Commit**: `abc1234`

**Test Runs**: 5

## Summary

- **Total Unique Tests**: 42
- **Stable (Passed)**: 41
- **Stable (Failed)**: 0
- **Flaky Tests**: 1

## Flaky Tests Detected

| Test Name | Failure Rate | Passed/Failed | Pattern |
|-----------|-------------|---------------|---------|
| `tests/integration/test_file.py::test_async_operation` | 40.0% | 3/2 | ✓✓✗✓✗ |
```

### Download Artifacts

1. Click "Summary" on the workflow run page
2. Scroll to "Artifacts" section
3. Download `flaky-test-report` for detailed analysis

## Using the Local Development Tool

### Running the Tool

```bash
# Basic usage (runs integration tests 5 times)
python3 scripts/development/flaky_test_detector.py

# Customize number of runs
python3 scripts/development/flaky_test_detector.py --runs 10

# Test different test paths
python3 scripts/development/flaky_test_detector.py --test-path tests/unit/

# Save results to JSON
python3 scripts/development/flaky_test_detector.py --save-json flaky_report.json
```

### Example Output

```
================================================================================
FLAKY TEST DETECTION REPORT
================================================================================

Total Runs: 5
Unique Tests: 42
  ✓ Stable (Passed): 41
  ✗ Stable (Failed): 0
  ⚠ Flaky: 1

FLAKY TESTS DETECTED:
--------------------------------------------------------------------------------
Test Name                                                         Rate     Pattern
--------------------------------------------------------------------------------
tests/integration/test_doc_tool_server.py::test_concurrent_oper   40.0%    ✓✓✗✓✗

Recommendations:
  1. Review test setup and teardown for race conditions
  2. Check for dependency on external services/APIs
  3. Look for non-deterministic behavior
  4. Check system resource availability
  5. Review test isolation and cleanup

================================================================================
```

## Diagnosing Flaky Tests

### Common Causes

1. **Race Conditions**
   - Concurrent test execution
   - Shared state not properly isolated
   - Timing-dependent operations

2. **External Dependencies**
   - API calls with inconsistent responses
   - File system operations with race conditions
   - Network timeouts

3. **Non-Deterministic Behavior**
   - Random value generation without seeding
   - Dictionary iteration order (pre-Python 3.7)
   - Floating-point comparisons

4. **Resource Constraints**
   - Insufficient memory
   - File descriptor limits
   - Database connection pool exhaustion

5. **Test Isolation Issues**
   - Global state not reset between tests
   - Fixtures with side effects
   - Database transactions not rolled back

### Investigation Steps

1. **Reproduce Locally**
   ```bash
   # Run the flaky test multiple times
   for i in {1..10}; do
     python3 scripts/development/flaky_test_detector.py --test-path tests/integration/test_file.py::test_name
   done
   ```

2. **Add Detailed Logging**
   - Enhance test logging to capture timing and state
   - Log setup/teardown execution
   - Track resource usage

3. **Isolate the Issue**
   - Run test in isolation vs. with others
   - Disable concurrency (pytest-xdist)
   - Check for dependency on test execution order

4. **Review Test Code**
   - Check for `time.sleep()` or timing-dependent logic
   - Verify fixture cleanup
   - Look for shared mutable state

## Fixing Flaky Tests

### Common Fixes

**Race Condition**:
```python
# Bad: Timing-dependent
time.sleep(0.1)
assert file_exists(path)

# Good: Wait for condition
assert wait_for(lambda: file_exists(path), timeout=5)
```

**Shared State**:
```python
# Bad: Global state
_global_config = {}

# Good: Use fixture-scoped state
@pytest.fixture
def config():
    return {}
```

**Non-Deterministic Behavior**:
```python
# Bad: Non-deterministic random
random_value = random.random()

# Good: Seed random for reproducibility
@pytest.fixture(autouse=True)
def seed_random():
    random.seed(42)
    yield
    random.seed()
```

**External Dependencies**:
```python
# Bad: Real API call
response = requests.get("https://api.example.com/data")

# Good: Mock external calls
@mock.patch('requests.get')
def test_something(mock_get):
    mock_get.return_value.json.return_value = {"status": "ok"}
```

## Best Practices

1. **Use Proper Timeouts**
   - Always set timeouts on external operations
   - Use `pytest-timeout` plugin for test timeouts

2. **Seed Random Values**
   - Seed `random`, `numpy.random` in tests
   - Use `faker` with fixed seeds for deterministic data

3. **Mock External Services**
   - Mock API calls, file operations, database
   - Use `responses`, `vcrpy`, or `pytest-mock`

4. **Isolate Tests**
   - Use fresh fixtures for each test
   - Clean up global state in teardown
   - Use temporary directories for file operations

5. **Test Isolation Plugins**
   - `pytest-xdist` for parallelization awareness
   - `pytest-timeout` for hung test detection
   - `pytest-randomly` to randomize test order

## CI/CD Integration

### Workflow Configuration

The flaky test detection workflow is configured to:

- Run 5 times by default (configurable in workflow)
- Run on schedule (nightly at 2 AM UTC)
- Post results to PR comments automatically
- Upload artifacts for 30 days
- Fail the workflow if flaky tests are detected

### Customizing the Workflow

Edit `.github/workflows/flaky-test-detection.yml`:

```yaml
# Change number of runs
RUNS=10  # Default is 5

# Change test path
TEST_PATH=tests/integration/  # Default path

# Change artifact retention
retention-days: 60  # Default is 30
```

## Troubleshooting

### Workflow Not Running

- Check branch protection rules
- Verify workflow file syntax: `uv run yamllint .github/workflows/flaky-test-detection.yml`
- Check Actions permissions in repository settings

### No Flaky Tests Detected

This is expected for a stable test suite! The workflow still:
- Validates test infrastructure
- Provides confidence in test reliability
- Catches intermittent issues early

### High Failure Rate in Report

1. Check if infrastructure is under load
2. Verify API keys/secrets are available
3. Check test environment configuration
4. Look for environmental factors (network, disk space)

## Metrics and Monitoring

### Key Metrics

The workflow tracks:
- Number of flaky tests per run
- Failure rate distribution
- Stable (passed) vs. stable (failed) test counts
- Test execution patterns

### Monitoring in Production

To monitor flaky tests in production:

1. Check PR comment history for patterns
2. Download artifacts for trend analysis
3. Set up GitHub alerts for failed workflows
4. Review nightly runs for environment-specific issues

## See Also

- [Testing Strategy](./TESTING_STRATEGY.md) - Comprehensive testing guide
- [Benchmarks Documentation](./BENCHMARKS.md) - Performance testing
- [pytest Documentation](https://docs.pytest.org/)
- [pytest-timeout Plugin](https://pytest-timeout.readthedocs.io/)
