# Document-MCP Agent Evaluation Suite

This directory contains the comprehensive evaluation infrastructure for testing agent performance, token usage, and reliability. It follows a **clean architecture** philosophy where agent logic is decoupled from evaluation logic.

## 1. Overview

The evaluation suite provides a robust foundation for measuring and improving agent performance. The core principle is:

**Agents collect their own performance metrics. The test layer optionally enhances them with LLM-based quality evaluation.**

This approach keeps the agent implementation clean and focused on its primary tasks, while providing a flexible way to assess the quality of its responses during testing.

## 2. Clean Architecture

The evaluation process is best demonstrated in `test_simple_integration.py` and follows these steps:

1.  **Agent Execution**: An agent is executed to perform a task (e.g., via `process_single_user_query_with_metrics`). It returns its standard response and a `AgentPerformanceMetrics` object containing metrics like execution time, token usage, and success status.
2.  **Test-Layer Enhancement**: In the test, the `enhance_test_metrics` function from `llm_evaluation_layer.py` is called. This function takes the performance metrics and the agent's query/response.
3.  **Optional LLM Evaluation**: If the `ENABLE_LLM_EVALUATION` environment variable is set to `true`, `enhance_test_metrics` uses a simple, cost-effective LLM (like `gemini-2.5-flash`) to score the response quality and provide brief feedback. This evaluation is designed to be fast and non-blocking.
4.  **Combined Assertions**: The test can then assert on both the concrete performance metrics (e.g., `execution_time < 5.0`) and the qualitative score from the LLM evaluation (e.g., `quality_score > 0.7`).

### Example: `test_simple_integration.py`

```python
# 1. Run agent normally to get standard performance metrics
response, performance_metrics = await process_single_user_query_with_metrics(agent, query)

# 2. Standard assertions on performance are always possible
assert performance_metrics.execution_time > 0

# 3. Enhance with optional LLM evaluation for testing
enhanced_metrics = await enhance_test_metrics(
    performance_metrics, query, response.summary
)

# 4. Assert on the optional quality score
if enhanced_metrics.llm_evaluation and enhanced_metrics.llm_evaluation.success:
    assert enhanced_metrics.llm_evaluation.score >= 0.7
```

## 3. Core Components

-   **`test_simple_integration.py`**: The primary example of the clean evaluation architecture. It shows how to run an agent and optionally enhance its metrics for a combined quality and performance assessment.
-   **`llm_evaluation_layer.py`**: Provides the `enhance_test_metrics` function. This is the heart of the optional, test-layer-only LLM evaluation. It is designed to be simple and robust, gracefully handling failures or timeouts.
-   **`config.py`**: Defines performance thresholds (`PerformanceThresholds`) and standardized test scenarios (`DEFAULT_TEST_SCENARIOS`) used across evaluation tests.
-   **`evaluation_utils.py`**: Contains helper classes and functions for performance tracking (`PerformanceTracker`), specialized assertions (`EvaluationAssertions`), and reporting.
-   **`run_evaluation.py`**: A standalone script for running the evaluation suite across multiple scenarios and agents.

## 4. Usage

### Running via Pytest (Modern Toolchain)

The simplest way to run evaluation tests is with pytest using the modern toolchain.

```bash
# Run all evaluation tests with uv (LLM scoring disabled by default)
uv run python -m pytest tests/evaluation/

# Run tests with LLM-based quality scoring enabled
ENABLE_LLM_EVALUATION=true uv run python -m pytest tests/evaluation/

# Run with extended timeout for real API calls
uv run python -m pytest tests/evaluation/ --timeout=600

# Traditional Python (alternative)
python3 -m pytest tests/evaluation/
ENABLE_LLM_EVALUATION=true python3 -m pytest tests/evaluation/
```

### Standalone Runner

The `run_evaluation.py` script provides a way to run the suite for multiple agents and scenarios, generating a comparative report.

```bash
# Run the full evaluation suite with uv
uv run python tests/evaluation/run_evaluation.py

# Run only specific categories
uv run python tests/evaluation/run_evaluation.py --categories basic intermediate

# Run with a real LLM for the agent's actions (E2E mode)
uv run python tests/evaluation/run_evaluation.py --real-llm

# Traditional Python (alternative)
python3 tests/evaluation/run_evaluation.py
```

## 5. Test Scenarios and Thresholds

The `config.py` file defines standardized test scenarios and performance thresholds.

### Test Categories
- **basic**: Simple single-operation tests (document creation, deletion).
- **intermediate**: Multi-step operations (document with chapters).
- **advanced**: Complex analysis and search operations.
- **complex**: Multi-step workflows with multiple operations.
- **query**: Read-only query operations (list, read, statistics).
- **error**: Tests for error handling and edge cases.

### Performance Thresholds
The configuration defines agent-specific limits to prevent performance regressions:
- **Token Usage**: e.g., Simple Agent: 150-400 tokens; React Agent: 400-800 tokens.
- **Execution Time**: e.g., Simple Agent: 3-8s; React Agent: 10-30s.
- **Tool Calls**: e.g., 2-7 tool calls per operation.

## 6. LLM Evaluation Configuration

The optional LLM quality evaluation is designed to be robust and cost-effective.

-   **Control**: Enabled via `export ENABLE_LLM_EVALUATION=true`.
-   **Cost-Effective**: Uses `gemini-2.5-flash` by default.
-   **Robust**: A 10-second timeout and graceful failure handling ensure that evaluation issues never break a test run.
-   **Integration**: Works seamlessly with CI/CD. Fast performance tests can run on every commit, with optional, scheduled runs for LLM-enhanced quality checks.
