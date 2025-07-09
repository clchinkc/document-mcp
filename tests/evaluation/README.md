# Document-MCP Agent Evaluation Suite

This directory contains the comprehensive evaluation infrastructure for testing document-mcp agents' performance, token usage, and reliability.

## Overview

The evaluation suite implements **Phase 1** of the LLM Test Suite and Prompt Optimization Plan, providing a robust foundation for measuring and improving agent performance.

## Architecture

### Core Components

- **`test_agent_performance.py`** - Main evaluation tests with pytest integration
- **`evaluation_utils.py`** - Utilities for metrics collection and analysis
- **`config.py`** - Configuration and performance thresholds
- **`run_evaluation.py`** - Standalone evaluation runner

### Key Features

- **Agent-Agnostic Testing**: Parameterized tests that work with both Simple and React agents
- **Mock and Real LLM Support**: Fast mock testing for development, real LLM testing for quality assurance
- **Performance Metrics**: Token usage, execution time, and tool call frequency tracking
- **Assertion Strategy**: Focus on structured `details` field and file system state validation
- **Comprehensive Reporting**: Detailed performance reports with comparisons

## Usage

### Running Tests via pytest

```bash
# Run all evaluation tests
pytest tests/evaluation/ -m evaluation

# Run specific test categories
pytest tests/evaluation/ -k "performance"

# Run with verbose output
pytest tests/evaluation/ -v -s
```

### Running Standalone Evaluation Suite

```bash
# Run full evaluation suite with mock LLM
python3 tests/evaluation/run_evaluation.py

# Run specific test categories
python3 tests/evaluation/run_evaluation.py --categories basic intermediate

# Run with real LLM (requires API keys)
python3 tests/evaluation/run_evaluation.py --real-llm

# Save results to file
python3 tests/evaluation/run_evaluation.py --save-results
```

## Test Categories

- **basic**: Simple single-operation tests (document creation, deletion)
- **intermediate**: Multi-step operations (document with chapters)
- **advanced**: Complex analysis and search operations
- **complex**: Multi-step workflows with multiple operations
- **query**: Read-only query operations (list, read, statistics)

## Performance Thresholds

The evaluation suite enforces performance thresholds to ensure agents remain efficient:

### Token Usage Limits
- **Simple Agent**: 150-400 tokens per operation
- **React Agent**: 400-800 tokens per operation

### Execution Time Limits
- **Simple Agent**: 3-8 seconds per operation
- **React Agent**: 10-30 seconds per operation

### Tool Call Limits
- **Most Operations**: 2-7 tool calls maximum

## Evaluation Strategy

### Mock LLM Testing (Fast)
- **Purpose**: Rapid development and CI/CD pipeline
- **Speed**: <1 second per test
- **Cost**: Zero API calls
- **Coverage**: Agent logic and MCP integration

### Real LLM Testing (Quality Assurance)
- **Purpose**: End-to-end quality validation
- **Speed**: 10-60 seconds per test
- **Cost**: Actual API calls
- **Coverage**: Complete system behavior

## Assertion Philosophy

The evaluation suite follows the **"Assert on Reality, Not Words"** principle:

1. **Structured Data Validation**: Assert on the `details` field containing MCP tool responses
2. **File System State**: Verify actual file system changes rather than LLM descriptions
3. **Performance Metrics**: Track concrete measurements (tokens, time, tool calls)
4. **Avoid LLM-Generated Text**: Don't rely on the `summary` field for test validation

## Example Usage

```python
# Import evaluation components
from tests.evaluation import AgentTestRunner, MockLLMResponse

# Create test runner
runner = AgentTestRunner(docs_root, use_mock_llm=True)

# Run a test scenario
metrics = await runner.run_simple_agent_test(
    query="Create a document called 'test'",
    expected_response=MockLLMResponse.create_document_response("test")
)

# Assert on performance and results
assert metrics.success
assert metrics.token_usage < 200
assert metrics.execution_time < 5.0
```

## Performance Reporting

The evaluation suite generates comprehensive performance reports:

```
=== Performance Report: Document Creation Tests ===
Total Tests: 2
Success Rate: 100%
Total Token Usage: 450
Total Execution Time: 3.45s
Average Tokens per Test: 225.0
Average Execution Time: 1.73s
  ✓ Test 1: 150 tokens, 0.12s
  ✓ Test 2: 300 tokens, 3.33s
==================================================
```

## Integration with CI/CD

The evaluation suite is designed for integration with continuous integration:

- **Fast Mock Tests**: Run on every commit for rapid feedback
- **Real LLM Tests**: Run on schedule or release candidates
- **Performance Regression Detection**: Fail builds if performance degrades
- **Metrics Tracking**: Historical performance data collection
