# Testing Guidelines for Document MCP

## 1. Overview

This document outlines the testing strategy, structure, and best practices for the Document MCP project. A robust testing culture is crucial for maintaining code quality, preventing regressions, and enabling confident refactoring. Our test suite is organized into four main categories—Unit, Integration, End-to-End (E2E), Evaluation, and Metrics—each serving a distinct purpose.

### Current Status (v0.0.3)
- **Total Tests**: 554 tests with 100% pass rate
- **Coverage**: Comprehensive 4-tier testing strategy (61% code coverage)
- **Features**: Full pagination system validation, GCP observability tests, tool selection benchmarking, A/B description testing, multi-model validation, and enhanced E2E reliability

**Special focus on Safety Features**: The testing framework includes comprehensive validation of the write-safety system, automatic micro-snapshots, version control operations, pagination system, and modification history tracking to ensure zero content loss.

## 2. Test Structure

The testing framework is organized by test type, ensuring a clear separation of concerns.

```
tests/
├── unit/         # Isolated function/class tests (341 tests)
├── integration/  # Component interaction tests (168 tests)
├── e2e/          # Full system tests (6 tests)
├── evaluation/   # Performance benchmarking + A/B testing + multi-model (33 tests)
├── test_metrics.py # Metrics validation (6 tests)
├── shared/       # Shared utilities, base classes, and fixtures
└── conftest.py   # Global pytest configuration and fixtures
```

## 3. Test Categories

### 3.1. Unit Tests (`tests/unit/`)
- **Purpose**: To test individual functions, classes, and components in complete isolation from external systems.
- **Characteristics**:
  - **No external dependencies**: Strictly no filesystem, network, or API calls. All external interactions must be mocked.
  - **Speed**: Must be extremely fast.
- **Example**: `test_atomic_paragraph_tools.py` tests individual paragraph manipulation functions.

### 3.2. Integration Tests (`tests/integration/`)
- **Purpose**: To test the interaction between different components of the system, particularly the agent's communication with the MCP server.
- **Characteristics**:
  - **Real MCP Server**: Uses a live `MCPServerStdio` instance or direct tool calls to test the full communication protocol.
  - **Mocked LLMs**: LLM calls are mocked to ensure tests are fast, deterministic, and do not incur API costs.
  - **Filesystem**: Use the `temp_docs_root` fixture for an isolated temporary directory.
  - **Safety Feature Testing**: `test_safety_features.py` and `test_automatic_snapshot_system.py` provide comprehensive validation of the write-safety system, version control, and snapshotting.
- **Example**: `test_doc_tool_server.py` verifies the behavior of all tools by calling them directly.

### 3.3. End-to-End (E2E) Tests (`tests/e2e/`)
- **Purpose**: To test the entire application workflow with real AI models, verifying that the agent's reasoning and tool usage are correct.
- **Characteristics**:
  - **Real API Keys**: Require real API keys (e.g., `OPENAI_API_KEY`) to be set in the environment.
  - **Assert on Reality**: Tests use the `DocumentSystemValidator` (`tests/e2e/validation_utils.py`) to assert on the actual file system state, not on the LLM's natural language response. This makes tests more robust.
  - **Conditional Execution**: Tests are automatically skipped via `@pytest.mark.skipif` if API keys are not available.
- **Example**: `test_agents_e2e.py` runs full user queries and validates the results on the filesystem.

### 3.4. Evaluation Tests (`tests/evaluation/`)
- **Purpose**: To benchmark agent performance on standardized tasks and assess the quality of responses.
- **Architecture**: Follows a **"Clean Architecture"** philosophy where agents collect their own performance metrics (time, tokens), and the test layer *optionally* enhances these with a simple, LLM-based quality score.
- **Usage**: See the [Evaluation README](./evaluation/README.md) for a detailed explanation.

## 4. Key Architectural Patterns

### 4.1. Stateful, Class-Based Tests
For some **Integration** and **E2E** tests, we use a stateful, class-based approach to manage the lifecycle of the agent and MCP server. This prevents `asyncio` event loop errors and ensures a clean state for each test class.

**Key Components**:
- **Base Classes**: Tests may inherit from base classes that provide setup and teardown logic.
- **Class-level Fixture**: A `pytest.fixture` with `scope="class"` and `autouse=True` can be used to initialize the agent and server once for all tests in that class.

```python
# Example of a stateful, class-based test structure
@pytest.mark.integration
class TestSomethingWithState:
    @pytest.fixture(scope="class", autouse=True)
    async def setup(self, request):
        # Code to set up a server or agent once for the class
        self.my_server = await setup_server()
        request.cls.server = self.my_server
        yield
        await self.my_server.close()

    @pytest.mark.asyncio
    async def test_some_workflow(self):
        # The server is already initialized and available via self.server
        # or self.__class__.server
        response = await self.server.do_something()
        assert response.is_ok
```

### 4.2. Centralized Tool Imports
To ensure stability during refactoring, all tests should import MCP tools from **`tests/tool_imports.py`**. This module provides a consistent interface to the tool functions, regardless of their location in the source code.

### 4.3. Shared Helpers and Fixtures
- **`tests/shared/`**: This directory is critical for reducing code duplication. It contains shared assertion helpers, environment setup functions, and fixtures.
- **`conftest.py`**: Defines project-wide fixtures, such as `temp_docs_root` (which provides a clean temporary directory) and `skip_if_no_api_key`.

## 5. Best Practices

- **Test Names**: Should be descriptive and follow the `test_<area>_<scenario>` pattern (e.g., `test_e2e_react_agent_document_and_chapter_workflow`).
- **Docstrings**: Should be used to explain the *purpose* of the test, not just repeat the test name.
- **Isolation**: Unit tests must be fully isolated. Integration/E2E tests should use fixtures (`temp_docs_root`) for all setup and teardown logic to ensure a clean environment.
- **Mocking**: Use the `mocker` fixture from `pytest-mock`. Mock at the boundaries of the system.

## 6. Running Tests

### Modern Toolchain (Recommended)

```bash
# Run all tests with uv (10-100x faster dependency resolution)
uv run python -m pytest

# Run specific categories
uv run python -m pytest tests/unit/          # Unit tests only
uv run python -m pytest tests/integration/   # Integration tests only
uv run python -m pytest tests/e2e/           # E2E tests only
uv run python -m pytest tests/evaluation/    # Evaluation tests only

# Run with coverage
uv run python -m pytest --cov=document_mcp --cov-report=html

# Code quality checks
uv run ruff check                             # Lint code
uv run ruff check --fix                       # Auto-fix issues
uv run ruff format                           # Format code
uv run mypy document_mcp/                     # Type checking

# Quality checks script
python3 scripts/quality.py full
```

### Traditional Python

```bash
# Run all tests
python3 -m pytest

# Run specific categories
python3 -m pytest tests/unit/
python3 -m pytest tests/integration/
python3 -m pytest tests/e2e/

# Run with coverage
python3 -m pytest --cov=document_mcp --cov-report=html
```

### Running E2E Tests
To run the E2E tests, you must first export your API key:
```bash
export OPENAI_API_KEY="your-key-here"
# or
export GEMINI_API_KEY="your-key-here"

# Run E2E tests with extended timeout
uv run python -m pytest tests/e2e/ --timeout=600
```

### CI/CD Testing (GitHub Actions)
The GitHub workflow runs:
```bash
# Unit + Integration + E2E tests (excludes evaluation for CI efficiency)
uv run pytest tests/unit/ tests/integration/ tests/e2e/ --timeout=600
```

### Development Infrastructure Testing
Additional testing tools for development infrastructure:
```bash
# Test production metrics system
python3 scripts/development/metrics/test_production.py

# Test development telemetry infrastructure
python3 scripts/development/telemetry/scripts/test.py

# Start development telemetry for Grafana Cloud testing
scripts/development/telemetry/scripts/start.sh
```

These tools test the **automatic metrics collection system** that provides anonymous usage analytics to help improve Document MCP for all users.

## 7. Adding New Tests

1.  **Determine the Type**:
    - **Unit**: A single, isolated function?
    - **Integration**: Interaction between agent and MCP?
    - **E2E**: A full user workflow that depends on AI reasoning?
2.  **Import Tools**: Import all MCP tools from `tests.tool_imports`.
3.  **Use Fixtures**: Use fixtures like `document_factory` and `temp_docs_root` for setup.
4.  **Follow the Pattern**: Mimic the structure of existing tests in the same category.
5.  **Write Clear Assertions**: Verify the specific outcomes of your test. For E2E tests, this means checking the filesystem for created documents or verifying specific content was written.
