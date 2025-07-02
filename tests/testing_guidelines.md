# Testing Guidelines for Document MCP

## 1. Overview

This document outlines the testing strategy, structure, and best practices for the Document MCP project. A robust testing culture is crucial for maintaining code quality, preventing regressions, and enabling confident refactoring. Our test suite is organized into three main categories—Unit, Integration, and End-to-End (E2E)—each serving a distinct purpose.

## 2. Test Structure

The testing framework is organized by test type, ensuring a clear separation of concerns.

```
tests/
├── unit/         # Isolated function/class tests (no I/O)
├── integration/  # Component interaction tests (mocked LLM, real MCP)
├── e2e/          # Full system tests (real LLM, real MCP)
├── shared/       # Shared utilities, base classes, and helpers
├── fixtures/     # Complex test data setup and fixtures
└── conftest.py   # Global pytest configuration and fixtures
```

## 3. Test Categories

### 3.1. Unit Tests (`tests/unit/`)
- **Purpose**: To test individual functions, classes, and components in complete isolation from external systems.
- **Characteristics**:
  - **No external dependencies**: Strictly no filesystem, network, or API calls. All external interactions must be mocked.
  - **Speed**: Must be extremely fast (typically <50ms per test).
  - **Fixtures**: Use `mocker` for patching and `tmp_path` for any required in-memory filesystem operations.
- **Example**: Testing a validation function like `validate_document_name()` with various string inputs.

### 3.2. Integration Tests (`tests/integration/`)
- **Purpose**: To test the interaction between different components of the system, particularly the agent's communication with the MCP server.
- **Characteristics**:
  - **Real MCP Server**: Uses a live `MCPServerStdio` instance to test the full communication protocol.
  - **Mocked LLMs**: LLM calls are mocked to ensure tests are fast, deterministic, and do not incur API costs.
  - **Stateful Patterns**: Tests are class-based and use fixtures to set up and tear down the agent and server, maintaining state across test methods. This is crucial for avoiding event loop errors.
  - **Filesystem**: Use the `test_docs_root` fixture for an isolated temporary directory.
- **Example**: Verifying that the React Agent correctly calls the `create_document` tool on the MCP server.

### 3.3. End-to-End (E2E) Tests (`tests/e2e/`)
- **Purpose**: To test the entire application workflow with real AI models, verifying that the agent's reasoning and tool usage are correct.
- **Characteristics**:
  - **Real API Keys**: Require real API keys (e.g., `OPENAI_API_KEY`) to be set in the environment.
  - **Stateful and Asynchronous**: Follows the same stateful, class-based, and asynchronous patterns as integration tests.
  - **Conditional Execution**: Tests are automatically skipped via `@skip_if_no_real_api_key` if API keys are not available. This prevents failures in CI environments without secrets.
- **Example**: Testing a full user query like "Create a document, add a chapter, and then summarize it."

## 4. Key Architectural Patterns

### 4.1. Stateful, Class-Based Tests
For **Integration** and **E2E** tests, we use a stateful, class-based approach to manage the lifecycle of the agent and MCP server. This prevents `asyncio` event loop errors and ensures a clean state for each test class.

**Key Components**:
- **`agent_base.py`**: Contains base classes (`IntegrationTestBase`, `E2ETestBase`) and mixins (`SimpleAgentTestMixin`, `ReactAgentTestMixin`).
- **Class-level Fixture**: Each test class uses a `pytest.fixture` with `scope="class"` and `autouse=True` to initialize the agent and server once for all tests in that class.
- **Stateful Helpers**: The mixins provide stateful helper methods like `initialize_..._agent_and_mcp_server()` and `run_..._query_on_agent()`, which are called via `self`.

```python
# Example from tests/integration/test_react_agent.py
@pytest.mark.integration
class TestReactAgentIntegration(IntegrationTestBase, ReactAgentTestMixin):
    @pytest.fixture(scope="class", autouse=True)
    async def react_agent_setup(self, request):
        agent, mcp_server = await self.initialize_react_agent_and_mcp_server()
        request.cls.agent = agent
        request.cls.mcp_server = mcp_server
        yield
        await agent.close()
        await mcp_server.close()

    @pytest.mark.asyncio
    async def test_react_agent_some_workflow(self):
        # The agent is already initialized and available via self.agent
        query = "Do something cool."
        await self.run_react_query_on_agent(self.agent, query)
```

### 4.2. Shared Helpers and Fixtures
- **`tests/shared/`**: This directory is critical for reducing code duplication. It contains shared assertion helpers, environment setup functions, and the base classes mentioned above.
- **`conftest.py`**: Defines project-wide fixtures, such as `test_docs_root` (which provides a clean temporary directory) and `skip_if_no_real_api_key`.

## 5. Best Practices

### 5.1. Naming and Documentation
- **Test Names**: Should be descriptive and follow the `test_<area>_<scenario>` pattern (e.g., `test_e2e_react_agent_document_and_chapter_workflow`).
- **Docstrings**: Should be used to explain the *purpose* of the test, not just repeat the test name. Keep them concise.
- **Comments**: Use comments only to explain *why* something is done in a particular way, not *what* is being done.

### 5.2. Test Isolation and State
- **Unit Tests**: Must be fully isolated with no side effects.
- **Integration/E2E Tests**: State is managed at the class level. Tests within a class may build on each other, but each test file should be runnable independently.
- **Fixtures**: Use fixtures (`test_docs_root`, `sample_documents_fixture`) for all setup and teardown logic to ensure a clean environment.

### 5.3. Mocking
- **Strategy**: Mock at the boundaries of the system. For integration tests, this means mocking the LLM but not the MCP server. For unit tests, mock any I/O.
- **Tool**: Use the `mocker` fixture from `pytest-mock`.

### 5.4. Assertions
- Be specific. Instead of `assert response is not None`, check for the specific content or structure you expect.
- Test both positive outcomes and expected error conditions.

## 6. Running Tests

### Run All Tests
```bash
pytest
```

### Run a Specific Category
```bash
pytest tests/unit/
pytest tests/integration/
pytest tests/e2e/
```

### Run with Coverage
```bash
pytest --cov=document_mcp --cov-report=html
```

### Run E2E Tests
To run the E2E tests, you must first export your API key:
```bash
export OPENAI_API_KEY="your-key-here"
pytest tests/e2e/
```

## 7. Adding New Tests

1.  **Determine the Type**:
    - **Unit**: A single, isolated function?
    - **Integration**: Interaction between agent and MCP?
    - **E2E**: A full user workflow that depends on AI reasoning?
2.  **Choose the Right Base Class**: Inherit from `UnitTestBase`, `IntegrationTestBase`, or `E2ETestBase`.
3.  **Add the Right Mixin**: Add `SimpleAgentTestMixin` or `ReactAgentTestMixin` for agent-specific helpers.
4.  **Follow the Pattern**: Mimic the structure of existing tests in the same category. Use the class-based, stateful fixture pattern for integration and E2E tests.
5.  **Write Clear Assertions**: Verify the specific outcomes of your test. For E2E tests, this might involve checking the filesystem for created documents or verifying specific content was written.

## Continuous Integration

Tests are automatically run on:
- Every pull request
- Every commit to main branch
- Nightly scheduled runs

E2E tests are skipped in CI unless API keys are configured as secrets.

## Maintenance

### Regular Tasks
- Update fixtures when adding new features
- Review and update skip conditions
- Keep test dependencies up to date

### Test Quality Metrics
- Aim for >80% code coverage
- Keep test execution time under 5 minutes
- Minimize flaky tests
- Maintain clear test output

## Common Patterns

### Testing Async Functions
```python
@pytest.mark.asyncio
async def test_async_function():
    result = await some_async_function()
    assert result == expected
```

### Testing with Temporary Files
```python
def test_file_operations(tmp_path):
    test_file = tmp_path / "test.txt"
    test_file.write_text("content")
    assert test_file.read_text() == "content"
```

### Testing Error Cases
```python
def test_error_handling(mocker):
    mock_func = mocker.patch('module.function')
    mock_func.side_effect = IOError("Disk error")
    
    with pytest.raises(IOError):
        call_function_that_uses_mock()
```

### Testing with Real MCP Server
```python
@pytest.mark.asyncio
async def test_mcp_integration(mcp_client):
    response = await mcp_client.call_tool(
        "list_documents",
        {}
    )
    assert response["documents"] == []
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure project root is in PYTHONPATH
   - Check for circular imports
   - Verify package structure

2. **Fixture Not Found**
   - Check fixture scope and availability
   - Ensure conftest.py is in the right location
   - Verify fixture names are correct

3. **Async Test Failures**
   - Use `@pytest.mark.asyncio` decorator
   - Ensure proper await usage
   - Check for event loop issues

4. **E2E Test Failures**
   - Verify API keys are set correctly
   - Check API rate limits
   - Ensure network connectivity

## Contributing

When adding new tests:
1. Follow the existing patterns
2. Add appropriate fixtures if needed
3. Update this documentation if adding new patterns
4. Ensure tests pass locally before submitting PR
5. Include both positive and negative test cases 