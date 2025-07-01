# Testing Guidelines for Document MCP

## Overview

This document outlines the testing strategy, structure, and best practices for the Document MCP project. Our test suite is organized into three main categories: unit tests, integration tests, and end-to-end (e2e) tests.

## Test Structure

```
tests/
├── unit/                  # Isolated function/class tests
├── integration/           # Component interaction tests  
├── e2e/                   # Full system tests with real APIs
├── shared/                # Shared test utilities
├── fixtures/              # Test data and fixtures
└── conftest.py           # Pytest configuration and fixtures
```

## Test Categories

### Unit Tests (`tests/unit/`)
- **Purpose**: Test individual functions and classes in isolation
- **Characteristics**:
  - Heavy use of mocks and stubs
  - No external dependencies (filesystem, network, APIs)
  - Fast execution (< 100ms per test)
  - Use `tmp_path` fixture for file operations
- **Example**: Testing `_count_words()` function with various inputs

### Integration Tests (`tests/integration/`)
- **Purpose**: Test component interactions and MCP server integration
- **Characteristics**:
  - May use real MCP server via stdio
  - Controlled environment with mocked LLMs
  - Test agent-server communication
  - Use `test_docs_root` fixture for isolated filesystem
- **Example**: Testing React Agent's ability to execute MCP tools

### End-to-End Tests (`tests/e2e/`)
- **Purpose**: Test complete workflows with real AI models
- **Characteristics**:
  - Require real API keys (OpenAI, Google, etc.)
  - Skip automatically when no API keys available
  - Test real AI reasoning and decision making
  - Use `@skip_if_no_real_api_key` decorator
- **Example**: Testing document creation and editing workflow

## Key Fixtures

### Environment Fixtures
- `test_docs_root`: Creates isolated temp directory for each test
- `mock_environment`: Sets up mock API keys and environment
- `skip_if_no_real_api_key`: Decorator for e2e tests requiring real APIs

### Document Fixtures
- `sample_documents_fixture`: Sets up sample documents for React Agent tests
- `e2e_sample_documents`: Sets up documents in `.documents_storage` for Simple Agent
- `document_factory`: Factory for creating various test document types

### Mock Fixtures
- `mock_path_operations`: Mock filesystem path operations
- `mock_file_operations`: Mock file read/write operations
- `mock_agent_operations`: Mock LLM and agent operations

## Best Practices

### 1. Test Naming
- Use descriptive test names that explain what is being tested
- Format: `test_<component>_<scenario>_<expected_outcome>`
- Example: `test_read_document_summary_file_not_found_returns_none`

### 2. Test Isolation
- Each test should be independent and idempotent
- Use fixtures for setup/teardown
- Never rely on test execution order
- Clean up resources in teardown

### 3. Mocking Strategy
- Mock at the boundary of your system
- Use `mocker` fixture from pytest-mock
- Prefer dependency injection over patching
- Mock external services, not internal logic

### 4. Assertions
- Use specific assertions with clear error messages
- Test both positive and negative cases
- Verify error handling and edge cases
- Use shared assertion helpers from `tests.shared.assertions`

### 5. Skip Conditions
- Use `@pytest.mark.skipif` for conditional test execution
- Always provide clear skip reasons
- For e2e tests, use `@skip_if_no_real_api_key`

### 6. Test Data
- Use fixtures for test data setup
- Keep test data minimal but realistic
- Use `tests.shared.test_data` for common data
- Avoid hardcoding values in tests

### 7. Documentation
- Keep test docstrings only when they add value
- Don't repeat the test name in docstrings
- Document complex test scenarios
- Use comments sparingly and meaningfully

## Running Tests

### Run All Tests
```bash
pytest
```

### Run Specific Category
```bash
pytest tests/unit/          # Unit tests only
pytest tests/integration/   # Integration tests only
pytest tests/e2e/          # E2E tests only
```

### Run with Coverage
```bash
pytest --cov=document_mcp --cov-report=html
```

### Skip Slow Tests
```bash
pytest -m "not slow"
```

### Run E2E Tests (requires API keys)
```bash
export OPENAI_API_KEY="your-key"
pytest tests/e2e/
```

## Adding New Tests

### 1. Determine Test Type
- Is it testing a single function? → Unit test
- Is it testing component interaction? → Integration test
- Is it testing real AI behavior? → E2E test

### 2. Use Appropriate Fixtures
- Check `conftest.py` for available fixtures
- Prefer existing fixtures over creating new ones
- Create reusable fixtures in `conftest.py`

### 3. Follow Patterns
- Look at existing tests in the same category
- Use consistent mocking strategies
- Follow the established directory structure

### 4. Error Handling Tests
- Test validation errors → WARNING level logs
- Test file not found → INFO level logs
- Test I/O errors → ERROR level logs

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