"""
Pytest configuration and fixtures for Document MCP testing.

This module provides pytest-specific configuration and fixtures,
using the shared testing infrastructure for consistency.
"""

import os
import shutil
import tempfile
import uuid
from pathlib import Path

import pytest

# Import shared testing utilities
from tests.shared import (
    UnifiedMCPServerManager,
    create_mock_environment,
    create_test_document,
    generate_unique_name,
)

# Import for direct doc_tool_server manipulation in tests
try:
    from document_mcp import doc_tool_server
except ImportError:
    import sys

    sys.path.insert(0, "..")
    from document_mcp import doc_tool_server


# ================================
# Pytest Hooks for Environment Management
# ================================


def pytest_runtest_setup(item):
    """
    Set up environment for each test run.

    This hook ensures PYTEST_CURRENT_TEST is properly set
    to avoid teardown errors in Python 3.13.
    """
    test_name = f"{item.nodeid}"
    os.environ["PYTEST_CURRENT_TEST"] = test_name


def pytest_runtest_teardown(item, nextitem):
    """
    Clean up environment after each test run.

    This hook safely removes PYTEST_CURRENT_TEST to prevent
    KeyError during pytest's internal teardown process.
    """
    # Safely remove the environment variable
    os.environ.pop("PYTEST_CURRENT_TEST", None)


# ================================
# Session-scoped Server Management
# ================================


@pytest.fixture(scope="session")
def session_server_manager():
    """
    Session-scoped server manager to minimize server startup/shutdown overhead.

    This fixture creates a single server instance that is shared across
    all tests in the session, significantly improving test performance.
    """
    # Create a temporary docs root for the entire test session
    temp_root = Path(tempfile.mkdtemp(prefix="mcp_test_session_"))

    # Create server manager with session-scoped temp directory
    manager = UnifiedMCPServerManager(test_docs_root=temp_root)
    manager.start_server()

    yield manager

    # Cleanup
    manager.stop_server()
    if temp_root.exists():
        shutil.rmtree(temp_root, ignore_errors=True)


@pytest.fixture
def test_docs_root(session_server_manager):
    """
    Create a clean subdirectory for each test.

    This fixture provides test isolation while reusing the same server instance.
    Each test gets its own document directory but shares the server.
    """
    test_id = str(uuid.uuid4().hex[:8])
    test_subdir = session_server_manager.test_docs_root / f"test_{test_id}"
    test_subdir.mkdir()

    # Override doc_tool_server paths for this test
    original_server_path = doc_tool_server.DOCS_ROOT_PATH
    doc_tool_server.DOCS_ROOT_PATH = test_subdir

    # Set environment variables for the test
    original_env_vars = {}
    test_env_vars = {
        "DOCUMENT_ROOT_DIR": str(test_subdir),
        "MCP_SERVER_PORT": str(session_server_manager.get_port()),
        "MCP_SERVER_HOST": "localhost",
    }

    for key, value in test_env_vars.items():
        original_env_vars[key] = os.environ.get(key)
        os.environ[key] = value

    yield test_subdir

    # Cleanup test data
    if test_subdir.exists():
        shutil.rmtree(test_subdir, ignore_errors=True)

    # Restore original environment
    for key, original_value in original_env_vars.items():
        if original_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original_value

    # Restore original server path
    doc_tool_server.DOCS_ROOT_PATH = original_server_path


# ================================
# Environment and Configuration Fixtures
# ================================


@pytest.fixture
def mock_environment():
    """
    Set up mock environment for testing AI agents.

    Uses shared mock environment creation for consistency.
    """
    env_vars = create_mock_environment(
        api_key_type="openai",
        include_server_config=True,
    )

    with pytest.MonkeyPatch().context() as m:
        for key, value in env_vars.items():
            m.setenv(key, value)
        yield


@pytest.fixture
def mock_gemini_environment():
    """Set up mock environment with Gemini API key."""
    env_vars = create_mock_environment(
        api_key_type="gemini",
        include_server_config=True,
    )

    with pytest.MonkeyPatch().context() as m:
        for key, value in env_vars.items():
            m.setenv(key, value)
        yield


@pytest.fixture
def mock_both_api_keys_environment():
    """Set up mock environment with both OpenAI and Gemini API keys."""
    env_vars = create_mock_environment(
        api_key_type="both",
        include_server_config=True,
    )

    with pytest.MonkeyPatch().context() as m:
        for key, value in env_vars.items():
            m.setenv(key, value)
        yield


# ================================
# Test Data Fixtures
# ================================


@pytest.fixture
def sample_document(test_docs_root):
    """
    Create a sample document with default structure for testing.

    Returns the document name for use in tests.
    """
    doc_name = create_test_document(
        docs_root=test_docs_root,
        doc_name=None,  # Auto-generate name
        chapter_count=3,
    )
    return doc_name


@pytest.fixture
def large_document(test_docs_root):
    """
    Create a large document for performance testing.

    Returns the document name for use in tests.
    """
    from tests.shared.test_data import create_large_test_document

    doc_name = create_large_test_document(
        docs_root=test_docs_root,
        doc_name=None,  # Auto-generate name
        chapter_count=10,  # Smaller than default for faster tests
        paragraphs_per_chapter=5,
    )
    return doc_name


@pytest.fixture
def searchable_document(test_docs_root):
    """
    Create a document with searchable content for search tests.

    Returns tuple of (document_name, search_terms).
    """
    from tests.shared.test_data import create_searchable_test_document

    return create_searchable_test_document(
        docs_root=test_docs_root,
        doc_name=None,  # Auto-generate name
    )


@pytest.fixture
def empty_document(test_docs_root):
    """
    Create an empty document directory for testing edge cases.

    Returns the document name.
    """
    doc_name = generate_unique_name("empty_doc")
    doc_path = test_docs_root / doc_name
    doc_path.mkdir(parents=True, exist_ok=True)
    return doc_name


# ================================
# Utility Fixtures
# ================================


@pytest.fixture
def unique_name():
    """Generate a unique name for test resources."""
    return generate_unique_name()


@pytest.fixture
def server_url(session_server_manager):
    """Get the server URL for the current test session."""
    return session_server_manager.get_server_url()


@pytest.fixture
def server_port(session_server_manager):
    """Get the server port for the current test session."""
    return session_server_manager.get_port()


# ================================
# Pytest Configuration
# ================================


def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "e2e: marks tests as end-to-end tests")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add markers based on test file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)

        # Mark slow tests
        if "large" in item.name or "performance" in item.name:
            item.add_marker(pytest.mark.slow)


# ================================
# Cleanup and Error Handling
# ================================


@pytest.fixture(autouse=True)
def cleanup_environment():
    """
    Automatically clean up environment after each test.

    This fixture runs automatically for every test to ensure
    clean state between tests.
    """
    # Setup - nothing needed here
    yield

    # Cleanup - ensure no test environment variables leak
    test_vars_to_clear = [
        "TEST_MODE",
    ]

    for var in test_vars_to_clear:
        if var in os.environ:
            del os.environ[var]


@pytest.fixture
def capture_logs(caplog):
    """
    Capture logs during testing with appropriate levels.

    This fixture configures logging for tests and returns
    the caplog fixture for log assertions.
    """
    import logging

    # Set appropriate log levels for testing
    logging.getLogger("document_mcp").setLevel(logging.WARNING)
    logging.getLogger("tests").setLevel(logging.DEBUG)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    return caplog
