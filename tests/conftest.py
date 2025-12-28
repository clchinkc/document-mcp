"""The pytest configuration for Document MCP testing.

This provides essential fixtures using the new test environment management
infrastructure for cleaner and more maintainable test setup.
"""

import os

import pytest
from dotenv import load_dotenv

from .shared.test_environment import EnvironmentManager
from .shared.test_environment import TemporaryDocumentRoot
from .shared.test_environment import check_api_key_available

# Load environment variables
load_dotenv()


@pytest.fixture(scope="session", autouse=True)
def disable_observability_for_tests():
    """Disable OpenTelemetry observability during tests to prevent shutdown warnings.

    This session-scoped fixture runs automatically before any tests and prevents
    the 'shutdown can only be called once' warning from OpenTelemetry.
    """
    original_value = os.environ.get("MCP_OBSERVABILITY_ENABLED")
    os.environ["MCP_OBSERVABILITY_ENABLED"] = "false"
    yield
    # Restore original value
    if original_value is not None:
        os.environ["MCP_OBSERVABILITY_ENABLED"] = original_value
    elif "MCP_OBSERVABILITY_ENABLED" in os.environ:
        del os.environ["MCP_OBSERVABILITY_ENABLED"]


@pytest.fixture
def temp_docs_root():
    """Provide a temporary directory for document storage with environment management.

    Uses the new EnvironmentManager for cleaner setup and automatic cleanup.
    """
    env_manager = EnvironmentManager()

    with TemporaryDocumentRoot(env_manager) as tmp_path:
        # Set up mock API environment for integration tests
        with env_manager.mock_api_environment(provider="openai"):
            yield tmp_path


@pytest.fixture
def document_factory(temp_docs_root):
    """Factory for creating test documents."""

    def _create_document(doc_name: str, chapters: dict = None):
        doc_path = temp_docs_root / doc_name
        doc_path.mkdir(exist_ok=True)
        if chapters:
            for chapter_name, content in chapters.items():
                (doc_path / chapter_name).write_text(content, encoding="utf-8")
        return doc_path

    return _create_document


@pytest.fixture
def test_env_manager():
    """Provide a test environment manager for advanced test scenarios."""
    return EnvironmentManager()


# Custom markers for pytest
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "stdio: marks tests as stdio-based integration tests")
    config.addinivalue_line("markers", "e2e: marks tests as end-to-end tests requiring real API keys")
    # Filter OpenTelemetry shutdown warning that appears when multiple providers exist
    config.addinivalue_line(
        "filterwarnings",
        "ignore:shutdown can only be called once:UserWarning",
    )


# Skip decorator for tests requiring real API keys
skip_if_no_api_key = pytest.mark.skipif(
    not check_api_key_available(),
    reason="Test requires a real API key (OPENAI_API_KEY or GEMINI_API_KEY)",
)
