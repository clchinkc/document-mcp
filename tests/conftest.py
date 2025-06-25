"""
Pytest configuration and fixtures for Document MCP testing.

This module provides pytest-specific configuration and fixtures,
using the shared testing infrastructure for consistency.
"""

import os
import shutil
import sys
import tempfile
import uuid
from pathlib import Path

import pytest
from unittest.mock import AsyncMock, Mock, MagicMock

# Import shared testing utilities
from tests.shared import (
    MCPServerManager,
    create_mock_environment,
    create_test_document,
    generate_unique_name,
)
from tests.shared.test_data import (
    TestDataRegistry,
    TestDataType,
    TestDocumentSpec,
    create_test_document_from_spec,
)
from tests.shared.mock_factories import create_mock_agent

# Import for direct doc_tool_server manipulation in tests
try:
    from document_mcp import doc_tool_server
except ImportError:
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
    manager = MCPServerManager(test_docs_root=temp_root)
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
# Test Data Management Fixtures
# ================================


@pytest.fixture
def test_data_registry(test_docs_root):
    """
    Provide a test data registry for tracking created test data.

    This fixture automatically cleans up all registered test data
    after the test completes.
    """
    registry = TestDataRegistry()
    yield registry

    # Cleanup all registered test data
    registry.cleanup_all(test_docs_root)


@pytest.fixture
def document_factory(test_docs_root, test_data_registry):
    """
    Factory fixture for creating test documents from specifications.

    Returns a function that creates documents and automatically
    registers them for cleanup.
    """
    def _create_document(
        doc_type=None,
        name=None,
        chapter_count=3,
        chapters=None,
        search_terms=None,
        target_word_count=None,
        target_paragraph_count=None,
        paragraphs_per_chapter=5,
        cleanup_on_error=True,
    ):
        """Create a test document from parameters."""
        # Convert string to enum if needed
        if isinstance(doc_type, str):
            doc_type = TestDataType(doc_type)
        elif doc_type is None:
            doc_type = TestDataType.SIMPLE

        spec = TestDocumentSpec(
            name=name,
            doc_type=doc_type,
            chapter_count=chapter_count,
            chapters=chapters,
            search_terms=search_terms,
            target_word_count=target_word_count,
            target_paragraph_count=target_paragraph_count,
            paragraphs_per_chapter=paragraphs_per_chapter,
            cleanup_on_error=cleanup_on_error,
        )

        return create_test_document_from_spec(
            docs_root=test_docs_root,
            spec=spec,
            registry=test_data_registry,
        )

    return _create_document


@pytest.fixture(params=[
    "simple",
    "large",
    "searchable",
    "multi_format",
    "statistical"
])
def parametrized_document(request, document_factory):
    """
    Parametrized fixture that creates different types of test documents.

    This fixture automatically runs tests with different document types,
    useful for comprehensive testing.
    """
    doc_type = request.param

    # Adjust parameters based on document type
    if doc_type == "large":
        return document_factory(
            doc_type=doc_type,
            chapter_count=5,  # Smaller for test performance
            paragraphs_per_chapter=3,
        )
    elif doc_type == "statistical":
        return document_factory(
            doc_type=doc_type,
            target_word_count=200,  # Smaller for test performance
            target_paragraph_count=10,
        )
    else:
        return document_factory(doc_type=doc_type)


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
def validate_test_data():
    """
    Fixture to validate test data integrity during and after tests.

    This fixture provides utilities to ensure test data is in the expected state.
    """
    class Validator:
        """Test data validator with bound methods."""

        def document_exists(self, docs_root, doc_name):
            """Validate that a document exists and has the expected structure."""
            doc_path = docs_root / doc_name
            assert doc_path.exists(), f"Document {doc_name} does not exist"
            assert doc_path.is_dir(), f"Document {doc_name} is not a directory"

            # Check for at least one markdown file
            md_files = list(doc_path.glob("*.md"))
            assert len(md_files) > 0, f"Document {doc_name} has no markdown files"

            return True

        def registry_state(self, registry, docs_root):
            """Validate that the registry state matches the actual filesystem."""
            issues = registry.validate_test_state(docs_root)
            assert len(issues) == 0, f"Test data validation issues: {issues}"
            return True

        def document_content(self, docs_root, doc_name, expected_chapters=None):
            """Validate document content structure."""
            doc_path = docs_root / doc_name

            if expected_chapters:
                for chapter_name, expected_content in expected_chapters:
                    chapter_path = doc_path / chapter_name
                    assert chapter_path.exists(), f"Chapter {chapter_name} does not exist"

                    actual_content = chapter_path.read_text()
                    if expected_content:
                        assert expected_content in actual_content, \
                            f"Expected content not found in {chapter_name}"

            return True

    return Validator()


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


# ================================
# Pytest-Mock Based Fixtures (Replaces unittest.mock patches)
# ================================

@pytest.fixture
def mock_path_operations(mocker):
    """
    Centralized fixture for mocking pathlib.Path operations.
    
    Returns a namespace object with common path mocking utilities.
    """
    class PathMocks:
        def __init__(self, mocker):
            self.mocker = mocker
            
        def mock_docs_root_path(self, path="/test/docs"):
            """Mock DOCS_ROOT_PATH with specified path."""
            # Create a mock Path object that can be configured
            mock_path_obj = MagicMock(spec=Path)
            mock_path_obj.exists.return_value = True
            mock_path_obj.is_dir.return_value = True
            mock_path_obj.iterdir.return_value = []
            
            # Make path division operations return real Path objects
            base_path = Path(path)
            mock_path_obj.__truediv__ = lambda self, other: base_path / other
            
            # Patch DOCS_ROOT_PATH to use our mock
            self.mocker.patch(
                "document_mcp.doc_tool_server.DOCS_ROOT_PATH", 
                mock_path_obj
            )
            
            # Return the mock object so tests can configure it
            return mock_path_obj
            
        def mock_document_path(self, document_name, path=None):
            """Mock _get_document_path function."""
            if path is None:
                path = Path(f"/test/docs/{document_name}")
            return self.mocker.patch(
                "document_mcp.doc_tool_server._get_document_path",
                return_value=path
            )
            
        def mock_chapter_path(self, document_name, chapter_name, path=None):
            """Mock _get_chapter_path function."""
            if path is None:
                path = Path(f"/test/docs/{document_name}/{chapter_name}")
            return self.mocker.patch(
                "document_mcp.doc_tool_server._get_chapter_path",
                return_value=path
            )
            
        def create_mock_file(self, name, content="", is_file=True, is_dir=False):
            """Create a mock file/directory object."""
            mock_file = Mock()
            mock_file.name = name
            mock_file.is_file.return_value = is_file
            mock_file.is_dir.return_value = is_dir
            if content:
                mock_file.read_text.return_value = content
            # Add sorting support for chapter ordering
            mock_file.__lt__ = lambda self, other: self.name < other.name
            return mock_file
            
    return PathMocks(mocker)


@pytest.fixture
def mock_file_operations(mocker):
    """
    Centralized fixture for mocking file I/O operations.
    
    Returns utilities for mocking file read/write operations.
    """
    class FileMocks:
        def __init__(self, mocker):
            self.mocker = mocker
            
        def mock_read_text(self, target, content="", side_effect=None):
            """Mock file read_text method."""
            if side_effect:
                return self.mocker.patch.object(target, 'read_text', side_effect=side_effect)
            return self.mocker.patch.object(target, 'read_text', return_value=content)
            
        def mock_write_text(self, target):
            """Mock file write_text method."""
            return self.mocker.patch.object(target, 'write_text')
            
        def mock_stat(self, target, mtime=1640995200.0):
            """Mock file stat method."""
            mock_stat = Mock()
            mock_stat.st_mtime = mtime
            return self.mocker.patch.object(target, 'stat', return_value=mock_stat)
            
        def mock_print(self):
            """Mock print function for error message testing."""
            return self.mocker.patch("builtins.print")
            
    return FileMocks(mocker)


@pytest.fixture  
def mock_agent_operations(mocker):
    """
    Centralized fixture for mocking AI agent operations.
    
    Returns utilities for mocking agent initialization and execution.
    """
    class AgentMocks:
        def __init__(self, mocker):
            self.mocker = mocker
            
        def mock_load_llm_config(self, return_value=None):
            """Mock load_llm_config function."""
            if return_value is None:
                return_value = Mock()
            # Handle both sync and async versions
            mock_sync = self.mocker.patch("src.agents.simple_agent.load_llm_config", return_value=return_value)
            mock_async = self.mocker.patch("src.agents.react_agent.main.load_llm_config", return_value=AsyncMock(return_value=return_value))
            return mock_sync, mock_async
            
        def mock_mcp_server(self, host="localhost", port=3001):
            """Mock MCP server class and instance."""
            mock_server_instance = Mock()
            mock_server_instance.host = host
            mock_server_instance.port = port
            mock_server_instance.server_url = f"http://{host}:{port}"
            mock_server_instance.__aenter__ = AsyncMock(return_value=mock_server_instance)
            mock_server_instance.__aexit__ = AsyncMock(return_value=False)
            
            mock_server_class = self.mocker.patch("src.agents.simple_agent.MCPServerSSE", return_value=mock_server_instance)
            mock_server_class_react = self.mocker.patch("src.agents.react_agent.main.MCPServerSSE", return_value=mock_server_instance)
            
            return mock_server_instance, mock_server_class, mock_server_class_react
            
        def mock_agent_class(self, response_data=None):
            """Mock Agent class with configurable response."""
            if response_data is None:
                response_data = {"summary": "Mock response", "details": None, "error_message": None}
                
            mock_agent = create_mock_agent(response_data)
            mock_agent_class = self.mocker.patch("src.agents.simple_agent.Agent", return_value=mock_agent)
            mock_agent_class_react = self.mocker.patch("src.agents.react_agent.main.get_cached_agent", return_value=mock_agent)
            
            return mock_agent, mock_agent_class, mock_agent_class_react
            
        def mock_execute_tool(self, return_value='{"success": true}'):
            """Mock MCP tool execution."""
            return self.mocker.patch("src.agents.react_agent.main.execute_mcp_tool_directly", return_value=return_value)
            
        def setup_agent_test_environment(self, api_type="openai"):
            """Set up complete agent testing environment."""
            # Mock agent components
            mock_agent, mock_agent_class, mock_agent_class_react = self.mock_agent_class()
            mock_server_instance, mock_server_class, mock_server_class_react = self.mock_mcp_server()
            mock_sync_llm, mock_async_llm = self.mock_load_llm_config()
            
            return {
                'agent': mock_agent,
                'agent_class': mock_agent_class,
                'agent_class_react': mock_agent_class_react,
                'server_instance': mock_server_instance,
                'server_class': mock_server_class,
                'server_class_react': mock_server_class_react,
                'llm_sync': mock_sync_llm,
                'llm_async': mock_async_llm
            }
            
    return AgentMocks(mocker)


@pytest.fixture
def mock_environment_operations(mocker, monkeypatch):
    """
    Centralized fixture for mocking environment operations.
    
    Combines monkeypatch with mocker for environment variable management.
    """
    class EnvironmentMocks:
        def __init__(self, mocker, monkeypatch):
            self.mocker = mocker
            self.monkeypatch = monkeypatch
            
        def set_api_environment(self, api_type="openai", **custom_vars):
            """Set up API environment variables using monkeypatch."""
            env_vars = create_mock_environment(
                api_key_type=api_type,
                include_server_config=True,
                custom_vars=custom_vars
            )
            
            for key, value in env_vars.items():
                self.monkeypatch.setenv(key, value)
                
            return env_vars
            
        def mock_os_environ(self, env_vars):
            """Mock os.environ with specified variables."""
            return self.mocker.patch.dict("os.environ", env_vars, clear=True)
            
    return EnvironmentMocks(mocker, monkeypatch)


@pytest.fixture
def mock_validation_operations(mocker):
    """
    Centralized fixture for mocking validation functions.
    """
    class ValidationMocks:
        def __init__(self, mocker):
            self.mocker = mocker
            
        def mock_validate_document_name(self, is_valid=True, error_message=""):
            """Mock document name validation."""
            return self.mocker.patch(
                "document_mcp.doc_tool_server._validate_document_name",
                return_value=(is_valid, error_message)
            )
            
        def mock_validate_chapter_name(self, is_valid=True, error_message=""):
            """Mock chapter name validation."""
            return self.mocker.patch(
                "document_mcp.doc_tool_server._validate_chapter_name", 
                return_value=(is_valid, error_message)
            )
            
    return ValidationMocks(mocker)


@pytest.fixture 
def mock_complete_test_environment(
    mock_path_operations, 
    mock_file_operations, 
    mock_agent_operations, 
    mock_environment_operations,
    mock_validation_operations
):
    """
    Comprehensive fixture that provides all mocking utilities in one place.
    
    This is a convenience fixture for tests that need multiple mocking capabilities.
    """
    class CompleteMockEnvironment:
        def __init__(self):
            self.paths = mock_path_operations
            self.files = mock_file_operations  
            self.agents = mock_agent_operations
            self.environment = mock_environment_operations
            self.validation = mock_validation_operations
            
        def setup_basic_document_test(self, doc_name="test_doc", chapters=None):
            """Set up a basic document testing environment."""
            if chapters is None:
                chapters = ["01-intro.md", "02-content.md"]
                
            # Mock docs root
            self.paths.mock_docs_root_path("/test/docs")
            
            # Create mock chapters
            mock_chapters = []
            for chapter_name in chapters:
                mock_chapter = self.paths.create_mock_file(
                    chapter_name, 
                    content=f"# {chapter_name}\n\nContent for {chapter_name}",
                    is_file=True
                )
                mock_chapters.append(mock_chapter)
                
            # Mock document directory
            mock_doc_path = Mock()
            mock_doc_path.is_dir.return_value = True
            mock_doc_path.iterdir.return_value = mock_chapters
            
            self.paths.mock_document_path(doc_name, mock_doc_path)
            
            return mock_doc_path, mock_chapters
            
        def setup_agent_test_environment(self, api_type="openai"):
            """Set up complete agent testing environment."""
            # Set environment variables
            self.environment.set_api_environment(api_type)
            
            # Mock agent components
            mock_agent, mock_agent_class, mock_agent_class_react = self.agents.mock_agent_class()
            mock_server_instance, mock_server_class, mock_server_class_react = self.agents.mock_mcp_server()
            mock_sync_llm, mock_async_llm = self.agents.mock_load_llm_config()
            
            return {
                'agent': mock_agent,
                'agent_class': mock_agent_class,
                'agent_class_react': mock_agent_class_react,
                'server_instance': mock_server_instance,
                'server_class': mock_server_class,
                'server_class_react': mock_server_class_react,
                'llm_sync': mock_sync_llm,
                'llm_async': mock_async_llm
            }
            
    return CompleteMockEnvironment()
