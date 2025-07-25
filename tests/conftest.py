"""Minimal pytest configuration for simplified Document MCP testing.

This provides only essential fixtures needed for the new stdio-based testing approach.
"""

import os
import tempfile
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def check_api_key_available() -> bool:
    """Check if a real API key is available for testing."""
    api_keys = ["OPENAI_API_KEY", "GEMINI_API_KEY"]
    for key in api_keys:
        value = os.environ.get(key, "").strip()
        if value and value != "test_key" and not value.startswith("sk-test"):
            return True
    return False


@pytest.fixture
def temp_docs_root():
    """Provide a temporary directory for document storage with mock API key."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        old_doc_root = os.environ.get("DOCUMENT_ROOT_DIR")
        old_openai_key = os.environ.get("OPENAI_API_KEY")
        old_gemini_key = os.environ.get("GEMINI_API_KEY")

        # Set up environment for integration tests
        os.environ["DOCUMENT_ROOT_DIR"] = tmp_dir

        # Use OpenAI with a mock key for integration tests (simpler schema)
        if not old_openai_key:
            os.environ["OPENAI_API_KEY"] = "sk-test-key-for-integration-testing"
            os.environ["OPENAI_MODEL_NAME"] = "gpt-4o"

        # Remove Gemini key to force OpenAI usage (avoid schema issues)
        if "GEMINI_API_KEY" in os.environ:
            del os.environ["GEMINI_API_KEY"]

        # Force reload of doc_tool_server to pick up new DOCUMENT_ROOT_DIR
        import importlib
        import sys

        if "document_mcp.doc_tool_server" in sys.modules:
            importlib.reload(sys.modules["document_mcp.doc_tool_server"])

        try:
            yield Path(tmp_dir)
        finally:
            # Restore environment
            if old_doc_root:
                os.environ["DOCUMENT_ROOT_DIR"] = old_doc_root
            else:
                os.environ.pop("DOCUMENT_ROOT_DIR", None)

            if old_openai_key:
                os.environ["OPENAI_API_KEY"] = old_openai_key
            elif "OPENAI_API_KEY" in os.environ:
                del os.environ["OPENAI_API_KEY"]

            if old_gemini_key:
                os.environ["GEMINI_API_KEY"] = old_gemini_key

            # Force reload again to restore original settings
            if "document_mcp.doc_tool_server" in sys.modules:
                importlib.reload(sys.modules["document_mcp.doc_tool_server"])


@pytest.fixture
def document_factory(temp_docs_root):
    """Factory for creating test documents."""

    def _create_document(doc_name: str, chapters: dict = None):
        doc_path = temp_docs_root / doc_name
        doc_path.mkdir(exist_ok=True)
        if chapters:
            for chapter_name, content in chapters.items():
                (doc_path / chapter_name).write_text(content, encoding='utf-8')
        return doc_path

    return _create_document


@pytest.fixture
def clean_documents():
    """Clean all documents before each test."""
    from document_mcp.mcp_client import delete_document
    from document_mcp.mcp_client import list_documents

    # Clean existing documents before test
    try:
        docs = list_documents()
        for doc in docs:
            delete_document(doc.document_name)
    except Exception:
        pass  # Ignore cleanup errors

    yield

    # Clean up after test
    try:
        docs = list_documents()
        for doc in docs:
            delete_document(doc.document_name)
    except Exception:
        pass  # Ignore cleanup errors


# Custom markers for pytest
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "stdio: marks tests as stdio-based integration tests")
    config.addinivalue_line("markers", "e2e: marks tests as end-to-end tests requiring real API keys")


# Skip decorator for tests requiring real API keys
skip_if_no_api_key = pytest.mark.skipif(
    not check_api_key_available(),
    reason="Test requires a real API key (OPENAI_API_KEY or GEMINI_API_KEY)",
)
