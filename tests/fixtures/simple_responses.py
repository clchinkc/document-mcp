"""
Simple test fixtures and response helpers.

This module provides minimal, straightforward fixtures for testing
without the complexity of the previous mock factory system.
"""

import os
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional

import pytest


@pytest.fixture
def clean_test_env():
    """Provide a clean test environment with temporary document storage."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Set up clean environment variables
        old_doc_root = os.environ.get("DOCUMENT_ROOT_DIR")
        old_api_key = os.environ.get("OPENAI_API_KEY")
        
        # Set test environment
        os.environ["DOCUMENT_ROOT_DIR"] = tmp_dir
        if not old_api_key:
            os.environ["OPENAI_API_KEY"] = "test-key-for-mocking"
        
        try:
            yield Path(tmp_dir)
        finally:
            # Restore environment
            if old_doc_root:
                os.environ["DOCUMENT_ROOT_DIR"] = old_doc_root
            else:
                os.environ.pop("DOCUMENT_ROOT_DIR", None)
            
            if not old_api_key and "OPENAI_API_KEY" in os.environ:
                os.environ.pop("OPENAI_API_KEY", None)


@pytest.fixture
def mock_api_environment(monkeypatch):
    """Mock API environment for integration tests."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-integration-key")
    monkeypatch.setenv("OPENAI_MODEL_NAME", "gpt-4o")


def create_simple_agent_response(
    summary: str,
    details: Optional[Any] = None,
    error_message: Optional[str] = None
) -> Dict[str, Any]:
    """Create a simple agent response for testing."""
    return {
        "summary": summary,
        "details": details,
        "error_message": error_message
    }


def create_document_success_response(doc_name: str) -> Dict[str, Any]:
    """Create a successful document creation response."""
    return create_simple_agent_response(
        summary=f"Document '{doc_name}' created successfully",
        details={"success": True, "message": f"Document '{doc_name}' created"}
    )


def create_chapter_success_response(doc_name: str, chapter_name: str) -> Dict[str, Any]:
    """Create a successful chapter creation response."""
    return create_simple_agent_response(
        summary=f"Chapter '{chapter_name}' created in document '{doc_name}'",
        details={"success": True, "message": f"Chapter created in {doc_name}"}
    )


def create_error_response(error_msg: str) -> Dict[str, Any]:
    """Create an error response."""
    return create_simple_agent_response(
        summary=f"Error: {error_msg}",
        details=None,
        error_message=error_msg
    )


def create_document_list_response(doc_names: list) -> Dict[str, Any]:
    """Create a document list response."""
    docs = [{"document_name": name, "total_chapters": 1} for name in doc_names]
    return create_simple_agent_response(
        summary=f"Found {len(docs)} documents",
        details=docs
    )


def create_chapter_content_response(content: str, doc_name: str, chapter_name: str) -> Dict[str, Any]:
    """Create a chapter content response."""
    return create_simple_agent_response(
        summary=f"Chapter content from {chapter_name}",
        details={
            "document_name": doc_name,
            "chapter_name": chapter_name,
            "content": content,
            "word_count": len(content.split()),
            "paragraph_count": len(content.split('\n\n'))
        }
    )


# Common test data
SAMPLE_DOCUMENT_NAMES = [
    "test_document_1",
    "test_document_2", 
    "sample_book",
    "user_guide"
]

SAMPLE_CHAPTER_NAMES = [
    "01-introduction.md",
    "02-getting-started.md",
    "03-advanced-topics.md",
    "04-conclusion.md"
]

SAMPLE_CONTENT = {
    "intro": "# Introduction\n\nThis is an introduction chapter.",
    "getting_started": "# Getting Started\n\nThis chapter covers the basics.",
    "advanced": "# Advanced Topics\n\nThis chapter covers advanced concepts.",
    "conclusion": "# Conclusion\n\nThis concludes our document."
}


def get_sample_content(content_type: str = "intro") -> str:
    """Get sample content for testing."""
    return SAMPLE_CONTENT.get(content_type, SAMPLE_CONTENT["intro"])


def assert_response_valid(response: Dict[str, Any], context: str = "test"):
    """Simple assertion helper for validating responses."""
    assert isinstance(response, dict), f"Response should be dict in {context}"
    assert "summary" in response, f"Response should have summary in {context}"
    assert response["summary"], f"Summary should not be empty in {context}"


def assert_success_response(response: Dict[str, Any], context: str = "test"):
    """Assert that a response indicates success."""
    assert_response_valid(response, context)
    summary = response["summary"].lower()
    error_msg = response.get("error_message", "")
    
    assert (
        "success" in summary or 
        "created" in summary or
        "completed" in summary
    ), f"Response should indicate success in {context}: {response['summary']}"
    
    assert not error_msg, f"Should not have error message in {context}: {error_msg}"


def assert_error_response(response: Dict[str, Any], expected_error: str, context: str = "test"):
    """Assert that a response indicates the expected error."""
    assert_response_valid(response, context)
    
    summary = response["summary"].lower()
    error_msg = (response.get("error_message") or "").lower()
    expected_lower = expected_error.lower()
    
    assert (
        expected_lower in summary or 
        expected_lower in error_msg or
        "error" in summary
    ), f"Response should indicate error '{expected_error}' in {context}: {response}"