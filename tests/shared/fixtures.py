"""Standardized fixtures for all test types in the Document MCP project.

This module provides centralized fixture management to eliminate duplication
across test files and ensure consistent test setup patterns.
"""

from collections.abc import Callable
from pathlib import Path

import pytest

from tests.tool_imports import delete_document


@pytest.fixture(scope="session")
def test_environment():
    """Global test environment setup and configuration."""
    # Any session-wide setup can go here
    yield
    # Session-wide cleanup


@pytest.fixture
def document_factory(
    temp_docs_root: Path,
) -> Callable[[str, dict[str, str] | None], Path]:
    """Standardized document creation factory for testing.

    Creates documents with chapters and handles cleanup automatically.
    Used across integration tests for consistent document creation patterns.

    Args:
        doc_name: Name of the document to create
        chapters: Optional dict of {chapter_name: content} to create

    Returns:
        Path to the created document directory
    """
    created_docs = []

    def _create_document(doc_name: str, chapters: dict[str, str] = None) -> Path:
        doc_path = temp_docs_root / doc_name
        doc_path.mkdir(exist_ok=True)
        created_docs.append(doc_name)
        if chapters:
            for chapter_name, content in chapters.items():
                (doc_path / chapter_name).write_text(content)
        return doc_path

    try:
        yield _create_document
    finally:
        for doc_name in created_docs:
            delete_document(doc_name)


@pytest.fixture
def para_doc(document_factory) -> tuple[str, str]:
    """Fixture for a document with paragraphs for manipulation tests.

    Creates a standard document with three paragraphs for testing
    paragraph-level operations like replace, insert, delete.

    Returns:
        Tuple of (document_name, chapter_name) for the test document
    """
    doc_name = "para_doc"
    chapter_name = "chap1.md"
    content = "Paragraph 1.\n\nParagraph 2.\n\nParagraph 3."
    document_factory(doc_name, {chapter_name: content})
    return doc_name, chapter_name


@pytest.fixture
def temp_workspace(tmp_path: Path) -> Path:
    """Isolated workspace for each test.

    Provides a clean temporary directory for tests that need
    file system operations without affecting other tests.
    """
    workspace = tmp_path / "test_workspace"
    workspace.mkdir()
    return workspace


@pytest.fixture
def sample_document_content() -> dict[str, str]:
    """Standard sample document content for testing.

    Provides consistent test data across different test modules.
    """
    return {
        "01-introduction.md": "# Introduction\n\nWelcome to the test document.",
        "02-chapter.md": "# Chapter 2\n\nThis is the main content.",
        "03-conclusion.md": "# Conclusion\n\nThat's all for now.",
    }


@pytest.fixture
def complex_document_content() -> dict[str, str]:
    """Complex document content for advanced testing scenarios.

    Provides structured content with multiple paragraphs, formatting,
    and cross-references for testing complex operations.
    """
    return {
        "intro.md": (
            "# Introduction\n\n"
            "This is the first paragraph.\n\n"
            "This is the second paragraph with **bold** text.\n\n"
            "Final paragraph in introduction."
        ),
        "methods.md": (
            "# Methods\n\n"
            "## Subsection 1\n\n"
            "Method description here.\n\n"
            "## Subsection 2\n\n"
            "Another method description."
        ),
        "results.md": (
            "# Results\n\n"
            "Key findings:\n\n"
            "- Finding 1\n"
            "- Finding 2\n"
            "- Finding 3\n\n"
            "Detailed analysis follows."
        ),
    }


@pytest.fixture
def search_test_content() -> dict[str, str]:
    """Content specifically designed for testing search and replace operations."""
    return {
        "chapter1.md": "The keyword appears here. And the keyword appears again.",
        "chapter2.md": "Another instance of the keyword in different chapter.",
        "chapter3.md": "No special terms in this chapter content.",
    }


# Compatibility fixtures for backward compatibility during transition
@pytest.fixture
def document_test_data(sample_document_content):
    """Backward compatibility alias for sample_document_content."""
    return sample_document_content


@pytest.fixture
def test_document_factory(document_factory):
    """Backward compatibility alias for document_factory."""
    return document_factory
