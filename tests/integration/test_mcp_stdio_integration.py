"""
Integration tests for Document MCP tool functions.

This module tests the tool functions by calling them directly,
ensuring they interact correctly with the filesystem and each other.
"""
import pytest
from document_mcp.doc_tool_server import (
    create_document,
    list_documents,
    create_chapter,
    list_chapters,
    read_chapter_content,
    delete_document,
    delete_chapter,
)

@pytest.mark.integration
def test_create_and_list_documents(test_docs_root):
    """Test creating a document and listing it."""
    # Create a document
    result = create_document(document_name="test_document")
    assert result.success is True
    assert "test_document" in result.message

    # List documents
    documents = list_documents()
    assert isinstance(documents, list)
    assert len(documents) > 0

    doc_names = [doc.document_name for doc in documents]
    assert "test_document" in doc_names

    # Find our document and verify its structure
    test_doc = next(doc for doc in documents if doc.document_name == "test_document")
    assert test_doc.total_chapters == 0
    assert test_doc.total_word_count == 0
    assert test_doc.has_summary is False

@pytest.mark.integration
def test_chapter_operations(test_docs_root):
    """Test chapter creation and reading."""
    doc_name = "chapter_test_doc"
    # Create a document first
    create_document(document_name=doc_name)

    # Create a chapter
    chapter_result = create_chapter(
        document_name=doc_name,
        chapter_name="01-introduction.md",
        initial_content="# Introduction\n\nThis is the introduction chapter."
    )
    assert chapter_result.success is True

    # List chapters
    chapters = list_chapters(document_name=doc_name)
    assert len(chapters) == 1
    assert chapters[0].chapter_name == "01-introduction.md"
    assert chapters[0].word_count > 0

    # Read chapter content
    content = read_chapter_content(
        document_name=doc_name,
        chapter_name="01-introduction.md"
    )
    assert content.content == "# Introduction\n\nThis is the introduction chapter."
    assert content.paragraph_count == 2  # Title and content

@pytest.mark.integration
def test_error_handling_and_deletion(test_docs_root):
    """Test error handling for invalid operations and deletion."""
    doc_name = "error_test_doc"
    create_document(document_name=doc_name)

    # Try to read non-existent chapter
    result = read_chapter_content(
        document_name=doc_name,
        chapter_name="non_existent_chapter.md"
    )
    assert result is None

    # Try to create chapter in non-existent document
    result = create_chapter(
        document_name="non_existent_doc",
        chapter_name="chapter.md",
        initial_content="Content"
    )
    assert result.success is False
    assert "not found" in result.message

    # Test deletion
    delete_chapter_result = delete_chapter(
        document_name=doc_name,
        chapter_name="01-some-chapter.md" # non-existent
    )
    assert delete_chapter_result.success is False

    delete_doc_result = delete_document(document_name=doc_name)
    assert delete_doc_result.success is True

    # Verify document is gone
    docs = list_documents()
    doc_names = [doc.document_name for doc in docs]
    assert doc_name not in doc_names 