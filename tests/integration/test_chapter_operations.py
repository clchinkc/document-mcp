"""Integration tests for chapter-level operations in the Document MCP tool server."""

from pathlib import Path

from document_mcp.mcp_client import create_chapter
from document_mcp.mcp_client import delete_chapter
from document_mcp.mcp_client import get_statistics
from document_mcp.mcp_client import list_chapters
from document_mcp.mcp_client import read_content
from document_mcp.mcp_client import write_chapter_content


def test_create_and_list_chapter(document_factory):
    """Test creating a chapter and verifying it's listed."""
    doc_name = "chapter_test_doc"
    chapter_name = "new_chap.md"
    document_factory(doc_name)

    assert list_chapters(doc_name) == []

    create_result = create_chapter(doc_name, chapter_name, "Initial content.")
    assert create_result.success is True

    chapters_list = list_chapters(doc_name)
    assert len(chapters_list) == 1
    assert chapters_list[0].chapter_name == chapter_name


def test_delete_chapter(document_factory):
    """Test deleting a chapter."""
    doc_name = "doc_with_chap_to_delete"
    chapter_name = "chap_to_delete.md"
    document_factory(doc_name, {chapter_name: "content"})

    assert len(list_chapters(doc_name)) == 1

    delete_result = delete_chapter(doc_name, chapter_name)
    assert delete_result.success is True

    assert list_chapters(doc_name) == []


def test_read_and_write_chapter_content(document_factory, temp_docs_root: Path):
    """Test reading and then overwriting chapter content."""
    doc_name = "read_write_doc"
    chapter_name = "chap.md"
    initial_content = "This is the original content."
    document_factory(doc_name, {chapter_name: initial_content})

    read_result = read_content(doc_name, scope="chapter", chapter_name=chapter_name)
    assert read_result.content == initial_content

    new_content = "This is the new, updated content."
    write_result = write_chapter_content(doc_name, chapter_name, new_content)
    assert write_result.success is True

    final_content = (temp_docs_root / doc_name / chapter_name).read_text()
    assert final_content == new_content


def test_chapter_statistics(document_factory):
    """Test getting statistics for a chapter."""
    doc_name = "chap_stats_doc"
    chapter_name = "chap.md"
    content = "This chapter has five words.\n\nAnd two paragraphs."
    document_factory(doc_name, {chapter_name: content})

    stats_result = get_statistics(doc_name, scope="chapter", chapter_name=chapter_name)
    assert stats_result.word_count == 8
    assert stats_result.paragraph_count == 2
