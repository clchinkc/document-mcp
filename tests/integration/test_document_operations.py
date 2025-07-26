"""Integration tests for document-level operations in the Document MCP tool server."""

from pathlib import Path

# Import tool functions from mcp_client
from document_mcp.mcp_client import create_document
from document_mcp.mcp_client import delete_document
from document_mcp.mcp_client import get_statistics
from document_mcp.mcp_client import list_documents
from document_mcp.mcp_client import read_content
from document_mcp.mcp_client import read_document_summary


def test_create_and_list_document(temp_docs_root: Path):
    """Test creating a document and verifying it's listed."""
    doc_name = "new_doc"

    assert list_documents() == []

    create_result = create_document(doc_name)
    assert create_result.success is True

    docs_list = list_documents()
    assert len(docs_list) == 1
    assert docs_list[0].document_name == doc_name
    assert (temp_docs_root / doc_name).is_dir()


def test_delete_document(document_factory):
    """Test deleting a document."""
    doc_name = "doc_to_delete"
    document_factory(doc_name, {"chap1.md": "content"})

    assert len(list_documents()) == 1

    delete_result = delete_document(doc_name)
    assert delete_result.success is True

    assert list_documents() == []


def test_read_full_document(document_factory):
    """Test reading a full document with multiple chapters."""
    doc_name = "full_doc_test"
    chapters = {
        "01-intro.md": "Introduction content.",
        "02-body.md": "Body content.",
    }
    document_factory(doc_name, chapters)

    full_doc = read_content(doc_name, scope="document")

    assert full_doc is not None
    assert full_doc.document_name == doc_name
    assert len(full_doc.chapters) == 2
    assert full_doc.chapters[0].chapter_name == "01-intro.md"
    assert full_doc.chapters[0].content == "Introduction content."
    assert full_doc.chapters[1].chapter_name == "02-body.md"
    assert full_doc.chapters[1].content == "Body content."


def test_document_statistics(document_factory):
    """Test getting statistics for a document."""
    doc_name = "stats_doc"
    chapters = {
        "chap1.md": "One two three.",
        "chap2.md": "Four five six seven.",
    }
    document_factory(doc_name, chapters)

    stats_result = get_statistics(doc_name, scope="document")
    assert stats_result.chapter_count == 2
    assert stats_result.word_count == 7
    assert stats_result.paragraph_count == 2


def test_read_document_summary(document_factory):
    """Test reading a document's summary file."""
    doc_name = "summary_doc"
    summary_content = "# Summary\n\nThis is the summary."
    doc_path = document_factory(doc_name)
    (doc_path / "_SUMMARY.md").write_text(summary_content, encoding="utf-8")

    result = read_document_summary(doc_name)
    assert result is not None
    assert result.document_name == doc_name
    assert result.content == summary_content
