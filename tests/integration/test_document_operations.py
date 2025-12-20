"""Integration tests for document-level operations in the Document MCP tool server."""

from pathlib import Path

# Import tool functions from mcp_client
from document_mcp.mcp_client import create_document
from document_mcp.mcp_client import delete_document
from document_mcp.mcp_client import get_statistics
from document_mcp.mcp_client import list_documents
from document_mcp.mcp_client import list_summaries
from document_mcp.mcp_client import read_content
from document_mcp.mcp_client import read_summary
from document_mcp.mcp_client import write_summary


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
    assert full_doc.scope == "document"
    assert full_doc.content is not None
    # Content should include both chapters (since total is small)
    assert "Introduction content." in full_doc.content
    assert "Body content." in full_doc.content
    # Check pagination metadata
    assert full_doc.pagination.page == 1
    assert full_doc.pagination.total_characters > 0


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
    """Test reading a document's summary file using new scoped tools."""
    doc_name = "summary_doc"
    summary_content = "# Summary\n\nThis is the summary."
    doc_path = document_factory(doc_name)

    # Create new organized summary structure
    summaries_dir = doc_path / "summaries"
    summaries_dir.mkdir(exist_ok=True)
    (summaries_dir / "document.md").write_text(summary_content, encoding="utf-8")

    result = read_summary(doc_name, scope="document")
    assert result is not None
    assert result.document_name == doc_name
    assert result.content == summary_content
    assert result.scope == "document"


def test_write_and_read_summary_document_scope(document_factory):
    """Test writing and reading document summary using new scoped tools."""
    doc_name = "scoped_summary_doc"
    document_factory(doc_name)

    summary_content = "# Document Summary\n\nThis document covers advanced topics."

    # Write document summary
    result = write_summary(doc_name, summary_content, scope="document")
    assert result.success is True
    assert "document summary" in result.message

    # Read document summary
    summary = read_summary(doc_name, scope="document")
    assert summary is not None
    assert summary.document_name == doc_name
    assert summary.content == summary_content
    assert summary.scope == "document"
    assert summary.target_name is None


def test_write_and_read_summary_chapter_scope(document_factory):
    """Test writing and reading chapter summary using new scoped tools."""
    doc_name = "chapter_summary_doc"
    chapters = {"01-intro.md": "# Introduction\n\nWelcome to the book."}
    document_factory(doc_name, chapters)

    summary_content = "# Chapter Summary\n\nThis chapter introduces the main concepts."

    # Write chapter summary
    result = write_summary(doc_name, summary_content, scope="chapter", target_name="01-intro.md")
    assert result.success is True
    assert "chapter summary for '01-intro.md'" in result.message

    # Read chapter summary
    summary = read_summary(doc_name, scope="chapter", target_name="01-intro.md")
    assert summary is not None
    assert summary.document_name == doc_name
    assert summary.content == summary_content
    assert summary.scope == "chapter"
    assert summary.target_name == "01-intro.md"


def test_write_and_read_summary_section_scope(document_factory):
    """Test writing and reading section summary using new scoped tools."""
    doc_name = "section_summary_doc"
    document_factory(doc_name)

    summary_content = "# Section Summary\n\nThis section covers fundamental principles."

    # Write section summary
    result = write_summary(doc_name, summary_content, scope="section", target_name="fundamentals")
    assert result.success is True
    assert "section summary for 'fundamentals'" in result.message

    # Read section summary
    summary = read_summary(doc_name, scope="section", target_name="fundamentals")
    assert summary is not None
    assert summary.document_name == doc_name
    assert summary.content == summary_content
    assert summary.scope == "section"
    assert summary.target_name == "fundamentals"


def test_list_summaries_multiple_types(document_factory):
    """Test listing multiple summary files of different types."""
    doc_name = "multi_summary_doc"
    chapters = {"01-intro.md": "# Introduction\n\nWelcome."}
    document_factory(doc_name, chapters)

    # Write different types of summaries
    write_summary(doc_name, "Document overview", scope="document")
    write_summary(doc_name, "Chapter overview", scope="chapter", target_name="01-intro.md")
    write_summary(doc_name, "Section overview", scope="section", target_name="concepts")

    # List all summaries
    summaries = list_summaries(doc_name)
    assert len(summaries) == 3
    assert "document.md" in summaries
    assert "chapter-01-intro.md" in summaries
    assert "section-concepts.md" in summaries
    # Should be sorted alphabetically
    assert summaries == sorted(summaries)


def test_read_summary_nonexistent(document_factory):
    """Test reading non-existent summary returns None."""
    doc_name = "empty_summary_doc"
    document_factory(doc_name)

    # Try to read non-existent document summary
    result = read_summary(doc_name, scope="document")
    assert result is None

    # Try to read non-existent chapter summary
    result = read_summary(doc_name, scope="chapter", target_name="01-intro.md")
    assert result is None


def test_write_summary_invalid_document(document_factory):
    """Test writing summary to non-existent document."""
    result = write_summary("nonexistent_doc", "Some content", scope="document")
    assert result.success is False
    assert "not found" in result.message


def test_list_summaries_empty(document_factory):
    """Test listing summaries for document with no summaries."""
    doc_name = "no_summaries_doc"
    document_factory(doc_name)

    summaries = list_summaries(doc_name)
    assert summaries == []


def test_summaries_directory_auto_creation(document_factory):
    """Test that summaries directory is automatically created."""
    import os
    from pathlib import Path

    doc_name = "auto_create_doc"
    document_factory(doc_name)

    # Get document root from environment or default
    doc_root = os.environ.get("DOCUMENT_ROOT_DIR", ".documents_storage")
    doc_path = Path(doc_root) / doc_name
    summaries_path = doc_path / "summaries"

    # Ensure summaries directory doesn't exist initially
    if summaries_path.exists():
        summaries_path.rmdir()

    # Write a summary - should auto-create directory
    result = write_summary(doc_name, "Auto-created summary", scope="document")
    assert result.success is True

    # Verify directory was created
    assert summaries_path.exists()
    assert summaries_path.is_dir()

    # Verify file was created
    summary_file = summaries_path / "document.md"
    assert summary_file.exists()
    assert summary_file.read_text(encoding="utf-8") == "Auto-created summary"
