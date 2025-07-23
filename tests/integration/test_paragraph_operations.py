"""Integration tests for paragraph-level operations in the Document MCP tool server."""

from pathlib import Path

import pytest

from document_mcp.mcp_client import append_paragraph_to_chapter
from document_mcp.mcp_client import delete_document
from document_mcp.mcp_client import delete_paragraph
from document_mcp.mcp_client import find_text
from document_mcp.mcp_client import insert_paragraph_after
from document_mcp.mcp_client import insert_paragraph_before
from document_mcp.mcp_client import replace_paragraph
from document_mcp.mcp_client import replace_text
from tests.shared.fixtures import document_factory




@pytest.fixture
def para_doc(document_factory):
    """Fixture for a document with paragraphs for manipulation tests."""
    doc_name = "para_doc"
    chapter_name = "chap1.md"
    content = "Paragraph 1.\n\nParagraph 2.\n\nParagraph 3."
    document_factory(doc_name, {chapter_name: content})
    return doc_name, chapter_name


def test_replace_paragraph(para_doc, temp_docs_root: Path):
    """Test replacing a paragraph."""
    doc_name, chapter_name = para_doc
    result = replace_paragraph(doc_name, chapter_name, 1, "New Paragraph 2.")
    assert result.success is True

    content = (temp_docs_root / doc_name / chapter_name).read_text()
    assert content == "Paragraph 1.\n\nNew Paragraph 2.\n\nParagraph 3."


def test_insert_paragraph_before(para_doc, temp_docs_root: Path):
    """Test inserting a paragraph before another."""
    doc_name, chapter_name = para_doc
    result = insert_paragraph_before(doc_name, chapter_name, 1, "Inserted Paragraph.")
    assert result.success is True

    content = (temp_docs_root / doc_name / chapter_name).read_text()
    assert (
        content == "Paragraph 1.\n\nInserted Paragraph.\n\nParagraph 2.\n\nParagraph 3."
    )


def test_insert_paragraph_after(para_doc, temp_docs_root: Path):
    """Test inserting a paragraph after another."""
    doc_name, chapter_name = para_doc
    result = insert_paragraph_after(doc_name, chapter_name, 1, "Inserted Paragraph.")
    assert result.success is True

    content = (temp_docs_root / doc_name / chapter_name).read_text()
    assert (
        content == "Paragraph 1.\n\nParagraph 2.\n\nInserted Paragraph.\n\nParagraph 3."
    )


def test_delete_paragraph(para_doc, temp_docs_root: Path):
    """Test deleting a paragraph."""
    doc_name, chapter_name = para_doc
    result = delete_paragraph(doc_name, chapter_name, 1)
    assert result.success is True

    content = (temp_docs_root / doc_name / chapter_name).read_text()
    assert content == "Paragraph 1.\n\nParagraph 3."


def test_append_paragraph_to_chapter(para_doc, temp_docs_root: Path):
    """Test appending a paragraph to a chapter."""
    doc_name, chapter_name = para_doc
    result = append_paragraph_to_chapter(doc_name, chapter_name, "Appended Paragraph.")
    assert result.success is True

    content = (temp_docs_root / doc_name / chapter_name).read_text()
    assert (
        content == "Paragraph 1.\n\nParagraph 2.\n\nParagraph 3.\n\nAppended Paragraph."
    )


def test_replace_text_in_chapter(document_factory, temp_docs_root: Path):
    """Test replacing text within a single chapter."""
    doc_name = "replace_doc"
    chapter_name = "chap.md"
    content = "The old text needs to be replaced. The old text is here."
    document_factory(doc_name, {chapter_name: content})

    result = replace_text(
        doc_name, "old text", "new text", scope="chapter", chapter_name=chapter_name
    )
    assert result.success is True
    assert result.details["occurrences_replaced"] == 2

    final_content = (temp_docs_root / doc_name / chapter_name).read_text()
    assert final_content == "The new text needs to be replaced. The new text is here."


def test_replace_text_in_document(document_factory, temp_docs_root: Path):
    """Test replacing text across an entire document."""
    doc_name = "replace_doc_full"
    chapters = {
        "chap1.md": "Replace this value.",
        "chap2.md": "This value needs replacement too.",
    }
    document_factory(doc_name, chapters)

    result = replace_text(doc_name, "value", "term", scope="document")
    assert result.success is True
    assert result.details["total_occurrences_replaced"] == 2

    chap1_content = (temp_docs_root / doc_name / "chap1.md").read_text()
    assert chap1_content == "Replace this term."

    chap2_content = (temp_docs_root / doc_name / "chap2.md").read_text()
    assert chap2_content == "This term needs replacement too."


def test_find_text_in_chapter(document_factory):
    """Test finding text within a single chapter."""
    doc_name = "search_doc"
    chapter_name = "chap.md"
    content = "Find this text. And find this too. Nothing else here."
    document_factory(doc_name, {chapter_name: content})

    results = find_text(
        doc_name,
        "find",
        scope="chapter",
        chapter_name=chapter_name,
        case_sensitive=False,
    )
    assert len(results) >= 1
    assert results[0].chapter_name == chapter_name


def test_find_text_in_document(document_factory):
    """Test finding text across an entire document."""
    doc_name = "search_doc_full"
    chapters = {
        "chap1.md": "Search for this keyword.",
        "chap2.md": "This chapter has the keyword too.",
    }
    document_factory(doc_name, chapters)

    results = find_text(doc_name, "keyword", scope="document")
    assert len(results) == 2
    assert results[0].chapter_name == "chap1.md"
    assert results[1].chapter_name == "chap2.md"
