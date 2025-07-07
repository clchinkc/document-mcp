"""
Integration tests for the Document MCP tool server, focusing on tool interactions
and file system state changes.
"""
import pytest
from pathlib import Path
from document_mcp.doc_tool_server import (
    create_document,
    create_chapter,
    list_documents,
    list_chapters,
    read_chapter_content,
    delete_document,
    delete_chapter,
    write_chapter_content,
    replace_paragraph,
    insert_paragraph_before,
    insert_paragraph_after,
    delete_paragraph,
    append_paragraph_to_chapter,
    # move_paragraph_before, # To be tested
    # move_paragraph_to_end, # To be tested
    replace_text_in_chapter,
    replace_text_in_document,
    get_chapter_statistics,
    get_document_statistics,
    find_text_in_chapter,
    find_text_in_document,
    read_document_summary,
    read_full_document,
)

@pytest.fixture
def document_factory(temp_docs_root: Path):
    """A factory to create documents with chapters for testing."""
    created_docs = []
    def _create_document(doc_name: str, chapters: dict[str, str] = None):
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

# ===================================
# Document-Level Tests
# ===================================

def test_create_and_list_document(temp_docs_root: Path):
    """Test creating a document and verifying it's listed."""
    doc_name = "new_doc"
    
    # Pre-condition: No documents exist
    assert list_documents() == []
    
    # Action: Create a document
    create_result = create_document(doc_name)
    assert create_result.success is True
    
    # Post-condition: Document is listed correctly
    docs_list = list_documents()
    assert len(docs_list) == 1
    assert docs_list[0].document_name == doc_name
    assert (temp_docs_root / doc_name).is_dir()

def test_delete_document(document_factory):
    """Test deleting a document."""
    doc_name = "doc_to_delete"
    document_factory(doc_name, {"chap1.md": "content"})
    
    # Pre-condition: Document exists
    assert len(list_documents()) == 1
    
    # Action: Delete the document
    delete_result = delete_document(doc_name)
    assert delete_result.success is True
    
    # Post-condition: Document is gone
    assert list_documents() == []

def test_read_full_document(document_factory):
    """Test reading a full document with multiple chapters."""
    doc_name = "full_doc_test"
    chapters = {
        "01-intro.md": "Introduction content.",
        "02-body.md": "Body content.",
    }
    document_factory(doc_name, chapters)
    
    # Action: Read the full document
    full_doc = read_full_document(doc_name)
    
    # Post-condition: Content is correct and ordered
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
    
    stats = get_document_statistics(doc_name)
    assert stats.chapter_count == 2
    assert stats.word_count == 7
    assert stats.paragraph_count == 2

def test_read_document_summary(document_factory):
    """Test reading a document's summary file."""
    doc_name = "summary_doc"
    summary_content = "# Summary\n\nThis is the summary."
    doc_path = document_factory(doc_name)
    (doc_path / "_SUMMARY.md").write_text(summary_content)
    
    result = read_document_summary(doc_name)
    assert result is not None
    assert result.document_name == doc_name
    assert result.content == summary_content

# ===================================
# Chapter-Level Tests
# ===================================

def test_create_and_list_chapter(document_factory):
    """Test creating a chapter and verifying it's listed."""
    doc_name = "chapter_test_doc"
    chapter_name = "new_chap.md"
    document_factory(doc_name)
    
    # Pre-condition: No chapters exist
    assert list_chapters(doc_name) == []
    
    # Action: Create a chapter
    create_result = create_chapter(doc_name, chapter_name, "Initial content.")
    assert create_result.success is True
    
    # Post-condition: Chapter is listed correctly
    chapters_list = list_chapters(doc_name)
    assert len(chapters_list) == 1
    assert chapters_list[0].chapter_name == chapter_name

def test_delete_chapter(document_factory):
    """Test deleting a chapter."""
    doc_name = "doc_with_chap_to_delete"
    chapter_name = "chap_to_delete.md"
    document_factory(doc_name, {chapter_name: "content"})
    
    # Pre-condition: Chapter exists
    assert len(list_chapters(doc_name)) == 1
    
    # Action: Delete the chapter
    delete_result = delete_chapter(doc_name, chapter_name)
    assert delete_result.success is True
    
    # Post-condition: Chapter is gone
    assert list_chapters(doc_name) == []

def test_read_and_write_chapter_content(document_factory, temp_docs_root: Path):
    """Test reading and then overwriting chapter content."""
    doc_name = "read_write_doc"
    chapter_name = "chap.md"
    initial_content = "This is the original content."
    document_factory(doc_name, {chapter_name: initial_content})
    
    # Read initial content
    read_result = read_chapter_content(doc_name, chapter_name)
    assert read_result.content == initial_content
    
    # Write new content
    new_content = "This is the new, updated content."
    write_result = write_chapter_content(doc_name, chapter_name, new_content)
    assert write_result.success is True
    
    # Verify new content was written
    final_content = (temp_docs_root / doc_name / chapter_name).read_text()
    assert final_content == new_content

def test_chapter_statistics(document_factory):
    """Test getting statistics for a chapter."""
    doc_name = "chap_stats_doc"
    chapter_name = "chap.md"
    content = "This chapter has five words.\n\nAnd two paragraphs."
    document_factory(doc_name, {chapter_name: content})
    
    stats = get_chapter_statistics(doc_name, chapter_name)
    assert stats.word_count == 8 # "five" is one word
    assert stats.paragraph_count == 2

# ===================================
# Paragraph-Level Tests
# ===================================

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
    assert content == "Paragraph 1.\n\nInserted Paragraph.\n\nParagraph 2.\n\nParagraph 3."

def test_insert_paragraph_after(para_doc, temp_docs_root: Path):
    """Test inserting a paragraph after another."""
    doc_name, chapter_name = para_doc
    result = insert_paragraph_after(doc_name, chapter_name, 1, "Inserted Paragraph.")
    assert result.success is True
    
    content = (temp_docs_root / doc_name / chapter_name).read_text()
    assert content == "Paragraph 1.\n\nParagraph 2.\n\nInserted Paragraph.\n\nParagraph 3."

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
    assert content == "Paragraph 1.\n\nParagraph 2.\n\nParagraph 3.\n\nAppended Paragraph."

# ===================================
# Search and Replace Tests
# ===================================

def test_replace_text_in_chapter(document_factory, temp_docs_root: Path):
    """Test replacing text within a single chapter."""
    doc_name = "replace_doc"
    chapter_name = "chap.md"
    content = "The old text needs to be replaced. The old text is here."
    document_factory(doc_name, {chapter_name: content})
    
    result = replace_text_in_chapter(doc_name, chapter_name, "old text", "new text")
    assert result.success is True
    assert result.details['occurrences_replaced'] == 2
    
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
    
    result = replace_text_in_document(doc_name, "value", "term")
    assert result.success is True
    assert result.details['total_occurrences_replaced'] == 2
    
    chap1_content = (temp_docs_root / doc_name / "chap1.md").read_text()
    chap2_content = (temp_docs_root / doc_name / "chap2.md").read_text()
    assert chap1_content == "Replace this term."
    assert chap2_content == "This term needs replacement too."

def test_find_text_in_chapter(document_factory):
    """Test finding text within a chapter (case-sensitive)."""
    doc_name = "find_doc"
    chapter_name = "chap.md"
    content = "Here is the text to find. Find this text."
    document_factory(doc_name, {chapter_name: content})
    
    # Case-sensitive find
    results = find_text_in_chapter(doc_name, chapter_name, "text to find")
    assert len(results) == 1
    assert results[0].paragraph_index_in_chapter == 0
    
    # Case-insensitive find (should not find)
    results_case = find_text_in_chapter(doc_name, chapter_name, "Text To Find", case_sensitive=False)
    assert len(results_case) == 1
    assert results_case[0].paragraph_index_in_chapter == 0

def test_find_text_in_document(document_factory):
    """Test finding text across an entire document."""
    doc_name = "find_doc_full"
    chapters = {
        "chap1.md": "The keyword is here.",
        "chap2.md": "Another mention of the keyword.",
        "chap3.md": "No mention here.",
    }
    document_factory(doc_name, chapters)
    
    results = find_text_in_document(doc_name, "keyword")
    assert len(results) == 2
    assert results[0].chapter_name == "chap1.md"
    assert results[1].chapter_name == "chap2.md"