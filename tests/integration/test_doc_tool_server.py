"""Integration tests for the Document MCP tool server.

Focuses on tool interactions and file system state changes.
"""

from pathlib import Path

import pytest

from document_mcp.mcp_client import add_paragraph
from document_mcp.mcp_client import create_chapter
from document_mcp.mcp_client import create_document
from document_mcp.mcp_client import delete_chapter
from document_mcp.mcp_client import delete_document
from document_mcp.mcp_client import delete_paragraph
from document_mcp.mcp_client import find_text
from document_mcp.mcp_client import get_statistics
from document_mcp.mcp_client import list_chapters
from document_mcp.mcp_client import list_documents
from document_mcp.mcp_client import list_summaries
from document_mcp.mcp_client import read_content
from document_mcp.mcp_client import read_summary
from document_mcp.mcp_client import replace_paragraph
from document_mcp.mcp_client import replace_text
from document_mcp.mcp_client import write_chapter_content
from document_mcp.mcp_client import write_summary

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
    """Test reading a full document with multiple chapters using pagination."""
    doc_name = "full_doc_test"
    chapters = {
        "01-intro.md": "Introduction content.",
        "02-body.md": "Body content.",
    }
    document_factory(doc_name, chapters)

    # Action: Read the full document (paginated response)
    full_doc = read_content(doc_name, scope="document")

    # Post-condition: Content is correct and includes both chapters
    assert full_doc is not None
    assert full_doc.document_name == doc_name
    assert full_doc.scope == "document"
    assert full_doc.content is not None

    # Check pagination metadata
    assert full_doc.pagination.page == 1
    assert full_doc.pagination.total_characters > 0

    # Content should include both chapters (since total is small)
    assert "Introduction content." in full_doc.content
    assert "Body content." in full_doc.content

    # For small documents, should fit in one page
    assert full_doc.pagination.total_pages == 1
    assert full_doc.pagination.has_more is False


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


def test_comprehensive_summary_workflow_all_scopes(document_factory):
    """Test complete workflow using all three new summary tools across all scopes."""
    doc_name = "comprehensive_summary_test"
    chapters = {
        "01-intro.md": "# Introduction\n\nWelcome to the comprehensive guide.",
        "02-basics.md": "# Basics\n\nFundamental concepts explained.",
        "03-advanced.md": "# Advanced\n\nAdvanced techniques and patterns.",
    }
    document_factory(doc_name, chapters)

    # Test 1: Write summaries using all scopes

    # Document summary
    doc_summary = "# Document Overview\n\nThis guide covers intro, basics, and advanced topics."
    result = write_summary(doc_name, doc_summary, scope="document")
    assert result.success is True
    assert "document summary" in result.message

    # Chapter summaries
    chapter_summaries = {
        "01-intro.md": "# Chapter 1 Summary\n\nIntroduces the main concepts.",
        "02-basics.md": "# Chapter 2 Summary\n\nCovers fundamental principles.",
        "03-advanced.md": "# Chapter 3 Summary\n\nExplores advanced techniques.",
    }

    for chapter_name, chapter_summary in chapter_summaries.items():
        result = write_summary(doc_name, chapter_summary, scope="chapter", target_name=chapter_name)
        assert result.success is True
        assert f"chapter summary for '{chapter_name}'" in result.message

    # Section summaries
    section_summaries = {
        "fundamentals": "# Fundamentals Section\n\nCore principles and concepts.",
        "patterns": "# Patterns Section\n\nCommon design patterns and best practices.",
        "troubleshooting": "# Troubleshooting Section\n\nCommon issues and solutions.",
    }

    for section_name, section_summary in section_summaries.items():
        result = write_summary(doc_name, section_summary, scope="section", target_name=section_name)
        assert result.success is True
        assert f"section summary for '{section_name}'" in result.message

    # Test 2: List all summaries
    summaries = list_summaries(doc_name)
    assert len(summaries) == 7  # 1 document + 3 chapters + 3 sections

    expected_files = [
        "document.md",
        "chapter-01-intro.md",
        "chapter-02-basics.md",
        "chapter-03-advanced.md",
        "section-fundamentals.md",
        "section-patterns.md",
        "section-troubleshooting.md",
    ]
    for expected_file in expected_files:
        assert expected_file in summaries

    # Should be sorted alphabetically
    assert summaries == sorted(summaries)

    # Test 3: Read back all summaries using read_summary

    # Read document summary
    doc_result = read_summary(doc_name, scope="document")
    assert doc_result is not None
    assert doc_result.content == doc_summary
    assert doc_result.scope == "document"
    assert doc_result.target_name is None

    # Read chapter summaries
    for chapter_name, expected_content in chapter_summaries.items():
        chapter_result = read_summary(doc_name, scope="chapter", target_name=chapter_name)
        assert chapter_result is not None
        assert chapter_result.content == expected_content
        assert chapter_result.scope == "chapter"
        assert chapter_result.target_name == chapter_name

    # Read section summaries
    for section_name, expected_content in section_summaries.items():
        section_result = read_summary(doc_name, scope="section", target_name=section_name)
        assert section_result is not None
        assert section_result.content == expected_content
        assert section_result.scope == "section"
        assert section_result.target_name == section_name


def test_summary_tools_error_handling(document_factory):
    """Test error handling across all new summary tools."""
    doc_name = "error_test_doc"
    document_factory(doc_name)

    # Test write_summary error cases

    # Invalid document
    result = write_summary("nonexistent_doc", "content", scope="document")
    assert result.success is False
    assert "not found" in result.message

    # Invalid scope
    result = write_summary(doc_name, "content", scope="invalid")
    assert result.success is False
    assert "Invalid scope" in result.message

    # Missing target_name for chapter scope
    result = write_summary(doc_name, "content", scope="chapter", target_name=None)
    assert result.success is False
    assert "target_name is required" in result.message

    # Missing target_name for section scope
    result = write_summary(doc_name, "content", scope="section", target_name=None)
    assert result.success is False
    assert "target_name is required" in result.message

    # Test read_summary error cases

    # Nonexistent document
    result = read_summary("nonexistent_doc", scope="document")
    assert result is None

    # Nonexistent summary
    result = read_summary(doc_name, scope="document")
    assert result is None

    # Test list_summaries error cases

    # Nonexistent document
    summaries = list_summaries("nonexistent_doc")
    assert summaries == []

    # Document with no summaries
    summaries = list_summaries(doc_name)
    assert summaries == []


def test_summary_tools_file_system_verification(document_factory):
    """Test that summary tools correctly manage file system state."""
    import os
    from pathlib import Path

    doc_name = "filesystem_test_doc"
    document_factory(doc_name)

    # Get document root from environment or default
    doc_root = os.environ.get("DOCUMENT_ROOT_DIR", ".documents_storage")
    doc_path = Path(doc_root) / doc_name
    summaries_path = doc_path / "summaries"

    # Initially no summaries directory
    assert not summaries_path.exists()

    # Write document summary - should create directory and file
    write_summary(doc_name, "Document overview", scope="document")

    assert summaries_path.exists()
    assert summaries_path.is_dir()

    doc_summary_file = summaries_path / "document.md"
    assert doc_summary_file.exists()
    assert doc_summary_file.read_text(encoding="utf-8") == "Document overview"

    # Write chapter summary
    write_summary(doc_name, "Chapter overview", scope="chapter", target_name="01-test.md")

    chapter_summary_file = summaries_path / "chapter-01-test.md"
    assert chapter_summary_file.exists()
    assert chapter_summary_file.read_text(encoding="utf-8") == "Chapter overview"

    # Write section summary
    write_summary(doc_name, "Section overview", scope="section", target_name="concepts")

    section_summary_file = summaries_path / "section-concepts.md"
    assert section_summary_file.exists()
    assert section_summary_file.read_text(encoding="utf-8") == "Section overview"

    # Verify list_summaries reflects file system state
    summaries = list_summaries(doc_name)
    assert len(summaries) == 3
    assert "document.md" in summaries
    assert "chapter-01-test.md" in summaries
    assert "section-concepts.md" in summaries


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
    assert chapters_list[0]["chapter_name"] == chapter_name


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
    read_result = read_content(doc_name, scope="chapter", chapter_name=chapter_name)
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

    stats_result = get_statistics(doc_name, scope="chapter", chapter_name=chapter_name)
    assert stats_result.word_count == 8  # "five" is one word
    assert stats_result.paragraph_count == 2


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


def test_add_paragraph_before(para_doc, temp_docs_root: Path):
    """Test inserting a paragraph before another using add_paragraph."""
    doc_name, chapter_name = para_doc
    result = add_paragraph(doc_name, chapter_name, "Inserted Paragraph.", "before", 1)
    assert result.success is True

    content = (temp_docs_root / doc_name / chapter_name).read_text()
    assert content == "Paragraph 1.\n\nInserted Paragraph.\n\nParagraph 2.\n\nParagraph 3."


def test_add_paragraph_after(para_doc, temp_docs_root: Path):
    """Test inserting a paragraph after another using add_paragraph."""
    doc_name, chapter_name = para_doc
    result = add_paragraph(doc_name, chapter_name, "Inserted Paragraph.", "after", 1)
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


def test_add_paragraph_end(para_doc, temp_docs_root: Path):
    """Test appending a paragraph to the end of a chapter using add_paragraph."""
    doc_name, chapter_name = para_doc
    result = add_paragraph(doc_name, chapter_name, "Appended Paragraph.", "end")
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

    result = replace_text(doc_name, "old text", "new text", scope="chapter", chapter_name=chapter_name)
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
    results = find_text(doc_name, "text to find", scope="chapter", chapter_name=chapter_name)
    assert len(results) == 1
    assert results[0].paragraph_index_in_chapter == 0

    # Case-insensitive find (should not find)
    results_case = find_text(
        doc_name,
        "Text To Find",
        scope="chapter",
        chapter_name=chapter_name,
        case_sensitive=False,
    )
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

    results = find_text(doc_name, "keyword", scope="document")
    assert len(results) == 2
    assert results[0].chapter_name == "chap1.md"
    assert results[1].chapter_name == "chap2.md"


class TestUnifiedReadContent:
    """Integration tests for the unified read_content tool."""

    def test_read_content_document_scope(self, document_factory):
        """Test reading full document using unified read_content tool with pagination."""
        doc_name = "test_unified_doc"
        chapters = {
            "01-intro.md": "# Introduction\n\nWelcome to the document.",
            "02-content.md": "# Content\n\nThis is the main content.",
        }
        document_factory(doc_name, chapters)

        # Test document scope (paginated response)
        result = read_content(doc_name, scope="document")

        assert result is not None
        assert result.document_name == doc_name
        assert result.scope == "document"
        assert result.content is not None
        assert len(result.content) > 0

        # Check pagination metadata
        assert result.pagination.page == 1
        assert result.pagination.page_size == 50000
        assert result.pagination.total_characters > 0
        assert result.pagination.total_pages >= 1

        # Content should include both chapters (since total is small)
        assert "Introduction" in result.content
        assert "Content" in result.content
        assert "Welcome to the document" in result.content
        assert "This is the main content" in result.content

    def test_read_content_chapter_scope(self, document_factory):
        """Test reading specific chapter using unified read_content tool."""
        doc_name = "test_unified_chapter"
        chapters = {
            "chapter1.md": "# Chapter 1\n\nFirst chapter content.",
            "chapter2.md": "# Chapter 2\n\nSecond chapter content.",
        }
        document_factory(doc_name, chapters)

        # Test chapter scope
        result = read_content(doc_name, scope="chapter", chapter_name="chapter1.md")

        assert result is not None
        assert result.document_name == doc_name
        assert result.chapter_name == "chapter1.md"
        assert "First chapter content" in result.content
        assert result.word_count > 0

    def test_read_content_paragraph_scope(self, document_factory):
        """Test reading specific paragraph using unified read_content tool."""
        doc_name = "test_unified_paragraph"
        chapters = {
            "test.md": "# Title\n\nFirst paragraph.\n\nSecond paragraph.\n\nThird paragraph.",
        }
        document_factory(doc_name, chapters)

        # Test paragraph scope
        result = read_content(doc_name, scope="paragraph", chapter_name="test.md", paragraph_index=1)

        assert result is not None
        assert result.document_name == doc_name
        assert result.chapter_name == "test.md"
        assert result.paragraph_index_in_chapter == 1
        assert result.content == "First paragraph."

    def test_read_content_invalid_scope(self, document_factory):
        """Test read_content with invalid scope."""
        doc_name = "test_invalid_scope"
        document_factory(doc_name, {"test.md": "Content"})

        result = read_content(doc_name, scope="invalid_scope")
        assert result is None

    def test_read_content_missing_chapter_name_for_chapter_scope(self, document_factory):
        """Test read_content chapter scope without chapter_name."""
        doc_name = "test_missing_chapter"
        document_factory(doc_name, {"test.md": "Content"})

        result = read_content(doc_name, scope="chapter")
        assert result is None

    def test_read_content_missing_parameters_for_paragraph_scope(self, document_factory):
        """Test read_content paragraph scope without required parameters."""
        doc_name = "test_missing_params"
        document_factory(doc_name, {"test.md": "Content"})

        # Missing both chapter_name and paragraph_index
        result = read_content(doc_name, scope="paragraph")
        assert result is None

        # Missing paragraph_index
        result = read_content(doc_name, scope="paragraph", chapter_name="test.md")
        assert result is None


class TestUnifiedTools:
    """Integration tests for the new unified tools."""

    def test_find_text_document_scope(self, document_factory):
        """Test unified find_text with document scope."""
        doc_name = "test_find_doc"
        chapters = {
            "chapter1.md": "This is the first chapter with important content.",
            "chapter2.md": "Second chapter also has important information.",
            "chapter3.md": "Third chapter with different content.",
        }
        document_factory(doc_name, chapters)

        # Test document scope search
        results = find_text(doc_name, "important", scope="document", case_sensitive=False)

        assert results is not None
        assert len(results) == 2
        # Results should be from chapters 1 and 2
        chapter_names = [r.chapter_name for r in results]
        assert "chapter1.md" in chapter_names
        assert "chapter2.md" in chapter_names

    def test_find_text_chapter_scope(self, document_factory):
        """Test unified find_text with chapter scope."""
        doc_name = "test_find_chapter"
        chapters = {
            "test.md": "This chapter contains multiple instances of the word test. Test again.",
        }
        document_factory(doc_name, chapters)

        # Test chapter scope search
        results = find_text(
            doc_name,
            "test",
            scope="chapter",
            chapter_name="test.md",
            case_sensitive=False,
        )

        assert results is not None
        assert len(results) >= 1  # Should find at least one match

    def test_find_text_invalid_scope(self, document_factory):
        """Test unified find_text with invalid scope."""
        doc_name = "test_find_invalid"
        document_factory(doc_name, {"test.md": "Content"})

        result = find_text(doc_name, "content", scope="invalid")
        assert result is None

    def test_replace_text_document_scope(self, document_factory, temp_docs_root):
        """Test unified replace_text with document scope."""
        doc_name = "test_replace_doc"
        chapters = {
            "chapter1.md": "Replace this old text in chapter 1.",
            "chapter2.md": "Also replace old text in chapter 2.",
        }
        document_factory(doc_name, chapters)

        # Test document scope replacement
        result = replace_text(doc_name, "old text", "new content", scope="document")

        assert result is not None
        assert result.success is True
        assert result.details["total_occurrences_replaced"] == 2

        # Verify actual file changes
        ch1_content = (temp_docs_root / doc_name / "chapter1.md").read_text()
        ch2_content = (temp_docs_root / doc_name / "chapter2.md").read_text()
        assert "new content" in ch1_content
        assert "new content" in ch2_content
        assert "old text" not in ch1_content
        assert "old text" not in ch2_content

    def test_replace_text_chapter_scope(self, document_factory, temp_docs_root):
        """Test unified replace_text with chapter scope."""
        doc_name = "test_replace_chapter"
        chapters = {
            "target.md": "Replace this specific text only in this chapter.",
            "other.md": "Do not replace this specific text in this chapter.",
        }
        document_factory(doc_name, chapters)

        # Test chapter scope replacement
        result = replace_text(
            doc_name,
            "specific text",
            "modified text",
            scope="chapter",
            chapter_name="target.md",
        )

        assert result is not None
        assert result.success is True
        assert result.details["occurrences_replaced"] == 1

        # Verify only target chapter was modified
        target_content = (temp_docs_root / doc_name / "target.md").read_text()
        other_content = (temp_docs_root / doc_name / "other.md").read_text()
        assert "modified text" in target_content
        assert "specific text" not in target_content
        assert "specific text" in other_content  # Should remain unchanged

    def test_get_statistics_document_scope(self, document_factory):
        """Test unified get_statistics with document scope."""
        doc_name = "test_stats_doc"
        chapters = {
            "ch1.md": "Chapter one has four words.",  # Actually: Chapter(1) one(2) has(3) four(4) words(5) = 5 words
            "ch2.md": "Chapter two has five total words.",  # Actually: Chapter(1) two(2) has(3) five(4) total(5) words(6) = 6 words
        }
        document_factory(doc_name, chapters)

        # Test document scope statistics
        result = get_statistics(doc_name, scope="document")

        assert result is not None
        assert result.scope.startswith("document:")
        assert result.chapter_count == 2
        assert result.word_count == 11  # 5 + 6 words total (corrected count)
        assert result.paragraph_count == 2

    def test_get_statistics_chapter_scope(self, document_factory):
        """Test unified get_statistics with chapter scope."""
        doc_name = "test_stats_chapter"
        chapters = {
            "target.md": "This chapter has exactly five words.",  # Actually: This(1) chapter(2) has(3) exactly(4) five(5) words(6) = 6 words
            "other.md": "This other chapter has more than five words total.",
        }
        document_factory(doc_name, chapters)

        # Test chapter scope statistics
        result = get_statistics(doc_name, scope="chapter", chapter_name="target.md")

        assert result is not None
        assert result.scope.endswith("target.md")
        assert result.word_count == 6  # Corrected count
        assert result.paragraph_count == 1
        # Note: chapter_count not checked since it's not accessible on Pydantic model for chapter scope

    def test_unified_tools_with_missing_parameters(self, document_factory):
        """Test unified tools with missing required parameters."""
        doc_name = "test_missing_params"
        document_factory(doc_name, {"test.md": "Content"})

        # Test find_text with missing chapter_name for chapter scope
        result = find_text(doc_name, "content", scope="chapter")
        assert result is None

        # Test replace_text with missing chapter_name for chapter scope
        result = replace_text(doc_name, "old", "new", scope="chapter")
        assert result is None

        # Test get_statistics with missing chapter_name for chapter scope
        result = get_statistics(doc_name, scope="chapter")
        assert result is None


# ===================================
# Token Limit Optimization Tests
# ===================================


def test_list_documents_include_chapters_parameter(document_factory):
    """Test list_documents with include_chapters parameter for token optimization."""
    doc_name = "chapters_test"
    chapters = {
        "01-intro.md": "Introduction content with multiple words here.",
        "02-body.md": "Body content with even more words here for testing.",
        "03-conclusion.md": "Conclusion content wrapping up the document.",
    }
    document_factory(doc_name, chapters)

    # Test default behavior (include_chapters=False)
    docs_fast = list_documents()
    assert len(docs_fast) == 1
    doc_info = docs_fast[0]
    assert doc_info.document_name == doc_name
    assert doc_info.total_chapters == 3  # Count is always shown
    assert doc_info.chapters == []  # But chapters list is empty for fast response

    # Test with include_chapters=True
    docs_detailed = list_documents(include_chapters=True)
    assert len(docs_detailed) == 1
    doc_info_detailed = docs_detailed[0]
    assert doc_info_detailed.document_name == doc_name
    assert doc_info_detailed.total_chapters == 3
    assert len(doc_info_detailed.chapters) == 3  # Full chapter metadata included
    assert doc_info_detailed.chapters[0].chapter_name == "01-intro.md"
    assert doc_info_detailed.chapters[1].chapter_name == "02-body.md"
    assert doc_info_detailed.chapters[2].chapter_name == "03-conclusion.md"


def test_read_content_pagination_parameter(document_factory):
    """Test read_content with pagination parameters for precise token optimization."""
    doc_name = "large_doc_test"
    chapters = {}
    for i in range(1, 6):
        chapters[f"{i:02d}-small.md"] = f"Short chapter {i} content."

    for i in range(6, 11):
        chapters[f"{i:02d}-large.md"] = f"Large chapter {i} content. " * 20 + f"This is chapter {i}."

    document_factory(doc_name, chapters)

    # Test default pagination
    doc_content_default = read_content(doc_name, scope="document")
    assert doc_content_default is not None
    assert doc_content_default.document_name == doc_name
    assert doc_content_default.pagination.page == 1
    assert doc_content_default.pagination.page_size == 50000
    assert doc_content_default.pagination.total_pages == 1
    assert doc_content_default.pagination.has_more is False

    # Test small page size
    doc_content_small = read_content(doc_name, scope="document", page=1, page_size=200)
    assert doc_content_small is not None
    assert doc_content_small.pagination.page == 1
    assert doc_content_small.pagination.page_size == 200
    assert len(doc_content_small.content) <= 200
    assert doc_content_small.pagination.total_pages > 1
    assert doc_content_small.pagination.has_more is True

    # Test second page
    doc_content_page2 = read_content(doc_name, scope="document", page=2, page_size=200)
    assert doc_content_page2 is not None
    assert doc_content_page2.pagination.page == 2
    assert doc_content_page2.pagination.has_previous is True
    assert len(doc_content_page2.content) <= 200

    assert doc_content_small.content != doc_content_page2.content

    # Test with medium page size
    doc_content_medium = read_content(doc_name, scope="document", page=1, page_size=1000)
    assert doc_content_medium is not None
    assert len(doc_content_medium.content) <= 1000

    assert len(doc_content_medium.content) > len(doc_content_small.content)

    # Test invalid parameters
    doc_content_invalid_page = read_content(doc_name, scope="document", page=0)
    assert doc_content_invalid_page is None  # Should return None for invalid page

    doc_content_invalid_size = read_content(doc_name, scope="document", page_size=0)
    assert doc_content_invalid_size is None  # Should return None for invalid page_size


def test_read_content_pagination_behavior(document_factory):
    """Test read_content pagination behavior with page boundaries."""
    doc_name = "pagination_test"
    chapters = {
        "01-exact.md": "A" * 100,
        "02-more.md": "B" * 200,
    }
    document_factory(doc_name, chapters)

    doc_content = read_content(doc_name, scope="document", page=1, page_size=150)
    assert doc_content is not None
    assert doc_content.pagination.page == 1
    assert doc_content.pagination.page_size == 150
    assert len(doc_content.content) == 150

    total_content_size = 100 + 2 + 200
    expected_pages = (total_content_size + 149) // 150
    assert doc_content.pagination.total_pages == expected_pages
    assert doc_content.pagination.has_more is True

    # Test second page
    doc_content_page2 = read_content(doc_name, scope="document", page=2, page_size=150)
    assert doc_content_page2 is not None
    assert doc_content_page2.pagination.page == 2
    assert doc_content_page2.pagination.has_previous is True

    # Content should be different between pages
    assert doc_content.content != doc_content_page2.content

    # Test last page
    last_page = doc_content.pagination.total_pages
    doc_content_last = read_content(doc_name, scope="document", page=last_page, page_size=150)
    assert doc_content_last is not None
    assert doc_content_last.pagination.has_more is False
    assert doc_content_last.pagination.next_page is None


def test_read_content_pagination_edge_cases(document_factory):
    """Test read_content pagination edge cases and boundary conditions."""
    doc_name = "edge_case_test"

    # Test empty document
    document_factory(doc_name, {})
    empty_content = read_content(doc_name, scope="document")
    assert empty_content is not None
    assert empty_content.content == ""
    assert empty_content.pagination.total_pages == 1
    assert empty_content.pagination.has_more is False

    # Test single character document
    single_char_doc = "single_char_doc"
    document_factory(single_char_doc, {"01-single.md": "X"})
    single_content = read_content(single_char_doc, scope="document", page_size=1)
    assert single_content is not None
    assert single_content.content == "X"
    assert single_content.pagination.total_pages == 1
    assert single_content.pagination.has_more is False

    # Test exact page boundary
    boundary_doc = "boundary_doc"
    document_factory(boundary_doc, {"01-boundary.md": "A" * 100})
    boundary_content = read_content(boundary_doc, scope="document", page_size=100)
    assert boundary_content is not None
    assert len(boundary_content.content) == 100
    assert boundary_content.pagination.total_pages == 1
    assert boundary_content.pagination.has_more is False

    # Test page boundary + 1
    boundary_plus_doc = "boundary_plus_doc"
    document_factory(boundary_plus_doc, {"01-boundary.md": "A" * 101})
    boundary_plus_content = read_content(boundary_plus_doc, scope="document", page_size=100)
    assert boundary_plus_content is not None
    assert len(boundary_plus_content.content) == 100
    assert boundary_plus_content.pagination.total_pages == 2
    assert boundary_plus_content.pagination.has_more is True

    # Test second page of boundary + 1
    boundary_plus_page2 = read_content(boundary_plus_doc, scope="document", page=2, page_size=100)
    assert boundary_plus_page2 is not None
    assert len(boundary_plus_page2.content) == 1
    assert boundary_plus_page2.content == "A"
    assert boundary_plus_page2.pagination.has_more is False


def test_read_content_pagination_out_of_bounds(document_factory):
    """Test read_content pagination with out-of-bounds page requests."""
    doc_name = "bounds_test"
    document_factory(doc_name, {"01-test.md": "Test content"})

    # Test page beyond available pages
    out_of_bounds = read_content(doc_name, scope="document", page=999, page_size=100)
    assert out_of_bounds is None  # Should return None for out of bounds

    # Test page 0 (invalid)
    page_zero = read_content(doc_name, scope="document", page=0)
    assert page_zero is None  # Should return None for invalid page

    # Test negative page
    negative_page = read_content(doc_name, scope="document", page=-1)
    assert negative_page is None  # Should return None for negative page


def test_read_content_pagination_large_document(document_factory):
    """Test pagination behavior with larger documents to validate scalability."""
    import time

    doc_name = "large_doc_test"
    # Create a reasonably large document
    chapters = {}
    for i in range(1, 11):  # 10 chapters
        # Each chapter ~2KB = 20KB total document
        chapters[f"{i:02d}-chapter.md"] = f"Chapter {i} content. " * 100 + f"End of chapter {i}."

    document_factory(doc_name, chapters)

    start_time = time.time()
    page1 = read_content(doc_name, scope="document", page=1, page_size=5000)
    page1_time = time.time() - start_time

    start_time = time.time()
    page2 = read_content(doc_name, scope="document", page=2, page_size=5000)
    page2_time = time.time() - start_time

    assert page1 is not None
    assert page2 is not None
    assert page1.pagination.total_characters > 15000
    assert page1.pagination.total_pages > 3
    assert page1.pagination.has_more is True
    assert page2.pagination.has_previous is True
    assert len(page1.content) == 5000
    assert len(page2.content) == 5000
    assert page1.content != page2.content

    assert page1_time < 1.0, f"Page 1 took too long: {page1_time:.2f}s"
    assert page2_time < 1.0, f"Page 2 took too long: {page2_time:.2f}s"

    last_page = page1.pagination.total_pages
    start_time = time.time()
    page_last = read_content(doc_name, scope="document", page=last_page, page_size=5000)
    last_page_time = time.time() - start_time

    assert page_last is not None
    assert page_last.pagination.has_more is False
    assert page_last.pagination.next_page is None
    assert last_page_time < 1.0, f"Last page took too long: {last_page_time:.2f}s"


def test_find_text_max_results_parameter(document_factory):
    """Test find_text with max_results parameter for token optimization."""
    doc_name = "search_test"
    # Create chapters with repeated search term
    chapters = {}
    for i in range(1, 6):  # 5 chapters
        chapters[f"chapter{i}.md"] = (
            f"Chapter {i} has the word test multiple times. Test again. And test once more."
        )
    document_factory(doc_name, chapters)

    # Test default behavior (max_results=100) - should get all results
    results_default = find_text(doc_name, "test", scope="document")
    assert results_default is not None
    total_results = len(results_default)
    assert total_results >= 5  # Should have at least 5 results (case-insensitive search)

    # Test with limited max_results
    results_limited = find_text(doc_name, "test", scope="document", max_results=5)
    assert results_limited is not None
    assert len(results_limited) == 5  # Limited to 5 results

    # Test with max_results=1
    results_single = find_text(doc_name, "test", scope="document", max_results=1)
    assert results_single is not None
    assert len(results_single) == 1

    # Test chapter scope with max_results
    results_chapter = find_text(doc_name, "test", scope="chapter", chapter_name="chapter1.md", max_results=2)
    assert results_chapter is not None
    assert len(results_chapter) <= 2  # Limited to at most 2 results from single chapter

    # Test invalid max_results
    results_invalid = find_text(doc_name, "test", scope="document", max_results=0)
    assert results_invalid is None  # Should return None for invalid limit

    results_negative = find_text(doc_name, "test", scope="document", max_results=-1)
    assert results_negative is None  # Should return None for negative limit
