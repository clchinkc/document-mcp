"""Integration tests for the Document MCP tool server, focusing on tool interactions
and file system state changes.
"""

from pathlib import Path

import pytest

from document_mcp.mcp_client import append_paragraph_to_chapter
from document_mcp.mcp_client import create_chapter
from document_mcp.mcp_client import create_document
from document_mcp.mcp_client import delete_chapter
from document_mcp.mcp_client import delete_document
from document_mcp.mcp_client import delete_paragraph
from document_mcp.mcp_client import find_text
from document_mcp.mcp_client import get_statistics
from document_mcp.mcp_client import insert_paragraph_after
from document_mcp.mcp_client import insert_paragraph_before
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
    """Test reading a full document with multiple chapters."""
    doc_name = "full_doc_test"
    chapters = {
        "01-intro.md": "Introduction content.",
        "02-body.md": "Body content.",
    }
    document_factory(doc_name, chapters)

    # Action: Read the full document
    full_doc = read_content(doc_name, scope="document")

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
        """Test reading full document using unified read_content tool."""
        doc_name = "test_unified_doc"
        chapters = {
            "01-intro.md": "# Introduction\n\nWelcome to the document.",
            "02-content.md": "# Content\n\nThis is the main content.",
        }
        document_factory(doc_name, chapters)

        # Test document scope
        result = read_content(doc_name, scope="document")

        assert result is not None
        assert result.document_name == doc_name
        assert len(result.chapters) == 2
        assert result.total_word_count > 0

        # Check that chapters are in correct order
        chapter_names = [ch.chapter_name for ch in result.chapters]
        assert chapter_names == ["01-intro.md", "02-content.md"]

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


