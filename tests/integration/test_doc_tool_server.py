"""Integration tests for the Document MCP tool server, focusing on tool interactions
and file system state changes.
"""

from pathlib import Path

import pytest

from document_mcp.mcp_client import append_paragraph_to_chapter
from document_mcp.mcp_client import batch_apply_operations
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
from document_mcp.mcp_client import read_content
from document_mcp.mcp_client import read_document_summary
from document_mcp.mcp_client import replace_paragraph
from document_mcp.mcp_client import replace_text
from document_mcp.mcp_client import write_chapter_content

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


# ===================================
# Search and Replace Tests
# ===================================


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
    results = find_text(
        doc_name, "text to find", scope="chapter", chapter_name=chapter_name
    )
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
        result = read_content(
            doc_name, scope="paragraph", chapter_name="test.md", paragraph_index=1
        )

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

    def test_read_content_missing_chapter_name_for_chapter_scope(
        self, document_factory
    ):
        """Test read_content chapter scope without chapter_name."""
        doc_name = "test_missing_chapter"
        document_factory(doc_name, {"test.md": "Content"})

        result = read_content(doc_name, scope="chapter")
        assert result is None

    def test_read_content_missing_parameters_for_paragraph_scope(
        self, document_factory
    ):
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
        results = find_text(
            doc_name, "important", scope="document", case_sensitive=False
        )

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


class TestBatchOperationsIntegration:
    """Integration tests for batch operations with real MCP server."""

    @pytest.mark.asyncio
    async def test_batch_apply_operations_create_document_and_chapter(
        self, document_factory
    ):
        """Test batch operation to create document and chapter atomically."""
        doc_name = "test_batch_doc"

        # Prepare batch operations
        operations = [
            {
                "operation_type": "create_document",
                "target": {},
                "parameters": {"document_name": doc_name},
                "order": 1,
                "operation_id": "create_doc",
            },
            {
                "operation_type": "create_chapter",
                "target": {"document_name": doc_name},
                "parameters": {
                    "chapter_name": "01-intro.md",
                    "initial_content": "# Introduction\n\nThis is a test chapter created via batch operation.",
                },
                "order": 2,
                "operation_id": "create_chapter",
                "depends_on": ["create_doc"],
            },
        ]

        # Execute batch
        result = batch_apply_operations(
            operations=operations, atomic=True, validate_only=False
        )

        # Verify batch result
        assert result.success is True
        assert result.total_operations == 2
        assert result.successful_operations == 2
        assert result.failed_operations == 0
        assert len(result.operation_results) == 2

        # Verify operations succeeded
        for op_result in result.operation_results:
            assert op_result.success is True

        # Verify actual document and chapter were created
        docs = list_documents()
        doc_names = [doc.document_name for doc in docs]
        assert doc_name in doc_names

        chapters = list_chapters(doc_name)
        assert len(chapters) == 1
        assert chapters[0].chapter_name == "01-intro.md"

        # Verify chapter content
        chapter_content = read_content(
            doc_name, scope="chapter", chapter_name="01-intro.md"
        )
        assert (
            "This is a test chapter created via batch operation"
            in chapter_content.content
        )

    @pytest.mark.asyncio
    async def test_batch_apply_operations_validate_only_mode(self, document_factory):
        """Test batch operation in validation-only mode."""
        operations = [
            {
                "operation_type": "create_document",
                "target": {},
                "parameters": {"document_name": "validate_test_doc"},
                "order": 1,
                "operation_id": "validate_doc",
            }
        ]

        # Execute in validation-only mode
        result = batch_apply_operations(operations=operations, validate_only=True)

        # Verify validation succeeded but no operations were executed
        assert result.success is True
        assert result.total_operations == 1
        assert result.successful_operations == 0  # None executed
        assert result.failed_operations == 0
        assert "Validation successful" in result.summary

        # Verify no actual document was created
        docs = list_documents()
        doc_names = [doc.document_name for doc in docs]
        assert "validate_test_doc" not in doc_names

    @pytest.mark.asyncio
    async def test_batch_apply_operations_atomic_failure(
        self, temp_docs_root, document_factory
    ):
        """Test batch operation atomic failure with rollback."""
        operations = [
            {
                "operation_type": "create_document",
                "target": {},
                "parameters": {"document_name": "atomic_test_doc"},
                "order": 1,
                "operation_id": "create_valid_doc",
            },
            {
                "operation_type": "unknown_operation",  # This will fail
                "target": {},
                "parameters": {},
                "order": 2,
                "operation_id": "fail_op",
            },
        ]

        # Execute with atomic=True (default)
        result = batch_apply_operations(operations=operations)

        # Verify batch failed
        assert result.success is False
        assert result.total_operations == 2
        assert result.successful_operations == 1  # First op succeeded before failure
        assert result.failed_operations == 1
        assert "Batch failed" in result.error_summary

        # Verify the failed operation details
        failed_op = result.operation_results[1]
        assert failed_op.success is False
        assert "Unknown operation type" in failed_op.error

    @pytest.mark.asyncio
    async def test_batch_apply_operations_continue_on_error_mode(
        self, temp_docs_root, document_factory
    ):
        """Test batch operation with continue_on_error=True."""
        doc_name = "continue_test_doc"

        operations = [
            {
                "operation_type": "create_document",
                "target": {},
                "parameters": {"document_name": doc_name},
                "order": 1,
                "operation_id": "create_doc",
            },
            {
                "operation_type": "unknown_operation",  # This will fail
                "target": {},
                "parameters": {},
                "order": 2,
                "operation_id": "fail_op",
            },
            {
                "operation_type": "create_chapter",
                "target": {"document_name": doc_name},
                "parameters": {"chapter_name": "test.md", "initial_content": "# Test"},
                "order": 3,
                "operation_id": "create_chapter",
            },
        ]

        # Execute with continue_on_error=True and atomic=False
        result = batch_apply_operations(
            operations=operations, atomic=False, continue_on_error=True
        )

        # Verify mixed results
        assert result.success is False  # Overall failure due to one failed op
        assert result.total_operations == 3
        assert result.successful_operations == 2  # First and third operations
        assert result.failed_operations == 1

        # Verify that successful operations actually completed
        docs = list_documents()
        doc_names = [doc.document_name for doc in docs]
        assert doc_name in doc_names

        chapters = list_chapters(doc_name)
        assert len(chapters) == 1
        assert chapters[0].chapter_name == "test.md"

    @pytest.mark.asyncio
    async def test_batch_apply_operations_with_unified_read_content(
        self, temp_docs_root, document_factory
    ):
        """Test batch operation using the unified read_content tool."""
        doc_name = "unified_batch_test"
        chapters = {
            "chapter1.md": "# Chapter 1\n\nFirst chapter content.",
            "chapter2.md": "# Chapter 2\n\nSecond chapter content.",
        }
        document_factory(doc_name, chapters)

        operations = [
            {
                "operation_type": "read_content",
                "target": {"document_name": doc_name},
                "parameters": {"scope": "document"},
                "order": 1,
                "operation_id": "read_full_doc",
            },
            {
                "operation_type": "read_content",
                "target": {"document_name": doc_name},
                "parameters": {"scope": "chapter", "chapter_name": "chapter1.md"},
                "order": 2,
                "operation_id": "read_chapter",
            },
        ]

        # Execute batch
        result = batch_apply_operations(operations=operations)

        # Verify batch succeeded
        assert result.success is True
        assert result.total_operations == 2
        assert result.successful_operations == 2
        assert result.failed_operations == 0

        # Verify operation results contain expected data
        doc_read_result = result.operation_results[0]
        assert doc_read_result.success is True
        assert doc_read_result.result["document_name"] == doc_name
        assert len(doc_read_result.result["chapters"]) == 2

        chapter_read_result = result.operation_results[1]
        assert chapter_read_result.success is True
        assert chapter_read_result.result["chapter_name"] == "chapter1.md"
        assert "First chapter content" in chapter_read_result.result["content"]


class TestBatchOperationsForDocumentCreation:
    """Integration tests for batch operations replacing legacy composite operations."""

    def test_batch_create_document_with_chapters_success(self, temp_docs_root):
        """Test successful document creation with multiple chapters using batch operations."""
        doc_name = "test_batch_doc"

        # Create operations for document + chapters
        operations = [
            {
                "operation_type": "create_document",
                "target": {},
                "parameters": {"document_name": doc_name},
                "order": 1,
                "operation_id": "create_doc",
            },
            {
                "operation_type": "create_chapter",
                "target": {},
                "parameters": {
                    "document_name": doc_name,
                    "chapter_name": "01-introduction.md",
                    "initial_content": "# Introduction\n\nWelcome to the guide.",
                },
                "order": 2,
                "operation_id": "create_chapter1",
                "depends_on": ["create_doc"],
            },
            {
                "operation_type": "create_chapter",
                "target": {},
                "parameters": {
                    "document_name": doc_name,
                    "chapter_name": "02-setup.md",
                    "initial_content": "# Setup\n\nInstallation instructions.",
                },
                "order": 3,
                "operation_id": "create_chapter2",
                "depends_on": ["create_doc"],
            },
            {
                "operation_type": "create_chapter",
                "target": {},
                "parameters": {
                    "document_name": doc_name,
                    "chapter_name": "03-usage.md",
                    "initial_content": "",
                },
                "order": 4,
                "operation_id": "create_chapter3",
                "depends_on": ["create_doc"],
            },
        ]

        # Execute batch operation
        result = batch_apply_operations(operations, atomic=True, snapshot_before=True)

        # Verify overall success
        assert result.success is True
        assert result.total_operations == 4
        assert result.successful_operations == 4

        # Verify document was actually created
        docs = list_documents()
        doc_names = [doc.document_name for doc in docs]
        assert doc_name in doc_names

        # Verify chapters were actually created
        chapters_list = list_chapters(doc_name)
        chapter_names = [ch.chapter_name for ch in chapters_list]
        assert len(chapter_names) == 3
        assert "01-introduction.md" in chapter_names
        assert "02-setup.md" in chapter_names
        assert "03-usage.md" in chapter_names

        # Verify chapter content
        intro_content = read_content(
            doc_name, scope="chapter", chapter_name="01-introduction.md"
        )
        assert "Welcome to the guide" in intro_content.content

        setup_content = read_content(
            doc_name, scope="chapter", chapter_name="02-setup.md"
        )
        assert "Installation instructions" in setup_content.content

        # Clean up
        delete_document(doc_name)

    def test_batch_create_document_only(self, temp_docs_root):
        """Test batch document creation with no chapters."""
        doc_name = "test_single_doc"

        operations = [
            {
                "operation_type": "create_document",
                "target": {},
                "parameters": {"document_name": doc_name},
                "order": 1,
                "operation_id": "create_doc",
            }
        ]

        result = batch_apply_operations(operations, atomic=True)

        # Verify success
        assert result.success is True
        assert result.total_operations == 1
        assert result.successful_operations == 1

        # Verify document was created
        docs = list_documents()
        doc_names = [doc.document_name for doc in docs]
        assert doc_name in doc_names

        # Clean up
        delete_document(doc_name)

    def test_batch_rollback_on_invalid_chapter(self, temp_docs_root):
        """Test batch operation rollback when chapter creation fails."""
        doc_name = "test_rollback_batch"

        operations = [
            {
                "operation_type": "create_document",
                "target": {},
                "parameters": {"document_name": doc_name},
                "order": 1,
                "operation_id": "create_doc",
            },
            {
                "operation_type": "create_chapter",
                "target": {},
                "parameters": {
                    "document_name": doc_name,
                    "chapter_name": "01-valid.md",
                    "initial_content": "# Valid Chapter",
                },
                "order": 2,
                "operation_id": "create_chapter1",
                "depends_on": ["create_doc"],
            },
            {
                "operation_type": "create_chapter",
                "target": {},
                "parameters": {
                    "document_name": doc_name,
                    "chapter_name": "invalid_chapter_name_without_md_extension",  # Invalid
                    "initial_content": "This should fail",
                },
                "order": 3,
                "operation_id": "create_chapter2",
                "depends_on": ["create_doc"],
            },
        ]

        # Execute with atomic=True and snapshot_before=True for proper rollback
        result = batch_apply_operations(operations, atomic=True, snapshot_before=True)

        # Verify operation failed
        assert result.success is False

        # Verify document was NOT created (rollback occurred)
        docs = list_documents()
        doc_names = [doc.document_name for doc in docs]
        assert doc_name not in doc_names

    def test_batch_duplicate_document_name(self, document_factory):
        """Test batch operation error handling when document name already exists."""
        doc_name = "existing_batch_doc"

        # Create a document first
        document_factory(doc_name, {"existing.md": "Content"})

        # Try to create another document with the same name
        operations = [
            {
                "operation_type": "create_document",
                "target": {},
                "parameters": {"document_name": doc_name},
                "order": 1,
                "operation_id": "create_doc",
            }
        ]

        result = batch_apply_operations(operations, atomic=True)

        # Verify operation failed
        assert result.success is False
        assert "already exists" in str(result).lower()

        # Verify original document is unchanged
        original_chapters = list_chapters(doc_name)
        assert len(original_chapters) == 1
        assert original_chapters[0].chapter_name == "existing.md"


class TestBatchOperationsWithDependencies:
    """Integration tests for batch operations with dependency resolution."""

    @pytest.mark.asyncio
    async def test_batch_operations_with_simple_dependencies(
        self, temp_docs_root, document_factory
    ):
        """Test batch operations with simple dependency chain."""
        doc_name = "dependency_test_doc"

        # Operations with dependencies: create document, then chapter, then append paragraph
        operations = [
            {
                "operation_type": "append_paragraph_to_chapter",
                "target": {"document_name": doc_name, "chapter_name": "intro.md"},
                "parameters": {"new_content": "This is an additional paragraph."},
                "order": 3,
                "operation_id": "append_para",
                "depends_on": ["create_chapter"],
            },
            {
                "operation_type": "create_chapter",
                "target": {"document_name": doc_name},
                "parameters": {
                    "chapter_name": "intro.md",
                    "initial_content": "# Introduction\n\nWelcome to the guide.",
                },
                "order": 2,
                "operation_id": "create_chapter",
                "depends_on": ["create_doc"],
            },
            {
                "operation_type": "create_document",
                "target": {},
                "parameters": {"document_name": doc_name},
                "order": 1,
                "operation_id": "create_doc",
                "depends_on": [],
            },
        ]

        # Execute batch with dependencies
        result = batch_apply_operations(operations=operations)

        # Verify batch succeeded
        assert result.success is True
        assert result.total_operations == 3
        assert result.successful_operations == 3
        assert result.failed_operations == 0

        # Verify operations executed in correct order (despite being defined out of order)
        assert len(result.operation_results) == 3
        assert result.operation_results[0].operation_id == "create_doc"
        assert result.operation_results[1].operation_id == "create_chapter"
        assert result.operation_results[2].operation_id == "append_para"

        # Verify actual document state
        docs = list_documents()
        doc_names = [doc.document_name for doc in docs]
        assert doc_name in doc_names

        chapters = list_chapters(doc_name)
        assert len(chapters) == 1
        assert chapters[0].chapter_name == "intro.md"

        chapter_content = read_content(
            doc_name, scope="chapter", chapter_name="intro.md"
        )
        assert "Welcome to the guide" in chapter_content.content
        assert "This is an additional paragraph" in chapter_content.content

    @pytest.mark.asyncio
    async def test_batch_operations_with_multiple_dependencies(self, document_factory):
        """Test batch operation where one operation depends on multiple others."""
        doc_name = "multi_dep_test_doc"

        operations = [
            {
                "operation_type": "replace_text",
                "target": {"document_name": doc_name},
                "parameters": {
                    "find_text": "placeholder",
                    "replace_text": "final content",
                    "scope": "document",
                },
                "order": 4,
                "operation_id": "replace_text_op",
                "depends_on": ["create_ch1", "create_ch2"],  # Depends on both chapters
            },
            {
                "operation_type": "create_chapter",
                "target": {"document_name": doc_name},
                "parameters": {
                    "chapter_name": "chapter2.md",
                    "initial_content": "# Chapter 2\n\nSecond placeholder content.",
                },
                "order": 3,
                "operation_id": "create_ch2",
                "depends_on": ["create_doc"],
            },
            {
                "operation_type": "create_document",
                "target": {},
                "parameters": {"document_name": doc_name},
                "order": 1,
                "operation_id": "create_doc",
                "depends_on": [],
            },
            {
                "operation_type": "create_chapter",
                "target": {"document_name": doc_name},
                "parameters": {
                    "chapter_name": "chapter1.md",
                    "initial_content": "# Chapter 1\n\nFirst placeholder content.",
                },
                "order": 2,
                "operation_id": "create_ch1",
                "depends_on": ["create_doc"],
            },
        ]

        # Execute batch
        result = batch_apply_operations(operations=operations)

        # Verify success
        assert result.success is True
        assert result.total_operations == 4
        assert result.successful_operations == 4

        # Verify execution order
        op_results = result.operation_results
        assert op_results[0].operation_id == "create_doc"
        assert op_results[-1].operation_id == "replace_text_op"

        # Both chapters should come before text replacement
        ch1_index = next(
            i for i, op in enumerate(op_results) if op.operation_id == "create_ch1"
        )
        ch2_index = next(
            i for i, op in enumerate(op_results) if op.operation_id == "create_ch2"
        )
        replace_index = next(
            i for i, op in enumerate(op_results) if op.operation_id == "replace_text_op"
        )

        assert ch1_index < replace_index
        assert ch2_index < replace_index

        # Verify text replacement worked on both chapters
        ch1_content = read_content(
            doc_name, scope="chapter", chapter_name="chapter1.md"
        )
        ch2_content = read_content(
            doc_name, scope="chapter", chapter_name="chapter2.md"
        )
        assert "final content" in ch1_content.content
        assert "final content" in ch2_content.content
        assert "placeholder" not in ch1_content.content
        assert "placeholder" not in ch2_content.content

    @pytest.mark.asyncio
    async def test_batch_operations_circular_dependency_failure(self):
        """Test that circular dependencies are properly detected and cause batch failure."""
        operations = [
            {
                "operation_type": "create_document",
                "target": {},
                "parameters": {"document_name": "circular_test"},
                "order": 1,
                "operation_id": "op_a",
                "depends_on": ["op_b"],
            },
            {
                "operation_type": "create_chapter",
                "target": {"document_name": "circular_test"},
                "parameters": {"chapter_name": "test.md", "initial_content": "Test"},
                "order": 2,
                "operation_id": "op_b",
                "depends_on": ["op_a"],
            },
        ]

        # Execute batch (should fail)
        result = batch_apply_operations(operations=operations)

        # Verify failure due to circular dependency
        assert result.success is False
        assert result.total_operations == 2
        assert result.successful_operations == 0
        assert result.failed_operations == 2
        assert "Circular dependency" in result.error_summary

    @pytest.mark.asyncio
    async def test_batch_operations_unknown_dependency_failure(self):
        """Test that unknown dependencies cause batch failure."""
        operations = [
            {
                "operation_type": "create_chapter",
                "target": {"document_name": "unknown_dep_test"},
                "parameters": {"chapter_name": "test.md", "initial_content": "Test"},
                "order": 1,
                "operation_id": "create_chapter",
                "depends_on": ["nonexistent_operation"],
            }
        ]

        # Execute batch (should fail)
        result = batch_apply_operations(operations=operations)

        # Verify failure due to unknown dependency
        assert result.success is False
        assert "unknown operation" in result.error_summary
