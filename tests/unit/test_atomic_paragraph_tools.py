"""Tests for the new atomic paragraph-level tools.

This module tests the atomic paragraph manipulation tools that were added
to replace the non-atomic modify_paragraph_content function.
"""

from document_mcp.mcp_client import append_paragraph_to_chapter
from document_mcp.mcp_client import delete_paragraph
from document_mcp.mcp_client import insert_paragraph_after
from document_mcp.mcp_client import insert_paragraph_before
from document_mcp.mcp_client import move_paragraph_before
from document_mcp.mcp_client import move_paragraph_to_end
from document_mcp.mcp_client import replace_paragraph


class TestReplaceParagraph:
    """Tests for the replace_paragraph atomic tool."""

    def test_replace_paragraph_success(self, document_factory):
        """Test successful paragraph replacement."""
        doc_name = "test_doc"
        chapter_name = "test_chapter.md"

        document_factory(
            doc_name,
            {chapter_name: "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."},
        )

        new_content = "This is the new second paragraph."
        result = replace_paragraph(doc_name, chapter_name, 1, new_content)

        assert result.success is True
        assert "replaced successfully" in result.message
        assert result.details is not None
        assert result.details["document_name"] == doc_name
        assert result.details["chapter_name"] == chapter_name
        assert result.details["paragraph_index"] == 1

    def test_replace_paragraph_out_of_bounds(self, document_factory):
        """Test replacing paragraph with invalid index."""
        doc_name = "test_doc"
        chapter_name = "test_chapter.md"

        document_factory(doc_name, {chapter_name: "Only one paragraph."})

        result = replace_paragraph(doc_name, chapter_name, 5, "New content")

        assert result.success is False
        assert "out of bounds" in result.message

    def test_replace_paragraph_invalid_inputs(self, document_factory):
        """Test replace_paragraph with invalid inputs."""
        result = replace_paragraph("", "test.md", 0, "content")
        assert result.success is False
        # Safety system catches this before validation, expecting safety error
        assert "Document name cannot be empty" in result.message or "Safety check failed" in result.message

        result = replace_paragraph("doc", "invalid", 0, "content")
        assert result.success is False
        # Either validation error or safety error is acceptable
        assert "must end with .md" in result.message or "Safety check failed" in result.message

        result = replace_paragraph("doc", "test.md", -1, "content")
        assert result.success is False
        # Either validation error or safety error is acceptable
        assert "cannot be negative" in result.message or "Safety check failed" in result.message


class TestInsertParagraphBefore:
    """Tests for the insert_paragraph_before atomic tool."""

    def test_insert_paragraph_before_success(self, document_factory):
        """Test successful paragraph insertion before existing paragraph."""
        doc_name = "test_doc"
        chapter_name = "test_chapter.md"

        document_factory(doc_name, {chapter_name: "First paragraph.\n\nSecond paragraph."})

        # Insert before the second paragraph (index 1)
        new_content = "This is inserted before the second paragraph."
        result = insert_paragraph_before(doc_name, chapter_name, 1, new_content)

        assert result.success is True
        assert "inserted before index 1" in result.message
        assert result.details is not None
        assert result.details["document_name"] == doc_name
        assert result.details["chapter_name"] == chapter_name
        assert result.details["paragraph_index"] == 1

    def test_insert_paragraph_before_at_beginning(self, document_factory):
        """Test inserting paragraph at the very beginning (index 0)."""
        doc_name = "test_doc"
        chapter_name = "test_chapter.md"

        document_factory(doc_name, {chapter_name: "Original first paragraph."})

        # Insert at index 0 (beginning)
        new_content = "This is now the first paragraph."
        result = insert_paragraph_before(doc_name, chapter_name, 0, new_content)

        assert result.success is True
        assert "inserted before index 0" in result.message


class TestInsertParagraphAfter:
    """Tests for the insert_paragraph_after atomic tool."""

    def test_insert_paragraph_after_success(self, document_factory):
        """Test successful paragraph insertion after existing paragraph."""
        doc_name = "test_doc"
        chapter_name = "test_chapter.md"

        document_factory(doc_name, {chapter_name: "First paragraph.\n\nSecond paragraph."})

        # Insert after the first paragraph (index 0)
        new_content = "This is inserted after the first paragraph."
        result = insert_paragraph_after(doc_name, chapter_name, 0, new_content)

        assert result.success is True
        assert "inserted after index 0" in result.message
        assert result.details is not None
        assert result.details["document_name"] == doc_name
        assert result.details["chapter_name"] == chapter_name
        assert result.details["paragraph_index"] == 1  # Index after insertion (0+1)

    def test_insert_paragraph_after_empty_chapter(self, document_factory):
        """Test inserting paragraph in empty chapter (should fail for index 0)."""
        doc_name = "test_doc"
        chapter_name = "test_chapter.md"

        document_factory(doc_name, {chapter_name: ""})

        # Insert in empty chapter - should fail because there's no paragraph at index 0 to insert after
        new_content = "This is the first paragraph in empty chapter."
        result = insert_paragraph_after(doc_name, chapter_name, 0, new_content)

        assert result.success is False
        assert "out of bounds" in result.message


class TestDeleteParagraph:
    """Tests for the delete_paragraph atomic tool."""

    def test_delete_paragraph_success(self, document_factory):
        """Test successful paragraph deletion."""
        doc_name = "test_doc"
        chapter_name = "test_chapter.md"

        document_factory(
            doc_name,
            {chapter_name: "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."},
        )

        # Delete the second paragraph (index 1)
        result = delete_paragraph(doc_name, chapter_name, 1)

        assert result.success is True
        assert "deleted from" in result.message
        assert result.details is not None
        assert result.details["document_name"] == doc_name
        assert result.details["chapter_name"] == chapter_name
        assert result.details["paragraph_index"] == 1

    def test_delete_paragraph_out_of_bounds(self, document_factory):
        """Test deleting paragraph with invalid index."""
        doc_name = "test_doc"
        chapter_name = "test_chapter.md"
        document_factory(doc_name, {chapter_name: "Only one paragraph."})

        result = delete_paragraph(doc_name, chapter_name, 5)

        assert result.success is False
        assert "out of bounds" in result.message

    def test_delete_paragraph_from_empty_chapter(self, document_factory):
        doc_name = "test_doc"
        chapter_name = "test_chapter.md"
        document_factory(doc_name, {chapter_name: ""})

        result = delete_paragraph(doc_name, chapter_name, 0)

        assert result.success is False
        assert "out of bounds" in result.message


class TestAppendParagraphToChapter:
    """Tests for the append_paragraph_to_chapter atomic tool."""

    def test_append_paragraph_to_chapter_success(self, document_factory):
        """Test successful paragraph appending."""
        doc_name = "test_doc"
        chapter_name = "test_chapter.md"

        document_factory(doc_name, {chapter_name: "Initial content."})

        # Append paragraph content
        additional_content = "This is appended content."
        result = append_paragraph_to_chapter(doc_name, chapter_name, additional_content)

        assert result.success is True
        assert "appended to chapter" in result.message
        assert result.details is not None
        assert result.details["document_name"] == doc_name
        assert result.details["chapter_name"] == chapter_name
        assert result.details["paragraph_index"] == 1  # Second paragraph (0-indexed)

    def test_append_paragraph_to_chapter_empty_chapter(self, document_factory):
        doc_name = "test_doc"
        chapter_name = "test_chapter.md"

        document_factory(doc_name, {chapter_name: ""})

        # Append to empty chapter
        content = "This is the first content."
        result = append_paragraph_to_chapter(doc_name, chapter_name, content)

        assert result.success is True
        assert "appended to chapter" in result.message


class TestMoveParagraphBefore:
    """Tests for the move_paragraph_before atomic tool."""

    def test_move_paragraph_before_success(self, document_factory):
        """Test successful paragraph movement."""
        doc_name = "test_doc"
        chapter_name = "test_chapter.md"

        document_factory(
            doc_name,
            {chapter_name: "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.\n\nFourth paragraph."},
        )

        # Move paragraph 3 (index 2) before paragraph 1 (index 0)
        result = move_paragraph_before(doc_name, chapter_name, 2, 0)

        assert result.success is True
        assert "moved before paragraph" in result.message
        assert result.details is not None
        assert result.details["changed"] is True

    def test_move_paragraph_before_same_index(self, document_factory):
        """Test moving paragraph before itself (should fail)."""
        doc_name = "test_doc"
        chapter_name = "test_chapter.md"

        document_factory(doc_name, {chapter_name: "First paragraph.\n\nSecond paragraph."})

        result = move_paragraph_before(doc_name, chapter_name, 1, 1)

        assert result.success is False
        assert "before itself" in result.message

    def test_move_paragraph_before_out_of_bounds(self, document_factory):
        """Test moving paragraph with invalid indices."""
        doc_name = "test_doc"
        chapter_name = "test_chapter.md"

        document_factory(doc_name, {chapter_name: "First paragraph.\n\nSecond paragraph."})

        result = move_paragraph_before(doc_name, chapter_name, 5, 0)

        assert result.success is False
        assert "out of bounds" in result.message


class TestMoveParagraphToEnd:
    """Tests for the move_paragraph_to_end atomic tool."""

    def test_move_paragraph_to_end_success(self, document_factory):
        """Test successful paragraph movement to end."""
        doc_name = "test_doc"
        chapter_name = "test_chapter.md"

        document_factory(
            doc_name,
            {chapter_name: "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."},
        )

        # Move first paragraph (index 0) to end
        result = move_paragraph_to_end(doc_name, chapter_name, 0)

        assert result.success is True
        assert "moved to end" in result.message
        assert result.details is not None
        assert result.details["changed"] is True

    def test_move_paragraph_to_end_already_at_end(self, document_factory):
        """Test moving paragraph that's already at the end."""
        doc_name = "test_doc"
        chapter_name = "test_chapter.md"
        document_factory(doc_name, {chapter_name: "First paragraph.\n\nSecond paragraph."})

        # Move last paragraph (index 1) to end (should be no-op)
        result = move_paragraph_to_end(doc_name, chapter_name, 1)

        assert result.success is True
        assert "already at the end" in result.message
        assert result.details["changed"] is False

    def test_move_paragraph_to_end_out_of_bounds(self, document_factory):
        """Test moving non-existent paragraph to end."""
        doc_name = "test_doc"
        chapter_name = "test_chapter.md"

        document_factory(doc_name, {chapter_name: "Only one paragraph."})

        result = move_paragraph_to_end(doc_name, chapter_name, 5)

        assert result.success is False
        assert "out of bounds" in result.message
