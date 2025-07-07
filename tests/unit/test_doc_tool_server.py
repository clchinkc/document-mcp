"""
Unit tests for document MCP tool server helper functions.
"""

import pytest

from document_mcp.doc_tool_server import (
    CHAPTER_MANIFEST_FILE,
    DOCUMENT_SUMMARY_FILE,
    _count_words,
    _is_valid_chapter_filename,
    _split_into_paragraphs,
    _validate_content,
)


class TestHelperFunctions:
    """Test suite for helper functions in doc_tool_server."""

    def test_count_words_empty_string(self):
        """Test word counting with empty string."""
        assert _count_words("") == 0

    def test_count_words_single_word(self):
        """Test word counting with single word."""
        assert _count_words("hello") == 1

    def test_count_words_multiple_words(self):
        """Test word counting with multiple words."""
        assert _count_words("hello world test") == 3

    def test_count_words_with_extra_spaces(self):
        """Test word counting with extra spaces."""
        assert _count_words("  hello   world  ") == 2

    def test_count_words_with_newlines(self):
        """Test word counting with newlines."""
        assert _count_words("hello\nworld\ntest") == 3

    def test_split_into_paragraphs_empty_string(self):
        """Test paragraph splitting with empty string."""
        assert _split_into_paragraphs("") == []

    def test_split_into_paragraphs_single_paragraph(self):
        """Test paragraph splitting with single paragraph."""
        text = "This is a single paragraph with multiple sentences."
        result = _split_into_paragraphs(text)
        assert len(result) == 1
        assert result[0] == text

    def test_split_into_paragraphs_multiple_paragraphs(self):
        """Test paragraph splitting with multiple paragraphs."""
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        result = _split_into_paragraphs(text)
        assert len(result) == 3
        assert result[0] == "First paragraph."
        assert result[1] == "Second paragraph."
        assert result[2] == "Third paragraph."

    def test_is_valid_chapter_filename_valid_md_files(self):
        """Test filename validation with valid markdown files."""
        assert _is_valid_chapter_filename("chapter1.md") is True
        assert _is_valid_chapter_filename("01-introduction.md") is True

    def test_is_valid_chapter_filename_invalid_extensions(self):
        """Test filename validation rejects non-markdown files."""
        assert _is_valid_chapter_filename("chapter.txt") is False
        assert _is_valid_chapter_filename("chapter") is False

    def test_is_valid_chapter_filename_manifest_file(self):
        """Test filename validation rejects manifest file."""
        assert _is_valid_chapter_filename(CHAPTER_MANIFEST_FILE) is False

    def test_is_valid_chapter_filename_summary_file(self):
        """Test filename validation rejects summary file."""
        assert _is_valid_chapter_filename(DOCUMENT_SUMMARY_FILE) is False


class TestInputValidationHelpers:
    """Test suite for input validation helper functions."""

    @pytest.mark.parametrize(
        "name, expected_valid, expected_error_msg",
        [
            ("doc_name", True, ""),
            ("doc-name-123", True, ""),
            (None, False, "Document name cannot be empty"),
            ("invalid/name", False, "Document name cannot contain path separators"),
        ],
    )
    def test_validate_document_name(self, name, expected_valid, expected_error_msg):
        from document_mcp.doc_tool_server import _validate_document_name

        is_valid, error = _validate_document_name(name)
        assert is_valid is expected_valid
        if not expected_valid:
            assert expected_error_msg == error

    @pytest.mark.parametrize(
        "name, expected_valid, expected_error_msg",
        [
            ("chapter1.md", True, ""),
            ("01-intro.md", True, ""),
            (None, False, "Chapter name cannot be empty"),
            (
                "invalid/chapter.md",
                False,
                "Chapter name cannot contain path separators",
            ),
            ("no_extension", False, "Chapter name must end with .md"),
        ],
    )
    def test_validate_chapter_name(self, name, expected_valid, expected_error_msg):
        from document_mcp.doc_tool_server import _validate_chapter_name

        is_valid, error = _validate_chapter_name(name)
        assert is_valid == expected_valid
        assert error == expected_error_msg

    @pytest.mark.parametrize(
        "content, expected_valid, expected_error_msg",
        [
            ("Some valid content.", True, ""),
            ("", True, ""),
            (None, False, "Content cannot be None"),
            (12345, False, "Content must be a string"),
        ],
    )
    def test_validate_content_general_cases(
        self, content, expected_valid, expected_error_msg
    ):
        """Test content validation for general cases (None, type, and short strings)."""
        is_valid, error = _validate_content(content)
        assert is_valid == expected_valid
        assert error == expected_error_msg

    @pytest.mark.parametrize(
        "index, expected_valid, expected_error_msg",
        [
            (0, True, ""),
            (100, True, ""),
            (-1, False, "Paragraph index cannot be negative"),
        ],
    )
    def test_validate_paragraph_index(self, index, expected_valid, expected_error_msg):
        from document_mcp.doc_tool_server import _validate_paragraph_index

        is_valid, error = _validate_paragraph_index(index)
        assert is_valid is expected_valid
        if not expected_valid:
            assert expected_error_msg == error

    @pytest.mark.parametrize(
        "query, expected_valid, expected_error_msg",
        [
            ("hello", True, ""),
            (None, False, "Search query cannot be None"),
            ("", False, "Search query cannot be empty or whitespace only"),
        ],
    )
    def test_validate_search_query(self, query, expected_valid, expected_error_msg):
        from document_mcp.doc_tool_server import _validate_search_query

        is_valid, error = _validate_search_query(query)
        assert is_valid is expected_valid
        if not expected_valid:
            assert expected_error_msg == error
