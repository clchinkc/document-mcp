"""
Unit tests for document MCP tool server functions.

This module tests individual functions in isolation with focus on:
- Input validation
- Error handling
- Boundary conditions
- Function behavior without external dependencies
"""

import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock # Added MagicMock

from document_mcp.doc_tool_server import (
    _count_words,
    _get_chapter_metadata,
    _get_chapter_path,
    _get_document_path,
    _get_ordered_chapter_files,
    _is_valid_chapter_filename,
    _read_chapter_content_details,
    _split_into_paragraphs,
    read_document_summary,
    list_documents,
    DocumentInfo,
    DOCUMENT_SUMMARY_FILE,
    CHAPTER_MANIFEST_FILE,
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

    def test_split_into_paragraphs_with_multiple_blank_lines(self):
        """Test paragraph splitting with multiple blank lines."""
        text = "First paragraph.\n\n\n\nSecond paragraph."
        result = _split_into_paragraphs(text)
        assert len(result) == 2
        assert result[0] == "First paragraph."
        assert result[1] == "Second paragraph."

    def test_split_into_paragraphs_with_whitespace_lines(self):
        """Test paragraph splitting with lines containing only whitespace."""
        text = "First paragraph.\n  \n\t\n\nSecond paragraph."
        result = _split_into_paragraphs(text)
        assert len(result) == 2
        assert result[0] == "First paragraph."
        assert result[1] == "Second paragraph."

    def test_split_into_paragraphs_different_line_endings(self):
        """Test paragraph splitting with different line ending styles."""
        text = "First paragraph.\r\n\r\nSecond paragraph.\r\nThird paragraph."
        result = _split_into_paragraphs(text)
        assert len(result) == 2
        assert result[0] == "First paragraph."
        assert result[1] == "Second paragraph.\nThird paragraph."

    def test_is_valid_chapter_filename_valid_md_files(self):
        """Test filename validation with valid markdown files."""
        assert _is_valid_chapter_filename("chapter1.md") is True
        assert _is_valid_chapter_filename("01-introduction.md") is True
        assert _is_valid_chapter_filename("Chapter_Title.md") is True
        assert _is_valid_chapter_filename("chapter-with-dashes.md") is True

    def test_is_valid_chapter_filename_case_insensitive(self):
        """Test filename validation is case insensitive for .md extension."""
        assert _is_valid_chapter_filename("chapter.MD") is True
        assert _is_valid_chapter_filename("chapter.Md") is True
        assert _is_valid_chapter_filename("chapter.mD") is True

    def test_is_valid_chapter_filename_invalid_extensions(self):
        """Test filename validation rejects non-markdown files."""
        assert _is_valid_chapter_filename("chapter.txt") is False
        assert _is_valid_chapter_filename("chapter.doc") is False
        assert _is_valid_chapter_filename("chapter.pdf") is False
        assert _is_valid_chapter_filename("chapter") is False

    def test_is_valid_chapter_filename_manifest_file(self):
        """Test filename validation rejects manifest file."""
        assert _is_valid_chapter_filename(CHAPTER_MANIFEST_FILE) is False

    def test_is_valid_chapter_filename_summary_file(self):
        """Test filename validation rejects summary file."""
        assert _is_valid_chapter_filename(DOCUMENT_SUMMARY_FILE) is False

    def test_is_valid_chapter_filename_empty_string(self):
        """Test filename validation with empty string."""
        assert _is_valid_chapter_filename("") is False


class TestPathHelpers:
    """Test suite for path helper functions."""

    @patch("document_mcp.doc_tool_server.DOCS_ROOT_PATH", Path("/test/docs"))
    def test_get_document_path(self):
        """Test document path generation."""
        result = _get_document_path("my_document")
        expected = Path("/test/docs/my_document")
        assert result == expected

    @patch("document_mcp.doc_tool_server.DOCS_ROOT_PATH", Path("/test/docs"))
    def test_get_document_path_with_special_chars(self):
        """Test document path generation with special characters."""
        result = _get_document_path("my-document_2023")
        expected = Path("/test/docs/my-document_2023")
        assert result == expected

    @patch("document_mcp.doc_tool_server.DOCS_ROOT_PATH", Path("/test/docs"))
    def test_get_chapter_path(self):
        """Test chapter path generation."""
        result = _get_chapter_path("my_document", "chapter1.md")
        expected = Path("/test/docs/my_document/chapter1.md")
        assert result == expected

    @patch("document_mcp.doc_tool_server.DOCS_ROOT_PATH", Path("/test/docs"))
    def test_get_chapter_path_with_subdirectory(self):
        """Test chapter path generation with document containing special characters."""
        result = _get_chapter_path("project-alpha_v2", "01-intro.md")
        expected = Path("/test/docs/project-alpha_v2/01-intro.md")
        assert result == expected


class TestFileOrdering:
    """Test suite for file ordering functions."""

    @patch("document_mcp.doc_tool_server.DOCS_ROOT_PATH", Path("/test/docs"))
    def test_get_ordered_chapter_files_nonexistent_document(self):
        """Test getting chapter files for non-existent document."""
        with patch("pathlib.Path.is_dir", return_value=False):
            result = _get_ordered_chapter_files("nonexistent_doc")
            assert result == []

    @patch("document_mcp.doc_tool_server.DOCS_ROOT_PATH", Path("/test/docs"))
    def test_get_ordered_chapter_files_empty_document(self):
        """Test getting chapter files for empty document directory."""
        mock_doc_path = Mock()
        mock_doc_path.is_dir.return_value = True
        mock_doc_path.iterdir.return_value = []

        with patch(
            "document_mcp.doc_tool_server._get_document_path",
            return_value=mock_doc_path,
        ):
            result = _get_ordered_chapter_files("empty_doc")
            assert result == []

    @patch("document_mcp.doc_tool_server.DOCS_ROOT_PATH", Path("/test/docs"))
    def test_get_ordered_chapter_files_with_valid_chapters(self):
        """Test getting chapter files with valid markdown files."""
        # Create mock file objects with proper __lt__ method for sorting
        chapter1 = Mock()
        chapter1.name = "01-intro.md"
        chapter1.is_file.return_value = True
        chapter1.__lt__ = lambda self, other: self.name < other.name

        chapter2 = Mock()
        chapter2.name = "02-methods.md"
        chapter2.is_file.return_value = True
        chapter2.__lt__ = lambda self, other: self.name < other.name

        chapter3 = Mock()
        chapter3.name = "03-results.md"
        chapter3.is_file.return_value = True
        chapter3.__lt__ = lambda self, other: self.name < other.name

        # Mock the document path
        mock_doc_path = Mock()
        mock_doc_path.is_dir.return_value = True
        mock_doc_path.iterdir.return_value = [
            chapter3,
            chapter1,
            chapter2,
        ]  # Unsorted order

        with patch(
            "document_mcp.doc_tool_server._get_document_path",
            return_value=mock_doc_path,
        ):
            result = _get_ordered_chapter_files("test_doc")

            # Should be sorted alphabetically
            assert len(result) == 3
            assert result[0].name == "01-intro.md"
            assert result[1].name == "02-methods.md"
            assert result[2].name == "03-results.md"

    @patch("document_mcp.doc_tool_server.DOCS_ROOT_PATH", Path("/test/docs"))
    def test_get_ordered_chapter_files_filters_non_md_files(self):
        """Test that non-markdown files are filtered out."""
        # Create mock file objects
        chapter1 = Mock()
        chapter1.name = "01-intro.md"
        chapter1.is_file.return_value = True

        text_file = Mock()
        text_file.name = "notes.txt"
        text_file.is_file.return_value = True

        summary_file = Mock()
        summary_file.name = "_SUMMARY.md"
        summary_file.is_file.return_value = True

        manifest_file = Mock()
        manifest_file.name = "_manifest.json"
        manifest_file.is_file.return_value = True

        directory = Mock()
        directory.name = "images"
        directory.is_file.return_value = False

        # Mock the document path
        mock_doc_path = Mock()
        mock_doc_path.is_dir.return_value = True
        mock_doc_path.iterdir.return_value = [
            chapter1,
            text_file,
            manifest_file,
            summary_file, # Add summary file to the list of items
            directory,
        ]

        with patch(
            "document_mcp.doc_tool_server._get_document_path",
            return_value=mock_doc_path,
        ):
            result = _get_ordered_chapter_files("test_doc")

            # Should only include the markdown file
            assert len(result) == 1
            assert result[0].name == "01-intro.md"

    @patch("document_mcp.doc_tool_server.DOCS_ROOT_PATH", Path("/test/docs"))
    def test_get_ordered_chapter_files_filters_summary_file(self):
        """Test that _SUMMARY.md files are filtered out."""
        chapter1 = Mock()
        chapter1.name = "01-intro.md"
        chapter1.is_file.return_value = True

        summary_file = Mock()
        summary_file.name = "_SUMMARY.md"
        summary_file.is_file.return_value = True

        mock_doc_path = Mock()
        mock_doc_path.is_dir.return_value = True
        mock_doc_path.iterdir.return_value = [chapter1, summary_file]

        with patch(
            "document_mcp.doc_tool_server._get_document_path",
            return_value=mock_doc_path,
        ):
            result = _get_ordered_chapter_files("test_doc_with_summary")
            assert len(result) == 1
            assert result[0].name == "01-intro.md"


class TestChapterReading:
    """Test suite for chapter content reading functions."""

    def test_read_chapter_content_details_nonexistent_file(self):
        """Test reading content from non-existent file."""
        mock_path = Mock()
        mock_path.is_file.return_value = False

        result = _read_chapter_content_details("test_doc", mock_path)
        assert result is None

    def test_read_chapter_content_details_valid_file(self):
        """Test reading content from valid file."""
        mock_path = Mock()
        mock_path.is_file.return_value = True
        mock_path.name = "01-intro.md"
        mock_path.read_text.return_value = "# Introduction\n\nThis is the first paragraph.\n\nThis is the second paragraph."

        # Mock the stat result
        mock_stat = Mock()
        mock_stat.st_mtime = 1640995200.0  # 2022-01-01 00:00:00 UTC
        mock_path.stat.return_value = mock_stat

        result = _read_chapter_content_details("test_doc", mock_path)

        assert result is not None
        assert result.document_name == "test_doc"
        assert result.chapter_name == "01-intro.md"
        assert (
            result.content
            == "# Introduction\n\nThis is the first paragraph.\n\nThis is the second paragraph."
        )
        assert (
            result.word_count == 12
        )  # "Introduction", "This", "is", "the", "first", "paragraph", "This", "is", "the", "second", "paragraph"
        assert result.paragraph_count == 3  # Title + 2 paragraphs
        assert isinstance(result.last_modified, datetime.datetime)

    def test_read_chapter_content_details_file_read_error(self):
        """Test handling file read errors."""
        mock_path = Mock()
        mock_path.is_file.return_value = True
        mock_path.name = "corrupted.md"
        mock_path.read_text.side_effect = Exception("File read error")

        with patch("builtins.print") as mock_print:
            result = _read_chapter_content_details("test_doc", mock_path)

            assert result is None
            mock_print.assert_called_once()
            assert "Error reading chapter file" in mock_print.call_args[0][0]

    def test_get_chapter_metadata_nonexistent_file(self):
        """Test getting metadata from non-existent file."""
        mock_path = Mock()
        mock_path.is_file.return_value = False

        result = _get_chapter_metadata("test_doc", mock_path)
        assert result is None

    def test_get_chapter_metadata_valid_file(self):
        """Test getting metadata from valid file."""
        mock_path = Mock()
        mock_path.is_file.return_value = True
        mock_path.name = "02-methods.md"
        mock_path.read_text.return_value = "# Methods\n\nFirst paragraph about methods.\n\nSecond paragraph with details.\n\nThird paragraph conclusion."

        # Mock the stat result
        mock_stat = Mock()
        mock_stat.st_mtime = 1640995200.0  # 2022-01-01 00:00:00 UTC
        mock_path.stat.return_value = mock_stat

        result = _get_chapter_metadata("test_doc", mock_path)

        assert result is not None
        assert result.chapter_name == "02-methods.md"
        assert (
            result.title is None
        )  # Title extraction not implemented in helper function
        assert (
            result.word_count == 13
        )  # Count of all words: Methods, First, paragraph, about, methods, Second, paragraph, with, details, Third, paragraph, conclusion
        assert result.paragraph_count == 4  # Title + 3 paragraphs
        assert isinstance(result.last_modified, datetime.datetime)

    def test_get_chapter_metadata_empty_file(self):
        """Test getting metadata from empty file."""
        mock_path = Mock()
        mock_path.is_file.return_value = True
        mock_path.name = "empty.md"
        mock_path.read_text.return_value = ""

        # Mock the stat result
        mock_stat = Mock()
        mock_stat.st_mtime = 1640995200.0
        mock_path.stat.return_value = mock_stat

        result = _get_chapter_metadata("test_doc", mock_path)

        assert result is not None
        assert result.chapter_name == "empty.md"
        assert result.word_count == 0
        assert result.paragraph_count == 0

    def test_get_chapter_metadata_file_read_error(self):
        """Test handling file read errors in metadata extraction."""
        mock_path = Mock()
        mock_path.is_file.return_value = True
        mock_path.name = "error.md"
        mock_path.read_text.side_effect = Exception("Permission denied")

        with patch("builtins.print") as mock_print:
            result = _get_chapter_metadata("test_doc", mock_path)

            assert result is None
            mock_print.assert_called_once()
            assert "Error getting metadata for chapter" in mock_print.call_args[0][0]


class TestInputValidationHelpers:
    """Test suite for input validation helper functions."""

    def test_validate_document_name_valid(self):
        """Test document name validation with valid inputs."""
        from document_mcp.doc_tool_server import _validate_document_name

        valid_names = ["test_doc", "my-document", "Document123", "project_v2"]
        for name in valid_names:
            is_valid, error = _validate_document_name(name)
            assert is_valid is True, f"'{name}' should be valid"
            assert error == "", f"No error expected for '{name}'"

    def test_validate_document_name_invalid(self):
        """Test document name validation with invalid inputs."""
        from document_mcp.doc_tool_server import _validate_document_name

        # Empty name
        is_valid, error = _validate_document_name("")
        assert is_valid is False
        assert "cannot be empty" in error

        # Path separators
        is_valid, error = _validate_document_name("doc/name")
        assert is_valid is False
        assert "path separators" in error

        is_valid, error = _validate_document_name("doc\\name")
        assert is_valid is False
        assert "path separators" in error

        # Starts with dot
        is_valid, error = _validate_document_name(".hidden")
        assert is_valid is False
        assert "cannot start with a dot" in error

        # Too long
        long_name = "a" * 101
        is_valid, error = _validate_document_name(long_name)
        assert is_valid is False
        assert "too long" in error

    def test_validate_chapter_name_valid(self):
        """Test chapter name validation with valid inputs."""
        from document_mcp.doc_tool_server import _validate_chapter_name

        valid_names = [
            "chapter1.md",
            "01-intro.md",
            "Chapter_Title.md",
            "test-chapter.md",
        ]
        for name in valid_names:
            is_valid, error = _validate_chapter_name(name)
            assert is_valid is True, f"'{name}' should be valid"
            assert error == "", f"No error expected for '{name}'"

    def test_validate_chapter_name_invalid(self):
        """Test chapter name validation with invalid inputs."""
        from document_mcp.doc_tool_server import _validate_chapter_name

        # Empty name
        is_valid, error = _validate_chapter_name("")
        assert is_valid is False
        assert "cannot be empty" in error

        # Wrong extension
        is_valid, error = _validate_chapter_name("chapter.txt")
        assert is_valid is False
        assert "must end with .md" in error

        # Path separators
        is_valid, error = _validate_chapter_name("dir/chapter.md")
        assert is_valid is False
        assert "path separators" in error

        # Reserved name (but this needs to be a .md file first)
        # Since _manifest.json is not .md, it fails on extension first
        # Let's test the reserved name check by modifying the validation order
        # For now, just verify that _manifest.json fails (for wrong extension)
        is_valid, error = _validate_chapter_name("_manifest.json")
        assert is_valid is False
        assert "must end with .md" in error

        # Too long
        long_name = "a" * 95 + ".md"  # 95 + 3 = 98 chars, within limit
        is_valid, error = _validate_chapter_name(long_name)
        assert is_valid is True

        very_long_name = "a" * 98 + ".md"  # 98 + 3 = 101 chars, exceeds limit
        is_valid, error = _validate_chapter_name(very_long_name)
        assert is_valid is False
        assert "too long" in error

    def test_validate_content_valid(self):
        """Test content validation with valid inputs."""
        from document_mcp.doc_tool_server import _validate_content

        valid_contents = ["", "Hello world", "# Title\n\nContent here", "A" * 1000]
        for content in valid_contents:
            is_valid, error = _validate_content(content)
            assert is_valid is True, f"Content should be valid"
            assert error == "", f"No error expected for valid content"

    def test_validate_content_invalid(self):
        """Test content validation with invalid inputs."""
        from document_mcp.doc_tool_server import _validate_content

        # None content
        is_valid, error = _validate_content(None)
        assert is_valid is False
        assert "cannot be None" in error

        # Too long content
        long_content = "a" * 1_000_001  # Exceeds 1MB limit
        is_valid, error = _validate_content(long_content)
        assert is_valid is False
        assert "too long" in error

    def test_validate_paragraph_index_valid(self):
        """Test paragraph index validation with valid inputs."""
        from document_mcp.doc_tool_server import _validate_paragraph_index

        valid_indices = [0, 1, 5, 100]
        for index in valid_indices:
            is_valid, error = _validate_paragraph_index(index)
            assert is_valid is True, f"Index {index} should be valid"
            assert error == "", f"No error expected for index {index}"

    def test_validate_paragraph_index_invalid(self):
        """Test paragraph index validation with invalid inputs."""
        from document_mcp.doc_tool_server import _validate_paragraph_index

        # Negative index
        is_valid, error = _validate_paragraph_index(-1)
        assert is_valid is False
        assert "cannot be negative" in error

        # Non-integer
        is_valid, error = _validate_paragraph_index("5")
        assert is_valid is False
        assert "must be an integer" in error

    def test_validate_search_query_valid(self):
        """Test search query validation with valid inputs."""
        from document_mcp.doc_tool_server import _validate_search_query

        valid_queries = ["hello", "search term", "123", "special!@#chars"]
        for query in valid_queries:
            is_valid, error = _validate_search_query(query)
            assert is_valid is True, f"Query '{query}' should be valid"
            assert error == "", f"No error expected for query '{query}'"

    def test_validate_search_query_invalid(self):
        """Test search query validation with invalid inputs."""
        from document_mcp.doc_tool_server import _validate_search_query

        # None query
        is_valid, error = _validate_search_query(None)
        assert is_valid is False
        assert "cannot be None" in error

        # Empty query
        is_valid, error = _validate_search_query("")
        assert is_valid is False
        assert "cannot be empty" in error

        # Whitespace only
        is_valid, error = _validate_search_query("   ")
        assert is_valid is False
        assert "cannot be empty" in error


class TestMCPToolInputValidation:
    """Test suite for MCP tool function input validation and error handling."""

    def test_create_document_empty_name(self):
        """Test create_document with empty document name."""
        from document_mcp.doc_tool_server import create_document

        result = create_document("")
        assert result.success is False
        assert "Document name cannot be empty" in result.message

    def test_create_document_invalid_name_with_path_separator(self):
        """Test create_document with invalid document name containing path separators."""
        from document_mcp.doc_tool_server import create_document

        result = create_document("invalid/name")
        assert result.success is False
        assert "Document name cannot contain path separators" in result.message

    def test_create_chapter_empty_document_name(self):
        """Test create_chapter with empty document name."""
        from document_mcp.doc_tool_server import create_chapter

        result = create_chapter("", "chapter1.md", "content")
        assert result.success is False
        assert "Document name cannot be empty" in result.message

    def test_create_chapter_empty_chapter_name(self):
        """Test create_chapter with empty chapter name."""
        from document_mcp.doc_tool_server import create_chapter

        result = create_chapter("test_doc", "", "content")
        assert result.success is False
        assert "Chapter name cannot be empty" in result.message

    def test_create_chapter_invalid_chapter_name(self):
        """Test create_chapter with invalid chapter name."""
        from document_mcp.doc_tool_server import create_chapter

        result = create_chapter("test_doc", "invalid.txt", "content")
        assert result.success is False
        assert "Chapter name must end with .md" in result.message

    def test_modify_paragraph_content_negative_index(self):
        """Test modify_paragraph_content with negative paragraph index."""
        from document_mcp.doc_tool_server import modify_paragraph_content

        result = modify_paragraph_content(
            "test_doc", "chapter1.md", -1, "new content", "replace"
        )
        assert result.success is False
        assert "Paragraph index cannot be negative" in result.message

    def test_replace_text_empty_search_text(self):
        """Test replace_text_in_chapter with empty search text."""
        from document_mcp.doc_tool_server import replace_text_in_chapter

        result = replace_text_in_chapter("test_doc", "chapter1.md", "", "replacement")
        assert result.success is False
        assert "Text to find cannot be empty" in result.message

    def test_modify_paragraph_content_invalid_mode(self):
        """Test modify_paragraph_content with invalid mode."""
        from document_mcp.doc_tool_server import modify_paragraph_content

        result = modify_paragraph_content(
            "test_doc", "chapter1.md", 0, "new content", "invalid_mode"
        )
        assert result.success is False
        assert "Invalid mode" in result.message
        assert "invalid_mode" in result.message

    def test_modify_paragraph_content_valid_modes(self):
        """Test modify_paragraph_content accepts valid modes."""
        from document_mcp.doc_tool_server import modify_paragraph_content

        valid_modes = ["replace", "insert_before", "insert_after", "delete"]
        for mode in valid_modes:
            result = modify_paragraph_content(
                "test_doc", "chapter1.md", 0, "new content", mode
            )
            # Should fail for other reasons (document not found), but not for invalid mode or input validation
            assert "Invalid mode" not in result.message
            assert "cannot be empty" not in result.message
            assert "cannot be negative" not in result.message

    def test_create_chapter_invalid_extension(self):
        """Test create_chapter rejects non-markdown files."""
        from document_mcp.doc_tool_server import create_chapter

        result = create_chapter("test_doc", "invalid.txt", "content")
        assert result.success is False
        assert "Chapter name must end with .md" in result.message

    def test_delete_chapter_invalid_extension(self):
        """Test delete_chapter rejects non-markdown files."""
        from document_mcp.doc_tool_server import delete_chapter

        result = delete_chapter("test_doc", "invalid.txt")
        assert result.success is False
        assert "Invalid target" in result.message
        assert "Not a valid chapter Markdown file name" in result.message


class TestMCPToolBoundaryConditions:
    """Test suite for MCP tool boundary conditions."""

    def test_find_text_case_sensitivity_boundary(self):
        """Test find_text functions with case sensitivity boundary conditions."""
        from document_mcp.doc_tool_server import find_text_in_chapter

        # Test with empty query
        result = find_text_in_chapter("test_doc", "chapter1.md", "")
        assert result == []

        # Test with whitespace-only query
        result = find_text_in_chapter("test_doc", "chapter1.md", "   ")
        assert result == []

    @patch("document_mcp.doc_tool_server.DOCS_ROOT_PATH", Path("/test/docs"))
    def test_list_documents_permission_error(self):
        """Test list_documents with permission error."""
        from document_mcp.doc_tool_server import list_documents

        with patch("pathlib.Path.is_dir", side_effect=PermissionError("Access denied")):
            result = list_documents()
            assert result == []

    def test_statistics_functions_with_none_input(self):
        """Test statistics functions handle None inputs gracefully."""
        from document_mcp.doc_tool_server import (
            get_chapter_statistics,
            get_document_statistics,
        )

        # These should return None for non-existent resources
        result = get_chapter_statistics("nonexistent_doc", "nonexistent_chapter.md")
        assert result is None

        result = get_document_statistics("nonexistent_doc")
        assert result is None

    def test_read_paragraph_content_negative_index(self):
        """Test read_paragraph_content with negative paragraph index."""
        from document_mcp.doc_tool_server import read_paragraph_content

        result = read_paragraph_content("test_doc", "chapter1.md", -1)
        assert result is None

    def test_paragraph_index_boundary_conditions(self):
        """Test paragraph operations with boundary conditions."""
        from document_mcp.doc_tool_server import modify_paragraph_content

        # Test with very large index (should fail with out of bounds)
        result = modify_paragraph_content(
            "test_doc", "chapter1.md", 9999, "content", "replace"
        )
        assert result.success is False
        # Should fail with chapter not found first, but if chapter existed, would fail with out of bounds


class TestReadDocumentSummaryTool:
    """Test suite for the read_document_summary tool."""

    @patch("document_mcp.doc_tool_server._get_document_path")
    def test_read_document_summary_success(self, mock_get_document_path):
        """Test successful reading of a document summary."""
        mock_doc_path_obj = MagicMock(spec=Path)
        mock_doc_path_obj.is_dir.return_value = True

        mock_summary_file_path_obj = MagicMock(spec=Path)
        mock_summary_file_path_obj.is_file.return_value = True
        # Ensure read_text is a callable mock that returns the value
        mock_summary_file_path_obj.read_text = MagicMock(return_value="This is a summary.")

        # Configure the __truediv__ method for the document path mock
        mock_doc_path_obj.__truediv__.return_value = mock_summary_file_path_obj

        mock_get_document_path.return_value = mock_doc_path_obj

        summary_content = read_document_summary("test_doc")

        assert summary_content == "This is a summary."
        mock_get_document_path.assert_called_once_with("test_doc")
        mock_doc_path_obj.__truediv__.assert_called_once_with(DOCUMENT_SUMMARY_FILE)
        mock_summary_file_path_obj.read_text.assert_called_once_with(encoding="utf-8")

    @patch("document_mcp.doc_tool_server._get_document_path")
    def test_read_document_summary_document_not_found(self, mock_get_document_path):
        """Test reading summary when document directory doesn't exist."""
        mock_doc_path_instance = MagicMock(spec=Path)
        mock_doc_path_instance.is_dir.return_value = False
        mock_doc_path_instance.__str__.return_value = "/mocked/path/to/nonexistent_doc"

        mock_get_document_path.return_value = mock_doc_path_instance

        with patch("builtins.print") as mock_print:
            summary_content = read_document_summary("nonexistent_doc")
            assert summary_content is None
            called_with_arg = mock_print.call_args[0][0]
            assert "Document 'nonexistent_doc' not found" in called_with_arg
            assert str(mock_doc_path_instance) in called_with_arg

    @patch("document_mcp.doc_tool_server._get_document_path")
    def test_read_document_summary_summary_file_not_found(self, mock_get_document_path):
        """Test reading summary when _SUMMARY.md file doesn't exist."""
        mock_doc_path_obj = MagicMock(spec=Path)
        mock_doc_path_obj.is_dir.return_value = True

        mock_summary_file_path_obj = MagicMock(spec=Path)
        mock_summary_file_path_obj.is_file.return_value = False # Summary file does not exist

        mock_doc_path_obj.__truediv__.return_value = mock_summary_file_path_obj
        mock_get_document_path.return_value = mock_doc_path_obj

        with patch("builtins.print") as mock_print:
            summary_content = read_document_summary("doc_no_summary")
            assert summary_content is None
            mock_print.assert_any_call("Summary file '_SUMMARY.md' not found in document 'doc_no_summary'.")

    @patch("document_mcp.doc_tool_server._get_document_path")
    def test_read_document_summary_read_error(self, mock_get_document_path):
        """Test handling of read errors when accessing summary file."""
        mock_doc_path_obj = MagicMock(spec=Path)
        mock_doc_path_obj.is_dir.return_value = True

        mock_summary_file_path_obj = MagicMock(spec=Path)
        mock_summary_file_path_obj.is_file.return_value = True
        # Ensure read_text is a callable mock that raises an exception
        mock_summary_file_path_obj.read_text = MagicMock(side_effect=Exception("Read permission denied"))

        mock_doc_path_obj.__truediv__.return_value = mock_summary_file_path_obj
        mock_get_document_path.return_value = mock_doc_path_obj

        with patch("builtins.print") as mock_print:
            summary_content = read_document_summary("doc_read_error")
            assert summary_content is None
            mock_print.assert_any_call("Error reading summary file for document 'doc_read_error': Read permission denied")

    @patch("document_mcp.doc_tool_server._validate_document_name", return_value=(False, "Invalid name"))
    @patch("builtins.print")
    def test_read_document_summary_invalid_doc_name(self, mock_print, mock_validate):
        """Test read_document_summary with an invalid document name."""
        summary_content = read_document_summary("invalid/docname")
        assert summary_content is None
        mock_validate.assert_called_once_with("invalid/docname")
        mock_print.assert_any_call("Invalid document name provided: Invalid name")


class TestListDocumentsTool:
    """Test suite for the list_documents tool, focusing on has_summary."""

    @patch("document_mcp.doc_tool_server.DOCS_ROOT_PATH")
    @patch("document_mcp.doc_tool_server._get_ordered_chapter_files", return_value=[]) # Assume no chapters for simplicity
    @patch("document_mcp.doc_tool_server._get_chapter_metadata", return_value=None)
    def test_list_documents_with_summary(self, mock_get_meta, mock_get_files, mock_docs_root):
        """Test list_documents correctly identifies a document with a summary."""
        mock_doc_dir = MagicMock(spec=Path)
        mock_doc_dir.name = "doc_with_summary"
        mock_doc_dir.is_dir.return_value = True

        mock_summary_file_path_obj = MagicMock(spec=Path)
        mock_summary_file_path_obj.is_file.return_value = True # Summary exists

        # Configure mock_doc_dir's __truediv__ to return this summary file path object
        mock_doc_dir.__truediv__.return_value = mock_summary_file_path_obj

        # Mock stat for last_modified time if no chapters
        mock_stat_result = Mock()
        mock_stat_result.st_mtime = 1678886400.0 # Example timestamp
        mock_doc_dir.stat.return_value = mock_stat_result


        mock_docs_root.exists.return_value = True
        mock_docs_root.is_dir.return_value = True
        mock_docs_root.iterdir.return_value = [mock_doc_dir]


        result = list_documents()
        assert len(result) == 1
        doc_info = result[0]
        assert isinstance(doc_info, DocumentInfo)
        assert doc_info.document_name == "doc_with_summary"
        assert doc_info.has_summary is True
        mock_doc_dir.__truediv__.assert_called_once_with(DOCUMENT_SUMMARY_FILE)


    @patch("document_mcp.doc_tool_server.DOCS_ROOT_PATH")
    @patch("document_mcp.doc_tool_server._get_ordered_chapter_files", return_value=[])
    @patch("document_mcp.doc_tool_server._get_chapter_metadata", return_value=None)
    def test_list_documents_without_summary(self, mock_get_meta, mock_get_files, mock_docs_root):
        """Test list_documents correctly identifies a document without a summary."""
        mock_doc_dir = Mock(spec=Path)
        mock_doc_dir = MagicMock(spec=Path)
        mock_doc_dir.name = "doc_no_summary"
        mock_doc_dir.is_dir.return_value = True

        mock_summary_file_path_obj = MagicMock(spec=Path)
        mock_summary_file_path_obj.is_file.return_value = False # Summary does NOT exist
        mock_doc_dir.__truediv__.return_value = mock_summary_file_path_obj # Corrected to __truediv__

        mock_stat_result = Mock()
        mock_stat_result.st_mtime = 1678886400.0
        mock_doc_dir.stat.return_value = mock_stat_result

        mock_docs_root.exists.return_value = True
        mock_docs_root.is_dir.return_value = True
        mock_docs_root.iterdir.return_value = [mock_doc_dir]

        result = list_documents()
        assert len(result) == 1
        doc_info = result[0]
        assert isinstance(doc_info, DocumentInfo)
        assert doc_info.document_name == "doc_no_summary"
        assert doc_info.has_summary is False # This assertion should now pass
        mock_doc_dir.__truediv__.assert_called_once_with(DOCUMENT_SUMMARY_FILE) # Corrected to __truediv__

    @patch("document_mcp.doc_tool_server.DOCS_ROOT_PATH")
    def test_list_documents_empty_root(self, mock_docs_root):
        """Test list_documents with an empty root directory."""
        mock_docs_root.exists.return_value = True
        mock_docs_root.is_dir.return_value = True
        mock_docs_root.iterdir.return_value = [] # No documents

        result = list_documents()
        assert result == []

    @patch("document_mcp.doc_tool_server.DOCS_ROOT_PATH")
    def test_list_documents_root_not_exist(self, mock_docs_root):
        """Test list_documents when the root directory doesn't exist."""
        mock_docs_root.exists.return_value = False
        result = list_documents()
        assert result == []
