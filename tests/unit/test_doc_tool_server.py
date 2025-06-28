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
# from unittest.mock import Mock, MagicMock  # Remove this import

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

    def test_get_document_path(self, mock_path_operations):
        """Test document path generation."""
        mock_path_operations.mock_docs_root_path("/test/docs")
        result = _get_document_path("my_document")
        expected = Path("/test/docs/my_document")
        assert result == expected

    def test_get_document_path_with_special_chars(self, mock_path_operations):
        """Test document path generation with special characters."""
        mock_path_operations.mock_docs_root_path("/test/docs")
        result = _get_document_path("my-document_2023")
        expected = Path("/test/docs/my-document_2023")
        assert result == expected

    def test_get_chapter_path(self, mock_path_operations):
        """Test chapter path generation."""
        mock_path_operations.mock_docs_root_path("/test/docs")
        result = _get_chapter_path("my_document", "chapter1.md")
        expected = Path("/test/docs/my_document/chapter1.md")
        assert result == expected

    def test_get_chapter_path_with_subdirectory(self, mock_path_operations):
        """Test chapter path generation with document containing special characters."""
        mock_path_operations.mock_docs_root_path("/test/docs")
        result = _get_chapter_path("project-alpha_v2", "01-intro.md")
        expected = Path("/test/docs/project-alpha_v2/01-intro.md")
        assert result == expected


class TestFileOrdering:
    """Test suite for file ordering functions."""

    def test_get_ordered_chapter_files_nonexistent_document(self, mock_path_operations):
        """Test getting chapter files for non-existent document."""
        mock_path_operations.mock_docs_root_path("/test/docs")
        
        # Create a mock path that returns False for is_dir
        mock_doc_path = mock_path_operations.create_mock_file("nonexistent_doc", is_file=False, is_dir=False)
        mock_path_operations.mock_document_path("nonexistent_doc", mock_doc_path)
        
        result = _get_ordered_chapter_files("nonexistent_doc")
        assert result == []

    def test_get_ordered_chapter_files_empty_document(self, mock_path_operations, mocker):
        """Test getting chapter files for empty document directory."""
        mock_path_operations.mock_docs_root_path("/test/docs")
        
        mock_doc_path = mocker.Mock()
        mock_doc_path.is_dir.return_value = True
        mock_doc_path.iterdir.return_value = []
        
        mock_path_operations.mock_document_path("empty_doc", mock_doc_path)
        
        result = _get_ordered_chapter_files("empty_doc")
        assert result == []

    def test_get_ordered_chapter_files_with_valid_chapters(self, mock_path_operations, mocker):
        """Test getting chapter files with valid markdown files."""
        mock_path_operations.mock_docs_root_path("/test/docs")
        
        # Create mock file objects with proper __lt__ method for sorting
        chapter1 = mocker.Mock()
        chapter1.name = "01-intro.md"
        chapter1.is_file.return_value = True
        chapter1.__lt__ = lambda self, other: self.name < other.name

        chapter2 = mocker.Mock()
        chapter2.name = "02-methods.md"
        chapter2.is_file.return_value = True
        chapter2.__lt__ = lambda self, other: self.name < other.name

        chapter3 = mocker.Mock()
        chapter3.name = "03-results.md"
        chapter3.is_file.return_value = True
        chapter3.__lt__ = lambda self, other: self.name < other.name

        # Mock the document path
        mock_doc_path = mocker.Mock()
        mock_doc_path.is_dir.return_value = True
        mock_doc_path.iterdir.return_value = [
            chapter3,
            chapter1,
            chapter2,
        ]  # Unsorted order

        mock_path_operations.mock_document_path("test_doc", mock_doc_path)
        
        result = _get_ordered_chapter_files("test_doc")

        # Should be sorted alphabetically
        assert len(result) == 3
        assert result[0].name == "01-intro.md"
        assert result[1].name == "02-methods.md"
        assert result[2].name == "03-results.md"

    def test_get_ordered_chapter_files_filters_non_md_files(self, mock_path_operations, mocker):
        """Test that non-markdown files are filtered out."""
        mock_path_operations.mock_docs_root_path("/test/docs")
        
        # Create mock file objects
        chapter1 = mocker.Mock()
        chapter1.name = "01-intro.md"
        chapter1.is_file.return_value = True

        text_file = mocker.Mock()
        text_file.name = "notes.txt"
        text_file.is_file.return_value = True

        summary_file = mocker.Mock()
        summary_file.name = "_SUMMARY.md"
        summary_file.is_file.return_value = True

        manifest_file = mocker.Mock()
        manifest_file.name = "_manifest.json"
        manifest_file.is_file.return_value = True

        directory = mocker.Mock()
        directory.name = "images"
        directory.is_file.return_value = False

        # Mock the document path
        mock_doc_path = mocker.Mock()
        mock_doc_path.is_dir.return_value = True
        mock_doc_path.iterdir.return_value = [
            chapter1,
            text_file,
            manifest_file,
            summary_file, # Add summary file to the list of items
            directory,
        ]

        mock_path_operations.mock_document_path("test_doc", mock_doc_path)
        
        result = _get_ordered_chapter_files("test_doc")

        # Should only include the markdown file
        assert len(result) == 1
        assert result[0].name == "01-intro.md"

    def test_get_ordered_chapter_files_filters_summary_file(self, mock_path_operations, mocker):
        """Test that _SUMMARY.md files are filtered out."""
        mock_path_operations.mock_docs_root_path("/test/docs")
        
        chapter1 = mocker.Mock()
        chapter1.name = "01-intro.md"
        chapter1.is_file.return_value = True

        summary_file = mocker.Mock()
        summary_file.name = "_SUMMARY.md"
        summary_file.is_file.return_value = True

        mock_doc_path = mocker.Mock()
        mock_doc_path.is_dir.return_value = True
        mock_doc_path.iterdir.return_value = [chapter1, summary_file]

        mock_path_operations.mock_document_path("test_doc_with_summary", mock_doc_path)
        
        result = _get_ordered_chapter_files("test_doc_with_summary")
        assert len(result) == 1
        assert result[0].name == "01-intro.md"


class TestChapterReading:
    """Test suite for chapter content reading functions."""

    def test_read_chapter_content_details_nonexistent_file(self, mocker):
        """Test reading content from non-existent file."""
        mock_path = mocker.Mock()
        mock_path.is_file.return_value = False

        result = _read_chapter_content_details("test_doc", mock_path)
        assert result is None

    def test_read_chapter_content_details_valid_file(self, mocker):
        """Test reading content from valid file."""
        mock_path = mocker.Mock()
        mock_path.is_file.return_value = True
        mock_path.name = "01-intro.md"
        mock_path.read_text.return_value = "# Introduction\n\nThis is the first paragraph.\n\nThis is the second paragraph."

        # Mock the stat result
        mock_stat = mocker.Mock()
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

    def test_read_chapter_content_details_file_read_error(self, mock_file_operations, mocker):
        """Test handling file read errors."""
        mock_path = mocker.Mock()
        mock_path.is_file.return_value = True
        mock_path.name = "corrupted.md"
        mock_path.read_text.side_effect = Exception("File read error")

        mock_print = mock_file_operations.mock_print()
        
        result = _read_chapter_content_details("test_doc", mock_path)

        assert result is None
        mock_print.assert_called_once()
        assert "Error reading chapter file" in mock_print.call_args[0][0]

    def test_get_chapter_metadata_nonexistent_file(self, mocker):
        """Test getting metadata from non-existent file."""
        mock_path = mocker.Mock()
        mock_path.is_file.return_value = False

        result = _get_chapter_metadata("test_doc", mock_path)
        assert result is None

    def test_get_chapter_metadata_valid_file(self, mocker):
        """Test getting metadata from valid file."""
        mock_path = mocker.Mock()
        mock_path.is_file.return_value = True
        mock_path.name = "02-methods.md"
        mock_path.read_text.return_value = "# Methods\n\nFirst paragraph about methods.\n\nSecond paragraph with details.\n\nThird paragraph conclusion."

        # Mock the stat result
        mock_stat = mocker.Mock()
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

    def test_get_chapter_metadata_empty_file(self, mocker):
        """Test getting metadata from empty file."""
        mock_path = mocker.Mock()
        mock_path.is_file.return_value = True
        mock_path.name = "empty.md"
        mock_path.read_text.return_value = ""

        # Mock the stat result
        mock_stat = mocker.Mock()
        mock_stat.st_mtime = 1640995200.0
        mock_path.stat.return_value = mock_stat

        result = _get_chapter_metadata("test_doc", mock_path)

        assert result is not None
        assert result.chapter_name == "empty.md"
        assert result.word_count == 0
        assert result.paragraph_count == 0

    def test_get_chapter_metadata_file_read_error(self, mock_file_operations, mocker):
        """Test handling file read errors in metadata extraction."""
        mock_path = mocker.Mock()
        mock_path.is_file.return_value = True
        mock_path.name = "error.md"
        mock_path.read_text.side_effect = Exception("Permission denied")

        mock_print = mock_file_operations.mock_print()
        
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

    def test_list_documents_permission_error(self, mock_path_operations, mocker):
        """Test list_documents with permission error."""
        from document_mcp.doc_tool_server import list_documents

        mock_path_operations.mock_docs_root_path("/test/docs")
        # Mock pathlib.Path.is_dir to raise PermissionError
        mocker.patch("pathlib.Path.is_dir", side_effect=PermissionError("Access denied"))
        
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

    def test_read_document_summary_success(self, mock_path_operations, mock_file_operations, mocker):
        """Test successful reading of a document summary."""
        # Create mock document directory with MagicMock to support __truediv__
        mock_doc_path = mocker.MagicMock(spec=Path)
        mock_doc_path.is_dir.return_value = True
        
        # Create mock summary file
        mock_summary_file = mocker.MagicMock(spec=Path)
        mock_summary_file.is_file.return_value = True
        mock_summary_file.read_text.return_value = "This is a summary."
        
        # Configure path division to return summary file
        mock_doc_path.__truediv__.return_value = mock_summary_file
        
        # Set up mocks
        mock_path_operations.mock_document_path("test_doc", mock_doc_path)

        summary_content = read_document_summary("test_doc")

        assert summary_content == "This is a summary."
        mock_summary_file.read_text.assert_called_once_with(encoding="utf-8")

    def test_read_document_summary_document_not_found(self, mock_path_operations, mock_file_operations, mocker):
        """Test reading summary when document directory doesn't exist."""
        mock_doc_path = mocker.MagicMock(spec=Path)
        mock_doc_path.is_dir.return_value = False
        mock_doc_path.__str__.return_value = "/mocked/path/to/nonexistent_doc"
        
        mock_path_operations.mock_document_path("nonexistent_doc", mock_doc_path)
        mock_print = mock_file_operations.mock_print()

        summary_content = read_document_summary("nonexistent_doc")
        
        assert summary_content is None
        called_with_arg = mock_print.call_args[0][0]
        assert "Document 'nonexistent_doc' not found" in called_with_arg
        assert str(mock_doc_path) in called_with_arg

    def test_read_document_summary_summary_file_not_found(self, mock_path_operations, mock_file_operations, mocker):
        """Test reading summary when _SUMMARY.md file doesn't exist."""
        mock_doc_path = mocker.MagicMock(spec=Path)
        mock_doc_path.is_dir.return_value = True
        
        mock_summary_file = mocker.MagicMock(spec=Path)
        mock_summary_file.is_file.return_value = False  # Summary file does not exist
        
        mock_doc_path.__truediv__.return_value = mock_summary_file
        mock_path_operations.mock_document_path("doc_no_summary", mock_doc_path)
        mock_print = mock_file_operations.mock_print()

        summary_content = read_document_summary("doc_no_summary")
        
        assert summary_content is None
        mock_print.assert_any_call("Summary file '_SUMMARY.md' not found in document 'doc_no_summary'.")

    def test_read_document_summary_read_error(self, mock_path_operations, mock_file_operations, mocker):
        """Test handling of read errors when accessing summary file."""
        mock_doc_path = mocker.MagicMock(spec=Path)
        mock_doc_path.is_dir.return_value = True
        
        mock_summary_file = mocker.MagicMock(spec=Path)
        mock_summary_file.is_file.return_value = True
        mock_summary_file.read_text.side_effect = Exception("Read permission denied")
        
        mock_doc_path.__truediv__.return_value = mock_summary_file
        mock_path_operations.mock_document_path("doc_read_error", mock_doc_path)
        mock_print = mock_file_operations.mock_print()

        summary_content = read_document_summary("doc_read_error")
        
        assert summary_content is None
        mock_print.assert_any_call("Error reading summary file for document 'doc_read_error': Read permission denied")

    def test_read_document_summary_invalid_doc_name(self, mock_validation_operations, mock_file_operations):
        """Test read_document_summary with an invalid document name."""
        mock_validation_operations.mock_validate_document_name(False, "Invalid name")
        mock_print = mock_file_operations.mock_print()
        
        summary_content = read_document_summary("invalid/docname")
        
        assert summary_content is None
        mock_print.assert_any_call("Invalid document name provided: Invalid name")


class TestListDocumentsTool:
    """Test suite for the list_documents tool, focusing on has_summary."""

    def test_list_documents_with_summary(self, mock_path_operations, mocker):
        """Test list_documents correctly identifies a document with a summary."""
        # Mock the dependencies
        mock_get_files = mocker.patch("document_mcp.doc_tool_server._get_ordered_chapter_files", return_value=[])
        mock_get_meta = mocker.patch("document_mcp.doc_tool_server._get_chapter_metadata", return_value=None)
        
        # Mock the actual Path methods at the module level
        real_docs_root = mock_path_operations.mock_docs_root_path("/test/docs")
        
        # Create a real Path object for the document directory
        doc_dir_path = real_docs_root / "doc_with_summary"
        
        # Mock file system operations
        mocker.patch.object(Path, 'exists', return_value=True)
        mocker.patch.object(Path, 'is_dir', return_value=True)
        
        # Mock iterdir to return our document directory
        mock_iterdir = mocker.patch.object(Path, 'iterdir')
        mock_iterdir.return_value = [doc_dir_path]
        
        # Mock is_file for the summary file check
        def is_file_side_effect(self):
            # Return True only for the summary file
            return str(self).endswith(DOCUMENT_SUMMARY_FILE)
        
        mocker.patch.object(Path, 'is_file', side_effect=is_file_side_effect, autospec=True)
        
        # Mock stat for last modified time
        mock_stat = mocker.Mock()
        mock_stat.st_mtime = 1678886400.0
        mocker.patch.object(Path, 'stat', return_value=mock_stat)

        result = list_documents()
        assert len(result) == 1
        doc_info = result[0]
        assert isinstance(doc_info, DocumentInfo)
        assert doc_info.document_name == "doc_with_summary"
        assert doc_info.has_summary is True

    def test_list_documents_without_summary(self, mock_path_operations, mocker):
        """Test list_documents correctly identifies a document without a summary."""
        # Mock the dependencies
        mock_get_files = mocker.patch("document_mcp.doc_tool_server._get_ordered_chapter_files", return_value=[])
        mock_get_meta = mocker.patch("document_mcp.doc_tool_server._get_chapter_metadata", return_value=None)
        
        # Mock the actual Path methods at the module level
        real_docs_root = mock_path_operations.mock_docs_root_path("/test/docs")
        
        # Create a real Path object for the document directory
        doc_dir_path = real_docs_root / "doc_no_summary"
        
        # Mock file system operations
        mocker.patch.object(Path, 'exists', return_value=True)
        mocker.patch.object(Path, 'is_dir', return_value=True)
        
        # Mock iterdir to return our document directory
        mock_iterdir = mocker.patch.object(Path, 'iterdir')
        mock_iterdir.return_value = [doc_dir_path]
        
        # Mock is_file to return False for all files (no summary)
        mocker.patch.object(Path, 'is_file', return_value=False)
        
        # Mock stat for last modified time
        mock_stat = mocker.Mock()
        mock_stat.st_mtime = 1678886400.0
        mocker.patch.object(Path, 'stat', return_value=mock_stat)

        result = list_documents()
        assert len(result) == 1
        doc_info = result[0]
        assert isinstance(doc_info, DocumentInfo)
        assert doc_info.document_name == "doc_no_summary"
        assert doc_info.has_summary is False

    def test_list_documents_empty_root(self, mock_path_operations, mocker):
        """Test list_documents with an empty root directory."""
        real_docs_root = mock_path_operations.mock_docs_root_path("/test/docs")
        
        # Mock file system operations
        mocker.patch.object(Path, 'exists', return_value=True)
        mocker.patch.object(Path, 'is_dir', return_value=True)
        mocker.patch.object(Path, 'iterdir', return_value=[])  # No documents

        result = list_documents()
        assert result == []

    def test_list_documents_root_not_exist(self, mock_path_operations, mocker):
        """Test list_documents when the root directory doesn't exist."""
        real_docs_root = mock_path_operations.mock_docs_root_path("/test/docs")
        
        # Mock the exists check to return False
        mocker.patch.object(Path, 'exists', return_value=False)
        
        result = list_documents()
        assert result == []
