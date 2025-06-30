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
import pytest

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
    get_chapter_statistics,
    get_document_statistics,
    read_full_document,
    find_text_in_chapter,
    find_text_in_document,
    DocumentInfo,
    StatisticsReport,
    FullDocumentContent,
    ParagraphDetail,
    DOCUMENT_SUMMARY_FILE,
    CHAPTER_MANIFEST_FILE,
)
from document_mcp.logger_config import ErrorCategory


class TestStructuredErrorLogging:
    """Test suite for structured error logging functionality."""

    def test_read_chapter_content_details_logs_error_on_file_read_failure(self, mocker):
        """Logs error when file read fails in chapter content details."""
        mock_log_error = mocker.patch('document_mcp.doc_tool_server.log_structured_error')
        mock_path = mocker.Mock()
        mock_path.is_file.return_value = True
        mock_path.name = "test.md"
        mock_path.exists.return_value = True
        mock_path.read_text.side_effect = PermissionError("Permission denied")

        result = _read_chapter_content_details("test_doc", mock_path)

        assert result is None
        mock_log_error.assert_called_once()
        call_args = mock_log_error.call_args
        assert call_args[1]['category'] == ErrorCategory.ERROR
        assert call_args[1]['message'] == "Failed to read chapter file: test.md"
        assert call_args[1]['operation'] == "read_chapter_content"
        assert call_args[1]['context']['document_name'] == "test_doc"
        assert call_args[1]['context']['chapter_file_path'] == str(mock_path)
        assert call_args[1]['context']['file_exists'] == True
        assert isinstance(call_args[1]['exception'], PermissionError)

    def test_get_chapter_metadata_logs_error_on_file_read_failure(self, mocker):
        """Logs error when file read fails in chapter metadata."""
        mock_log_error = mocker.patch('document_mcp.doc_tool_server.log_structured_error')
        mock_path = mocker.Mock()
        mock_path.is_file.return_value = True
        mock_path.name = "metadata.md"
        mock_path.exists.return_value = True
        mock_path.read_text.side_effect = OSError("Disk I/O error")

        result = _get_chapter_metadata("test_doc", mock_path)

        assert result is None
        mock_log_error.assert_called_once()
        call_args = mock_log_error.call_args
        assert call_args[1]['category'] == ErrorCategory.ERROR
        assert call_args[1]['message'] == "Failed to get metadata for chapter: metadata.md"
        assert call_args[1]['operation'] == "get_chapter_metadata"
        assert call_args[1]['context']['document_name'] == "test_doc"
        assert call_args[1]['context']['chapter_file_path'] == str(mock_path)
        assert call_args[1]['context']['file_exists'] == True
        assert isinstance(call_args[1]['exception'], OSError)

    def test_read_document_summary_logs_warning_on_invalid_document_name(self, mocker):
        """Logs warning for invalid document name."""
        mock_log_error = mocker.patch('document_mcp.doc_tool_server.log_structured_error')
        mock_validate = mocker.patch('document_mcp.doc_tool_server._validate_document_name')
        mock_validate.return_value = (False, "Invalid document name")
        result = read_document_summary("invalid/doc")
        assert result is None
        mock_log_error.assert_called_once()
        call_args = mock_log_error.call_args
        assert call_args[1]['category'] == ErrorCategory.WARNING
        assert call_args[1]['message'] == "Invalid document name provided"
        assert call_args[1]['operation'] == "read_document_summary"
        assert call_args[1]['context']['document_name'] == "invalid/doc"
        assert call_args[1]['context']['validation_error'] == "Invalid document name"

    def test_read_document_summary_logs_info_on_document_not_found(self, mocker):
        """Logs info when document is not found."""
        mock_log_error = mocker.patch('document_mcp.doc_tool_server.log_structured_error')
        mock_validate = mocker.patch('document_mcp.doc_tool_server._validate_document_name')
        mock_get_path = mocker.patch('document_mcp.doc_tool_server._get_document_path')
        mock_validate.return_value = (True, "")
        mock_doc_path = mocker.Mock()
        mock_doc_path.is_dir.return_value = False
        mock_get_path.return_value = mock_doc_path
        result = read_document_summary("nonexistent_doc")
        assert result is None
        mock_log_error.assert_called_once()
        call_args = mock_log_error.call_args
        assert call_args[1]['category'] == ErrorCategory.INFO
        assert call_args[1]['message'] == "Document not found"
        assert call_args[1]['operation'] == "read_document_summary"
        assert call_args[1]['context']['document_name'] == "nonexistent_doc"
        assert call_args[1]['context']['attempted_path'] == str(mock_doc_path)

    def test_read_document_summary_logs_info_on_summary_file_not_found(self, mock_path_operations, mocker):
        """Logs info when summary file is missing."""
        mock_log_error = mocker.patch('document_mcp.doc_tool_server.log_structured_error')
        mock_doc_path = mocker.MagicMock(spec=Path)
        mock_doc_path.is_dir.return_value = True
        mock_summary_file = mocker.MagicMock(spec=Path)
        mock_summary_file.is_file.return_value = False
        mock_doc_path.__truediv__.return_value = mock_summary_file
        mock_path_operations.mock_document_path("doc_no_summary", mock_doc_path)
        summary_content = read_document_summary("doc_no_summary")
        assert summary_content is None
        mock_log_error.assert_called()
        call_args = mock_log_error.call_args
        assert call_args[1]['category'] == ErrorCategory.INFO
        assert call_args[1]['message'] == "Summary file not found in document"
        assert call_args[1]['operation'] == "read_document_summary"

    def test_read_document_summary_logs_error_on_file_read_failure(self, mocker):
        """Logs error when summary file read fails."""
        mock_log_error = mocker.patch('document_mcp.doc_tool_server.log_structured_error')
        mock_validate = mocker.patch('document_mcp.doc_tool_server._validate_document_name')
        mock_get_path = mocker.patch('document_mcp.doc_tool_server._get_document_path')
        mock_validate.return_value = (True, "")
        mock_doc_path = mocker.Mock()
        mock_doc_path.is_dir.return_value = True
        mock_summary_path = mocker.Mock()
        mock_summary_path.is_file.return_value = True
        mock_summary_path.read_text.side_effect = UnicodeDecodeError("utf-8", b"", 0, 1, "invalid start byte")
        mock_doc_path.__truediv__ = mocker.Mock(return_value=mock_summary_path)
        mock_get_path.return_value = mock_doc_path
        result = read_document_summary("doc_with_corrupt_summary")
        assert result is None
        mock_log_error.assert_called_once()
        call_args = mock_log_error.call_args
        assert call_args[1]['category'] == ErrorCategory.ERROR
        assert call_args[1]['message'] == "Failed to read summary file"
        assert call_args[1]['operation'] == "read_document_summary"
        assert call_args[1]['context']['document_name'] == "doc_with_corrupt_summary"
        assert call_args[1]['context']['summary_file_path'] == str(mock_summary_path)
        assert isinstance(call_args[1]['exception'], UnicodeDecodeError)

    def test_read_paragraph_content_logs_warning_on_index_out_of_bounds(self, mocker):
        """Logs warning when paragraph index is out of bounds."""
        mock_log_error = mocker.patch('document_mcp.doc_tool_server.log_structured_error')
        mock_read_chapter = mocker.patch('document_mcp.doc_tool_server.read_chapter_content')
        mock_chapter = mocker.Mock()
        mock_chapter.content = "First paragraph.\n\nSecond paragraph."
        mock_read_chapter.return_value = mock_chapter
        from document_mcp.doc_tool_server import read_paragraph_content
        result = read_paragraph_content("test_doc", "chapter.md", 5)
        assert result is None
        mock_log_error.assert_called_once()
        call_args = mock_log_error.call_args
        assert call_args[1]['category'] == ErrorCategory.WARNING
        assert call_args[1]['message'] == "Paragraph index out of bounds"
        assert call_args[1]['operation'] == "read_paragraph_content"
        assert call_args[1]['context']['document_name'] == "test_doc"
        assert call_args[1]['context']['chapter_name'] == "chapter.md"
        assert call_args[1]['context']['paragraph_index'] == 5
        assert call_args[1]['context']['total_paragraphs'] == 2
        assert call_args[1]['context']['valid_range'] == "0-1"

    def test_read_full_document_logs_info_on_document_not_found(self, mocker):
        """Logs info when full document is not found."""
        mock_log_error = mocker.patch('document_mcp.doc_tool_server.log_structured_error')
        mock_get_path = mocker.patch('document_mcp.doc_tool_server._get_document_path')
        mock_doc_path = mocker.Mock()
        mock_doc_path.is_dir.return_value = False
        mock_get_path.return_value = mock_doc_path
        result = read_full_document("missing_doc")
        assert result is None
        mock_log_error.assert_called_once()
        call_args = mock_log_error.call_args
        assert call_args[1]['category'] == ErrorCategory.INFO
        assert call_args[1]['message'] == "Document not found for full read"
        assert call_args[1]['operation'] == "read_full_document"
        assert call_args[1]['context']['document_name'] == "missing_doc"
        assert call_args[1]['context']['attempted_path'] == str(mock_doc_path)

    def test_read_full_document_logs_warning_on_unreadable_chapter(self, mocker):
        """Logs warning when a chapter cannot be read during full document read."""
        mock_log_error = mocker.patch('document_mcp.doc_tool_server.log_structured_error')
        mock_get_path = mocker.patch('document_mcp.doc_tool_server._get_document_path')
        mock_get_files = mocker.patch('document_mcp.doc_tool_server._get_ordered_chapter_files')
        mock_read_details = mocker.patch('document_mcp.doc_tool_server._read_chapter_content_details')
        mock_doc_path = mocker.Mock()
        mock_doc_path.is_dir.return_value = True
        mock_get_path.return_value = mock_doc_path
        mock_chapter_path = mocker.Mock()
        mock_chapter_path.name = "corrupted.md"
        mock_get_files.return_value = [mock_chapter_path]
        mock_read_details.return_value = None
        result = read_full_document("doc_with_corrupted_chapter")
        assert result is not None
        assert len(result.chapters) == 0
        mock_log_error.assert_called_once()
        call_args = mock_log_error.call_args
        assert call_args[1]['category'] == ErrorCategory.WARNING
        assert call_args[1]['message'] == "Could not read chapter file, skipping"
        assert call_args[1]['operation'] == "read_full_document"
        assert call_args[1]['context']['document_name'] == "doc_with_corrupted_chapter"
        assert call_args[1]['context']['chapter_file_name'] == "corrupted.md"
        assert call_args[1]['context']['chapter_file_path'] == str(mock_chapter_path)


class TestMCPLoggerDecorator:
    """Test suite for MCP logger decorator error handling."""

    def test_log_mcp_call_decorator_logs_function_errors(self, mocker):
        """Test that the log_mcp_call decorator catches and logs function errors."""
        mock_log_error = mocker.patch('document_mcp.logger_config.log_structured_error')
        from document_mcp.logger_config import log_mcp_call
        
        @log_mcp_call
        def failing_function(arg1, **kwargs):
            raise ValueError("Test error message")
        
        with pytest.raises(ValueError):
            failing_function("test_arg", kwarg1="test_kwarg")
        
        # Verify structured error logging was called
        mock_log_error.assert_called()
        call_args = mock_log_error.call_args
        assert call_args[1]['category'] == ErrorCategory.ERROR
        assert call_args[1]['message'] == "Tool execution failed: failing_function"
        assert call_args[1]['operation'] == "tool_execution"
        assert call_args[1]['function'] == "failing_function"
        assert isinstance(call_args[1]['exception'], ValueError)

    def test_log_mcp_call_decorator_logs_argument_serialization_errors(self, mocker):
        """Test that argument serialization errors are logged appropriately."""
        mock_log_error = mocker.patch('document_mcp.logger_config.log_structured_error')
        from document_mcp.logger_config import log_mcp_call
        
        # Create an object that will cause serialization issues
        class UnserializableObject:
            def __repr__(self):
                raise RuntimeError("Cannot serialize this object")
        
        @log_mcp_call
        def function_with_bad_args(bad_arg):
            return "success"
        
        result = function_with_bad_args(UnserializableObject())
        
        assert result == "success"  # Function should still execute
        
        # Check that argument serialization error was logged
        mock_log_error.assert_called()
        call_args = mock_log_error.call_args
        assert call_args[1]['category'] == ErrorCategory.WARNING
        assert call_args[1]['message'] == "Failed to serialize function arguments for logging"
        assert call_args[1]['operation'] == "argument_serialization"
        assert call_args[1]['function'] == "function_with_bad_args"


class TestErrorCategorization:
    """Test suite for proper error categorization."""

    def test_file_not_found_categorized_as_info(self, mocker):
        """Test that file not found errors are categorized as INFO."""
        mock_log_error = mocker.patch('document_mcp.doc_tool_server.log_structured_error')
        mock_validate = mocker.patch('document_mcp.doc_tool_server._validate_document_name')
        mock_get_path = mocker.patch('document_mcp.doc_tool_server._get_document_path')
        
        mock_validate.return_value = (True, "")
        mock_doc_path = mocker.Mock()
        mock_doc_path.is_dir.return_value = False
        mock_get_path.return_value = mock_doc_path
        
        read_document_summary("nonexistent")
        
        call_args = mock_log_error.call_args
        assert call_args[1]['category'] == ErrorCategory.INFO

    def test_validation_errors_categorized_as_warning(self, mocker):
        """Test that validation errors are categorized as WARNING."""
        mock_log_error = mocker.patch('document_mcp.doc_tool_server.log_structured_error')
        mock_validate = mocker.patch('document_mcp.doc_tool_server._validate_document_name')
        
        mock_validate.return_value = (False, "Invalid name")
        
        read_document_summary("invalid/name")
        
        call_args = mock_log_error.call_args
        assert call_args[1]['category'] == ErrorCategory.WARNING

    def test_io_errors_categorized_as_error(self, mocker):
        """Test that I/O errors are categorized as ERROR."""
        mock_log_error = mocker.patch('document_mcp.doc_tool_server.log_structured_error')
        mock_path = mocker.Mock()
        mock_path.is_file.return_value = True
        mock_path.name = "test.md"
        mock_path.exists.return_value = True
        mock_path.read_text.side_effect = IOError("Disk failure")
        
        _read_chapter_content_details("test_doc", mock_path)
        
        call_args = mock_log_error.call_args
        assert call_args[1]['category'] == ErrorCategory.ERROR


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

    def test_read_chapter_content_details_file_read_error(self, mocker):
        """Test handling file read errors."""
        mock_log_error = mocker.patch('document_mcp.doc_tool_server.log_structured_error')
        mock_path = mocker.Mock()
        mock_path.is_file.return_value = True
        mock_path.name = "corrupted.md"
        mock_path.exists.return_value = True
        mock_path.read_text.side_effect = Exception("File read error")
        
        result = _read_chapter_content_details("test_doc", mock_path)

        assert result is None
        mock_log_error.assert_called_once()
        
        # Verify the structured error call
        call_args = mock_log_error.call_args
        assert call_args[1]['category'] == ErrorCategory.ERROR
        assert call_args[1]['message'] == "Failed to read chapter file: corrupted.md"
        assert call_args[1]['operation'] == "read_chapter_content"

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

    def test_get_chapter_metadata_file_read_error(self, mocker):
        """Test handling file read errors in metadata extraction."""
        mock_log_error = mocker.patch('document_mcp.doc_tool_server.log_structured_error')
        mock_path = mocker.Mock()
        mock_path.is_file.return_value = True
        mock_path.name = "error.md"
        mock_path.exists.return_value = True
        mock_path.read_text.side_effect = Exception("Permission denied")
        
        result = _get_chapter_metadata("test_doc", mock_path)

        assert result is None
        mock_log_error.assert_called_once()
        
        # Verify the structured error call
        call_args = mock_log_error.call_args
        assert call_args[1]['category'] == ErrorCategory.ERROR
        assert call_args[1]['message'] == "Failed to get metadata for chapter: error.md"
        assert call_args[1]['operation'] == "get_chapter_metadata"


class TestInputValidationHelpers:
    """Test suite for input validation helper functions."""

    @pytest.mark.parametrize("name, expected_valid, expected_error_msg", [
        ("doc_name", True, ""),
        ("doc-name-123", True, ""),
        ("doc_with_underscores", True, ""),
        (None, False, "Document name cannot be empty"),
        ("", False, "Document name cannot be empty"),
        ("   ", False, "Document name cannot be empty"),
        ("a" * 101, False, "Document name too long (max 100 characters)"),
        ("invalid/name", False, "Document name cannot contain path separators"),
        ("invalid\\name", False, "Document name cannot contain path separators"),
        (".invalid", False, "Document name cannot start with a dot"),
    ])
    def test_validate_document_name(self, name, expected_valid, expected_error_msg):
        from document_mcp.doc_tool_server import _validate_document_name
        is_valid, error = _validate_document_name(name)
        assert is_valid is expected_valid
        if not expected_valid:
            assert expected_error_msg == error

    @pytest.mark.parametrize("name, expected_valid, expected_error_msg", [
        ("chapter1.md", True, ""),
        ("01-intro.md", True, ""),
        ("File_With_Underscores.MD", True, ""),
        (None, False, "Chapter name cannot be empty"),
        ("", False, "Chapter name cannot be empty"),
        ("a" * 98 + ".md", False, "Chapter name too long (max 100 characters)"),
        ("invalid/chapter.md", False, "Chapter name cannot contain path separators"),
        ("no_extension", False, "Chapter name must end with .md"),
        ("wrong.txt", False, "Chapter name must end with .md"),
        ("_manifest.json", False, "Chapter name cannot be reserved name '_manifest.json'"),
    ])
    def test_validate_chapter_name(self, name, expected_valid, expected_error_msg):
        from document_mcp.doc_tool_server import _validate_chapter_name
        is_valid, error = _validate_chapter_name(name)
        assert is_valid is expected_valid
        if not expected_valid:
            assert expected_error_msg == error

    @pytest.mark.parametrize("content, expected_valid, expected_error_msg", [
        ("Some valid content.", True, ""),
        ("", True, ""),
        ("a" * 1_000_000, True, ""),
        (None, False, "Content cannot be None"),
        (12345, False, "Content must be a string"),
        ("a" * 1_000_001, False, "Content too long (max 1000000 characters)"),
    ])
    def test_validate_content(self, content, expected_valid, expected_error_msg):
        from document_mcp.doc_tool_server import _validate_content
        is_valid, error = _validate_content(content)
        assert is_valid is expected_valid
        if not expected_valid:
            assert expected_error_msg == error

    @pytest.mark.parametrize("index, expected_valid, expected_error_msg", [
        (0, True, ""),
        (100, True, ""),
        (-1, False, "Paragraph index cannot be negative"),
        (None, False, "Paragraph index must be an integer"),
        ("5", False, "Paragraph index must be an integer"),
    ])
    def test_validate_paragraph_index(self, index, expected_valid, expected_error_msg):
        from document_mcp.doc_tool_server import _validate_paragraph_index
        is_valid, error = _validate_paragraph_index(index)
        assert is_valid is expected_valid
        if not expected_valid:
            assert expected_error_msg == error

    @pytest.mark.parametrize("query, expected_valid, expected_error_msg", [
        ("hello", True, ""),
        ("search term", True, ""),
        (None, False, "Search query cannot be None"),
        ("", False, "Search query cannot be empty or whitespace only"),
        ("   ", False, "Search query cannot be empty or whitespace only"),
    ])
    def test_validate_search_query(self, query, expected_valid, expected_error_msg):
        from document_mcp.doc_tool_server import _validate_search_query
        is_valid, error = _validate_search_query(query)
        assert is_valid is expected_valid
        if not expected_valid:
            assert expected_error_msg == error


class TestMCPToolInputValidation:
    """Test suite for MCP tool function input validation and error handling."""

    @pytest.mark.parametrize("doc_name, expected_error_msg", [
        ("", "Document name cannot be empty"),
        ("invalid/name", "Document name cannot contain path separators"),
    ])
    def test_create_document_validation(self, doc_name, expected_error_msg):
        """Tests create_document with various invalid names."""
        from document_mcp.doc_tool_server import create_document
        result = create_document(doc_name)
        assert result.success is False
        assert expected_error_msg in result.message

    @pytest.mark.parametrize("doc_name, chap_name, content, expected_error_msg", [
        ("", "chapter1.md", "content", "Document name cannot be empty"),
        ("test_doc", "", "content", "Chapter name cannot be empty"),
        ("test_doc", "invalid.txt", "content", "Chapter name must end with .md"),
        ("test_doc", "chapter/invalid.md", "content", "Chapter name cannot contain path separators"),
    ])
    def test_create_chapter_validation(self, doc_name, chap_name, content, expected_error_msg):
        """Tests create_chapter with various invalid inputs."""
        from document_mcp.doc_tool_server import create_chapter
        result = create_chapter(doc_name, chap_name, content)
        assert result.success is False
        assert expected_error_msg in result.message

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
        from document_mcp.doc_tool_server import replace_paragraph

        # Test with very large index (should fail with out of bounds)
        result = replace_paragraph(
            "test_doc", "chapter1.md", 9999, "content"
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

    def test_read_document_summary_document_not_found(self, mock_path_operations, mocker):
        """Test reading summary when document directory doesn't exist."""
        mock_log_error = mocker.patch('document_mcp.doc_tool_server.log_structured_error')
        mock_doc_path = mocker.MagicMock(spec=Path)
        mock_doc_path.is_dir.return_value = False
        mock_doc_path.__str__.return_value = "/mocked/path/to/nonexistent_doc"
        
        mock_path_operations.mock_document_path("nonexistent_doc", mock_doc_path)

        summary_content = read_document_summary("nonexistent_doc")
        
        assert summary_content is None
        mock_log_error.assert_called()
        
        # Verify the structured error call
        call_args = mock_log_error.call_args
        assert call_args[1]['category'] == ErrorCategory.INFO
        assert call_args[1]['message'] == "Document not found"
        assert call_args[1]['operation'] == "read_document_summary"

    def test_read_document_summary_summary_file_not_found(self, mock_path_operations, mocker):
        """Test reading summary when _SUMMARY.md file doesn't exist."""
        mock_log_error = mocker.patch('document_mcp.doc_tool_server.log_structured_error')
        mock_doc_path = mocker.MagicMock(spec=Path)
        mock_doc_path.is_dir.return_value = True
        
        mock_summary_file = mocker.MagicMock(spec=Path)
        mock_summary_file.is_file.return_value = False  # Summary file does not exist
        
        mock_doc_path.__truediv__.return_value = mock_summary_file
        mock_path_operations.mock_document_path("doc_no_summary", mock_doc_path)

        summary_content = read_document_summary("doc_no_summary")
        
        assert summary_content is None
        mock_log_error.assert_called()
        
        # Verify the structured error call
        call_args = mock_log_error.call_args
        assert call_args[1]['category'] == ErrorCategory.INFO
        assert call_args[1]['message'] == "Summary file not found in document"
        assert call_args[1]['operation'] == "read_document_summary"

    def test_read_document_summary_read_error(self, mock_path_operations, mocker):
        """Test handling of read errors when accessing summary file."""
        mock_log_error = mocker.patch('document_mcp.doc_tool_server.log_structured_error')
        mock_doc_path = mocker.MagicMock(spec=Path)
        mock_doc_path.is_dir.return_value = True
        
        mock_summary_file = mocker.MagicMock(spec=Path)
        mock_summary_file.is_file.return_value = True
        mock_summary_file.read_text.side_effect = Exception("Read permission denied")
        
        mock_doc_path.__truediv__.return_value = mock_summary_file
        mock_path_operations.mock_document_path("doc_read_error", mock_doc_path)

        summary_content = read_document_summary("doc_read_error")
        
        assert summary_content is None
        mock_log_error.assert_called()
        
        # Verify the structured error call
        call_args = mock_log_error.call_args
        assert call_args[1]['category'] == ErrorCategory.ERROR
        assert call_args[1]['message'] == "Failed to read summary file"
        assert call_args[1]['operation'] == "read_document_summary"

    def test_read_document_summary_invalid_doc_name(self, mock_validation_operations, mocker):
        """Test read_document_summary with an invalid document name."""
        mock_log_error = mocker.patch('document_mcp.doc_tool_server.log_structured_error')
        mock_validation_operations.mock_validate_document_name(False, "Invalid name")
        
        summary_content = read_document_summary("invalid/docname")
        
        assert summary_content is None
        mock_log_error.assert_called()
        
        # Verify the structured error call
        call_args = mock_log_error.call_args
        assert call_args[1]['category'] == ErrorCategory.WARNING
        assert call_args[1]['message'] == "Invalid document name provided"
        assert call_args[1]['operation'] == "read_document_summary"


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


class TestAdvancedDocumentTools:
    """Test suite for advanced document-level tools."""

    def test_get_chapter_statistics_success(self, document_factory):
        """Test get_chapter_statistics with a valid chapter."""
        doc_spec = {
            "name": "stat_doc",
            "chapters": [("01-intro.md", "# Intro\n\nPara one.\n\nPara two has 3 words.")]
        }
        doc_name = document_factory(chapters=[("01-intro.md", "# Intro\n\nPara one.\n\nPara two has 3 words.")])

        stats = get_chapter_statistics(doc_name, "01-intro.md")
        assert stats is not None
        assert isinstance(stats, StatisticsReport)
        assert stats.word_count == 9  # '# Intro', 'Para', 'one.', 'Para', 'two', 'has', '3', 'words.'
        assert stats.paragraph_count == 3  # Title, Para one, Para two...

    def test_get_chapter_statistics_not_found(self):
        """Test get_chapter_statistics with a non-existent chapter."""
        stats = get_chapter_statistics("non_existent_doc", "non_existent_chapter.md")
        assert stats is None

    def test_get_document_statistics_success(self, document_factory):
        """Test get_document_statistics for a valid document."""
        doc_name = document_factory(chapters=[
            ("01.md", "Chapter one has four words."),
            ("02.md", "Chapter two has another four words.")
        ])
        
        stats = get_document_statistics(doc_name)
        assert stats is not None
        assert isinstance(stats, StatisticsReport)
        assert stats.word_count == 11
        assert stats.paragraph_count == 2
        assert stats.chapter_count == 2

    def test_get_document_statistics_not_found(self):
        """Test get_document_statistics for a non-existent document."""
        stats = get_document_statistics("non_existent_doc")
        assert stats is None

    def test_read_full_document_success(self, document_factory):
        """Test read_full_document for a valid document."""
        doc_name = document_factory(chapters=[
            ("01-first.md", "Content one."),
            ("02-second.md", "Content two.")
        ])

        full_doc = read_full_document(doc_name)
        assert full_doc is not None
        assert isinstance(full_doc, FullDocumentContent)
        assert full_doc.document_name == doc_name
        assert len(full_doc.chapters) == 2
        assert full_doc.chapters[0].chapter_name == "01-first.md"
        assert full_doc.chapters[0].content == "Content one."
        assert full_doc.total_word_count == 4
        assert full_doc.total_paragraph_count == 2

    def test_read_full_document_not_found(self):
        """Test read_full_document for a non-existent document."""
        doc = read_full_document("non_existent_doc")
        assert doc is None

    def test_find_text_in_chapter_found(self, searchable_document):
        """Test find_text_in_chapter when text is found."""
        doc_name, search_terms = searchable_document
        chapter_name = "01-intro.md"
        query = search_terms[0] # Should be 'apple'

        results = find_text_in_chapter(doc_name, chapter_name, query)
        assert results is not None
        assert len(results) > 0
        assert isinstance(results[0], ParagraphDetail)
        assert query.lower() in results[0].content.lower()

    def test_find_text_in_chapter_not_found(self, searchable_document):
        """Test find_text_in_chapter when text is not found."""
        doc_name, _ = searchable_document
        results = find_text_in_chapter(doc_name, "01-intro.md", "nonexistentqueryword")
        assert results == []

    def test_find_text_in_chapter_case_sensitive(self, document_factory):
        """Test find_text_in_chapter with case sensitivity."""
        doc_name = document_factory(chapters=[("case_test.md", "Here is a Test sentence.\n\nAnother test sentence.")])
        
        # Case-sensitive search
        results_sensitive = find_text_in_chapter(doc_name, "case_test.md", "Test", case_sensitive=True)
        assert len(results_sensitive) == 1
        assert "Test" in results_sensitive[0].content

        # Case-insensitive search
        results_insensitive = find_text_in_chapter(doc_name, "case_test.md", "test", case_sensitive=False)
        assert len(results_insensitive) == 2

    def test_find_text_in_document_found(self, searchable_document):
        """Test find_text_in_document when text is found across chapters."""
        doc_name, search_terms = searchable_document
        query = search_terms[0] # 'apple'

        results = find_text_in_document(doc_name, query)
        assert results is not None
        assert len(results) > 1  # Should be in multiple chapters
        assert query.lower() in results[0].content.lower()
        assert results[0].chapter_name != results[1].chapter_name

    def test_find_text_in_document_not_found(self, searchable_document):
        """Test find_text_in_document when text is not found."""
        doc_name, _ = searchable_document
        results = find_text_in_document(doc_name, "nonexistentqueryword")
        assert results == []
