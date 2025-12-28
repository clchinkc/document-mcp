"""Unit tests for document MCP tool server helper functions."""

import datetime

import pytest

from document_mcp.helpers import DOCUMENT_SUMMARY_FILE
from document_mcp.helpers import _count_words
from document_mcp.helpers import _get_modification_history_path
from document_mcp.helpers import _get_snapshots_path
from document_mcp.helpers import _get_summaries_path
from document_mcp.helpers import _get_summary_file_path
from document_mcp.helpers import _is_valid_chapter_filename
from document_mcp.helpers import _split_into_paragraphs

# Import constants and models
from document_mcp.utils.validation import CHAPTER_MANIFEST_FILE

# Import helper functions
from document_mcp.utils.validation import check_file_freshness as _check_file_freshness
from document_mcp.utils.validation import validate_content


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
    def testvalidate_document_name(self, name, expected_valid, expected_error_msg):
        from document_mcp.utils.validation import validate_document_name

        is_valid, error = validate_document_name(name)
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
    def testvalidate_chapter_name(self, name, expected_valid, expected_error_msg):
        from document_mcp.utils.validation import validate_chapter_name

        is_valid, error = validate_chapter_name(name)
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
    def test_validate_content_general_cases(self, content, expected_valid, expected_error_msg):
        """Test content validation for general cases (None, type, and short strings)."""
        is_valid, error = validate_content(content)
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
    def testvalidate_paragraph_index(self, index, expected_valid, expected_error_msg):
        from document_mcp.utils.validation import validate_paragraph_index

        is_valid, error = validate_paragraph_index(index)
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
    def testvalidate_search_query(self, query, expected_valid, expected_error_msg):
        from document_mcp.utils.validation import validate_search_query

        is_valid, error = validate_search_query(query)
        assert is_valid is expected_valid
        if not expected_valid:
            assert expected_error_msg == error


class TestSafetyHelperFunctions:
    """Test suite for safety feature helper functions."""

    def test_get_snapshots_path_with_default_root(self):
        """Test snapshots path generation with default DOCS_ROOT_PATH."""
        import os

        # Test without DOCUMENT_ROOT_DIR environment variable (production behavior)
        old_doc_root = os.environ.get("DOCUMENT_ROOT_DIR")
        if "DOCUMENT_ROOT_DIR" in os.environ:
            del os.environ["DOCUMENT_ROOT_DIR"]

        try:
            # Reset settings singleton to pick up environment change
            from document_mcp.config.settings import reset_settings

            reset_settings()

            from document_mcp.utils.file_operations import DOCS_ROOT_PATH

            result = _get_snapshots_path("test_doc")
            expected = (DOCS_ROOT_PATH / "test_doc" / ".snapshots").resolve()
            assert result == expected
        finally:
            # Restore environment
            if old_doc_root:
                os.environ["DOCUMENT_ROOT_DIR"] = old_doc_root
            # Reset settings again to pick up the restored environment
            reset_settings()

    def test_get_snapshots_path_with_custom_root(self):
        """Test snapshots path generation with custom DOCUMENT_ROOT_DIR."""
        import os
        import tempfile
        from pathlib import Path

        # Test with DOCUMENT_ROOT_DIR environment variable (test isolation behavior)
        old_doc_root = os.environ.get("DOCUMENT_ROOT_DIR")

        with tempfile.TemporaryDirectory() as temp_dir:
            os.environ["DOCUMENT_ROOT_DIR"] = temp_dir

            try:
                result = _get_snapshots_path("test_doc")
                expected = (Path(temp_dir) / "test_doc" / ".snapshots").resolve()
                assert result == expected
            finally:
                # Restore environment
                if old_doc_root:
                    os.environ["DOCUMENT_ROOT_DIR"] = old_doc_root
                else:
                    os.environ.pop("DOCUMENT_ROOT_DIR", None)

    def test_get_modification_history_path_with_default_root(self):
        """Test modification history path generation with default DOCS_ROOT_PATH."""
        import os

        # Test without DOCUMENT_ROOT_DIR environment variable (production behavior)
        old_doc_root = os.environ.get("DOCUMENT_ROOT_DIR")
        if "DOCUMENT_ROOT_DIR" in os.environ:
            del os.environ["DOCUMENT_ROOT_DIR"]

        try:
            # Reset settings singleton to pick up environment change
            from document_mcp.config.settings import reset_settings

            reset_settings()

            from document_mcp.utils.file_operations import DOCS_ROOT_PATH

            result = _get_modification_history_path("test_doc")
            expected = (DOCS_ROOT_PATH / "test_doc" / ".mod_history.json").resolve()
            assert result == expected
        finally:
            # Restore environment
            if old_doc_root:
                os.environ["DOCUMENT_ROOT_DIR"] = old_doc_root
            # Reset settings again to pick up the restored environment
            reset_settings()

    def test_get_modification_history_path_with_custom_root(self):
        """Test modification history path generation with custom DOCUMENT_ROOT_DIR."""
        import os
        import tempfile
        from pathlib import Path

        # Test with DOCUMENT_ROOT_DIR environment variable (test isolation behavior)
        old_doc_root = os.environ.get("DOCUMENT_ROOT_DIR")

        with tempfile.TemporaryDirectory() as temp_dir:
            os.environ["DOCUMENT_ROOT_DIR"] = temp_dir

            try:
                result = _get_modification_history_path("test_doc")
                expected = (Path(temp_dir) / "test_doc" / ".mod_history.json").resolve()
                assert result == expected
            finally:
                # Restore environment
                if old_doc_root:
                    os.environ["DOCUMENT_ROOT_DIR"] = old_doc_root
                else:
                    os.environ.pop("DOCUMENT_ROOT_DIR", None)

    def test_check_file_freshness_file_not_exists(self):
        from unittest.mock import Mock

        mock_path = Mock()
        mock_path.exists.return_value = False

        result = _check_file_freshness(mock_path)

        assert result.__class__.__name__ == "ContentFreshnessStatus"
        assert not result.is_fresh
        assert result.safety_status == "conflict"
        assert "Verify file was not accidentally deleted" in result.recommendations

    def test_check_file_freshness_file_exists_fresh(self):
        from unittest.mock import Mock

        current_time = datetime.datetime.now()
        mock_path = Mock()
        mock_path.exists.return_value = True
        mock_path.stat.return_value.st_mtime = current_time.timestamp()

        result = _check_file_freshness(mock_path, current_time)

        assert result.__class__.__name__ == "ContentFreshnessStatus"
        assert result.is_fresh
        assert result.safety_status == "safe"

    def test_check_file_freshness_file_exists_stale(self):
        from unittest.mock import Mock

        old_time = datetime.datetime.now() - datetime.timedelta(hours=1)
        new_time = datetime.datetime.now()
        mock_path = Mock()
        mock_path.exists.return_value = True
        mock_path.stat.return_value.st_mtime = new_time.timestamp()

        result = _check_file_freshness(mock_path, old_time)

        assert result.__class__.__name__ == "ContentFreshnessStatus"
        assert not result.is_fresh
        assert result.safety_status == "warning"
        assert "Content was modified" in result.message


class TestSummaryHelperFunctions:
    """Test suite for summary helper functions."""

    def test_get_summaries_path_with_default_root(self):
        """Test summaries path generation with default DOCS_ROOT_PATH."""
        import os

        # Test without DOCUMENT_ROOT_DIR environment variable (production behavior)
        old_doc_root = os.environ.get("DOCUMENT_ROOT_DIR")
        if "DOCUMENT_ROOT_DIR" in os.environ:
            del os.environ["DOCUMENT_ROOT_DIR"]

        try:
            # Reset settings singleton to pick up environment change
            from document_mcp.config.settings import reset_settings

            reset_settings()

            from document_mcp.utils.file_operations import DOCS_ROOT_PATH

            result = _get_summaries_path("test_doc")
            expected = (DOCS_ROOT_PATH / "test_doc" / "summaries").resolve()
            assert result == expected
        finally:
            # Restore environment
            if old_doc_root:
                os.environ["DOCUMENT_ROOT_DIR"] = old_doc_root
            # Reset settings again to pick up the restored environment
            reset_settings()

    def test_get_summaries_path_with_custom_root(self):
        """Test summaries path generation with custom DOCUMENT_ROOT_DIR."""
        import os
        import tempfile
        from pathlib import Path

        # Test with DOCUMENT_ROOT_DIR environment variable (test isolation behavior)
        old_doc_root = os.environ.get("DOCUMENT_ROOT_DIR")

        with tempfile.TemporaryDirectory() as temp_dir:
            os.environ["DOCUMENT_ROOT_DIR"] = temp_dir

            try:
                result = _get_summaries_path("test_doc")
                expected = (Path(temp_dir) / "test_doc" / "summaries").resolve()
                assert result == expected
            finally:
                # Restore environment
                if old_doc_root:
                    os.environ["DOCUMENT_ROOT_DIR"] = old_doc_root
                else:
                    if "DOCUMENT_ROOT_DIR" in os.environ:
                        del os.environ["DOCUMENT_ROOT_DIR"]

    def test_get_summary_file_path_document_scope(self):
        """Test summary file path generation for document scope."""
        result = _get_summary_file_path("test_doc", "document", None)
        assert result.name == "document.md"
        assert result.parent.name == "summaries"

    def test_get_summary_file_path_chapter_scope(self):
        """Test summary file path generation for chapter scope."""
        result = _get_summary_file_path("test_doc", "chapter", "01-intro.md")
        assert result.name == "chapter-01-intro.md"
        assert result.parent.name == "summaries"

    def test_get_summary_file_path_section_scope(self):
        """Test summary file path generation for section scope."""
        result = _get_summary_file_path("test_doc", "section", "introduction")
        assert result.name == "section-introduction.md"
        assert result.parent.name == "summaries"

    def test_get_summary_file_path_chapter_scope_missing_target(self):
        """Test that chapter scope requires target_name."""
        import pytest

        with pytest.raises(ValueError, match="target_name is required for chapter scope"):
            _get_summary_file_path("test_doc", "chapter", None)

    def test_get_summary_file_path_section_scope_missing_target(self):
        """Test that section scope requires target_name."""
        import pytest

        with pytest.raises(ValueError, match="target_name is required for section scope"):
            _get_summary_file_path("test_doc", "section", None)

    def test_get_summary_file_path_invalid_scope(self):
        """Test that invalid scope raises ValueError."""
        import pytest

        with pytest.raises(ValueError, match="Invalid scope.*Must be 'document', 'chapter', or 'section'"):
            _get_summary_file_path("test_doc", "invalid", None)


class TestUnifiedContentTools:
    """Test suite for unified content access tools."""

    def test_read_content_document_scope_validation(self):
        """Test read_content with document scope parameter validation."""
        from document_mcp.mcp_client import read_content

        # Test invalid document name
        result = read_content("", scope="document")
        assert result is None

        # Test invalid scope
        result = read_content("test_doc", scope="invalid")
        assert result is None

        # Test valid document scope for non-existent document
        # Should return None since document doesn't exist
        result = read_content("definitely_nonexistent_doc_12345", scope="document")
        assert result is None

    def test_read_content_chapter_scope_validation(self):
        """Test read_content with chapter scope parameter validation."""
        from document_mcp.mcp_client import read_content

        # Test chapter scope without chapter_name
        result = read_content("test_doc", scope="chapter")
        assert result is None

        # Test chapter scope with invalid chapter_name
        result = read_content("test_doc", scope="chapter", chapter_name="")
        assert result is None

    def test_read_content_paragraph_scope_validation(self):
        """Test read_content with paragraph scope parameter validation."""
        from document_mcp.mcp_client import read_content

        # Test paragraph scope without chapter_name
        result = read_content("test_doc", scope="paragraph")
        assert result is None

        # Test paragraph scope without paragraph_index
        result = read_content("test_doc", scope="paragraph", chapter_name="01-intro.md")
        assert result is None

        # Test paragraph scope with negative paragraph_index
        result = read_content(
            "test_doc",
            scope="paragraph",
            chapter_name="01-intro.md",
            paragraph_index=-1,
        )
        assert result is None

    def test_find_text_document_scope_validation(self):
        """Test find_text with document scope parameter validation."""
        from document_mcp.mcp_client import find_text

        # Test invalid document name
        result = find_text("", "search_term", scope="document")
        assert result is None

        # Test empty search text
        result = find_text("test_doc", "", scope="document")
        assert result is None

        # Test invalid scope
        result = find_text("test_doc", "search_term", scope="invalid")
        assert result is None

    def test_find_text_chapter_scope_validation(self):
        """Test find_text with chapter scope parameter validation."""
        from document_mcp.mcp_client import find_text

        # Test chapter scope without chapter_name
        result = find_text("test_doc", "search_term", scope="chapter")
        assert result is None

        # Test chapter scope with empty chapter_name
        result = find_text("test_doc", "search_term", scope="chapter", chapter_name="")
        assert result is None

    def test_replace_text_document_scope_validation(self):
        """Test replace_text with document scope parameter validation."""
        from document_mcp.mcp_client import replace_text

        # Test invalid document name
        result = replace_text("", "find_text", "replace_text", scope="document")
        assert result is None

        # Test empty find_text
        result = replace_text("test_doc", "", "replace_text", scope="document")
        assert result is None

        # Test invalid scope
        result = replace_text("test_doc", "find_text", "replace_text", scope="invalid")
        assert result is None

    def test_replace_text_chapter_scope_validation(self):
        """Test replace_text with chapter scope parameter validation."""
        from document_mcp.mcp_client import replace_text

        # Test chapter scope without chapter_name
        result = replace_text("test_doc", "find_text", "replace_text", scope="chapter")
        assert result is None

        # Test chapter scope with empty chapter_name
        result = replace_text("test_doc", "find_text", "replace_text", scope="chapter", chapter_name="")
        assert result is None

    def test_get_statistics_document_scope_validation(self):
        """Test get_statistics with document scope parameter validation."""
        from document_mcp.mcp_client import get_statistics

        # Test invalid document name
        result = get_statistics("", scope="document")
        assert result is None

        # Test invalid scope
        result = get_statistics("test_doc", scope="invalid")
        assert result is None

    def test_get_statistics_chapter_scope_validation(self):
        """Test get_statistics with chapter scope parameter validation."""
        from document_mcp.mcp_client import get_statistics

        # Test chapter scope without chapter_name
        result = get_statistics("test_doc", scope="chapter")
        assert result is None

        # Test chapter scope with empty chapter_name
        result = get_statistics("test_doc", scope="chapter", chapter_name="")
        assert result is None

    def test_unified_tools_scope_dispatch(self):
        """Test that unified tools properly dispatch to correct internal functions."""
        from document_mcp.mcp_client import find_text
        from document_mcp.mcp_client import get_statistics
        from document_mcp.mcp_client import read_content
        from document_mcp.mcp_client import replace_text

        # Test that each unified tool properly validates scope parameters
        # This tests the parameter validation and dispatch logic without requiring actual file operations

        # Test all tools with invalid scopes
        invalid_scope = "invalid_scope"

        assert read_content("test_doc", scope=invalid_scope) is None
        assert find_text("test_doc", "search", scope=invalid_scope) is None
        assert replace_text("test_doc", "find", "replace", scope=invalid_scope) is None
        assert get_statistics("test_doc", scope=invalid_scope) is None

    def test_unified_tools_error_handling(self):
        """Test unified tools error handling with various invalid inputs."""
        from document_mcp.mcp_client import find_text
        from document_mcp.mcp_client import get_statistics
        from document_mcp.mcp_client import read_content
        from document_mcp.mcp_client import replace_text

        # Test with None values
        assert read_content(None, scope="document") is None
        assert find_text(None, "search", scope="document") is None
        assert replace_text(None, "find", "replace", scope="document") is None
        assert get_statistics(None, scope="document") is None

        # Test with empty strings
        assert read_content("", scope="document") is None
        assert find_text("", "search", scope="document") is None
        assert replace_text("", "find", "replace", scope="document") is None
        assert get_statistics("", scope="document") is None

    def test_unified_tools_parameter_combinations(self):
        """Test unified tools with various parameter combinations."""
        from document_mcp.mcp_client import find_text
        from document_mcp.mcp_client import read_content

        # Test read_content with all scope variations
        # Document scope (default) for non-existent document
        result = read_content("nonexistent_test_doc_12345")
        assert result is None  # Non-existent document returns None

        # Chapter scope with chapter_name
        result = read_content("test_doc", scope="chapter", chapter_name="01-intro.md")
        assert result is None  # Document doesn't exist, but validation passes

        # Paragraph scope with all required parameters
        result = read_content("test_doc", scope="paragraph", chapter_name="01-intro.md", paragraph_index=0)
        assert result is None  # Document doesn't exist, but validation passes

        # Test find_text with case sensitivity
        result = find_text("test_doc", "search", scope="document", case_sensitive=True)
        assert result is not None  # Returns empty list when document doesn't exist
        assert result == []

        result = find_text(
            "test_doc",
            "search",
            scope="chapter",
            chapter_name="01-intro.md",
            case_sensitive=False,
        )
        assert result is not None  # Returns empty list when chapter doesn't exist
        assert result == []
