"""Unit tests for file operations utilities."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

from document_mcp.utils import file_operations


class TestDocsRootPath:
    """Tests for _DocsRootPath dynamic path wrapper."""

    def test_truediv_operator(self):
        """Test path concatenation with / operator."""
        result = file_operations.DOCS_ROOT_PATH / "test_doc"
        assert isinstance(result, Path)
        assert result.name == "test_doc"

    def test_str_conversion(self):
        """Test string conversion."""
        result = str(file_operations.DOCS_ROOT_PATH)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_repr_conversion(self):
        """Test repr conversion."""
        result = repr(file_operations.DOCS_ROOT_PATH)
        assert isinstance(result, str)

    def test_resolve_method(self):
        """Test resolve method."""
        result = file_operations.DOCS_ROOT_PATH.resolve()
        assert isinstance(result, Path)
        assert result.is_absolute()

    def test_getattr_delegation(self):
        """Test that attribute access is delegated to Path."""
        # Access a Path attribute through the wrapper
        name = file_operations.DOCS_ROOT_PATH.name
        assert isinstance(name, str)


class TestGetDocsRootPath:
    """Tests for get_docs_root_path function."""

    def test_returns_path(self):
        """Test that get_docs_root_path returns a Path."""
        result = file_operations.get_docs_root_path()
        assert isinstance(result, Path)

    def test_respects_environment_variable(self):
        """Test that environment variable is respected."""
        # Mock the get_settings to return a custom path
        from unittest.mock import MagicMock

        mock_settings = MagicMock()
        mock_settings.document_root_path = Path("/custom/path")

        with patch("document_mcp.utils.file_operations.get_settings", return_value=mock_settings):
            result = file_operations.get_docs_root_path()
            # Use as_posix() for cross-platform path comparison
            assert result.as_posix() == "/custom/path"


class TestGetDocumentPath:
    """Tests for get_document_path function."""

    def test_returns_path_for_document(self):
        """Test that get_document_path returns correct path."""
        result = file_operations.get_document_path("my_document")
        assert isinstance(result, Path)
        assert result.name == "my_document"

    def test_path_is_under_docs_root(self):
        """Test that document path is under docs root."""
        result = file_operations.get_document_path("test_doc")
        docs_root = file_operations.get_docs_root_path()
        assert str(result).startswith(str(docs_root))


class TestGetChapterPath:
    """Tests for get_chapter_path function."""

    def test_returns_path_for_chapter(self):
        """Test that get_chapter_path returns correct path."""
        result = file_operations.get_chapter_path("my_doc", "01-intro.md")
        assert isinstance(result, Path)
        assert result.name == "01-intro.md"

    def test_path_includes_document(self):
        """Test that chapter path includes document name."""
        result = file_operations.get_chapter_path("my_doc", "01-intro.md")
        assert "my_doc" in str(result)


class TestGetOperationPath:
    """Tests for get_operation_path function."""

    def test_returns_chapter_path_when_chapter_provided(self):
        """Test that chapter path is returned when chapter name is provided."""
        result = file_operations.get_operation_path("doc", "chapter.md")
        assert result.name == "chapter.md"

    def test_returns_document_path_when_chapter_is_none(self):
        """Test that document path is returned when chapter is None."""
        result = file_operations.get_operation_path("doc", None)
        assert result.name == "doc"

    def test_returns_document_path_when_chapter_is_empty(self):
        """Test that document path is returned when chapter is empty string."""
        # Empty string is falsy, so should return document path
        result = file_operations.get_operation_path("doc", "")
        assert result.name == "doc"


class TestGetSnapshotsPath:
    """Tests for get_snapshots_path function."""

    def test_returns_snapshots_directory_path(self):
        """Test that snapshots path is correct."""
        result = file_operations.get_snapshots_path("my_doc")
        assert isinstance(result, Path)
        assert result.name == ".snapshots"
        assert "my_doc" in str(result)


class TestGetModificationHistoryPath:
    """Tests for get_modification_history_path function."""

    def test_returns_history_file_path(self):
        """Test that modification history path is correct."""
        result = file_operations.get_modification_history_path("my_doc")
        assert isinstance(result, Path)
        assert result.name == ".mod_history.json"
        assert "my_doc" in str(result)


class TestIsValidChapterFilename:
    """Tests for is_valid_chapter_filename function."""

    def test_valid_chapter_filename(self):
        """Test valid chapter filenames."""
        assert file_operations.is_valid_chapter_filename("01-intro.md") is True
        assert file_operations.is_valid_chapter_filename("chapter.md") is True
        assert file_operations.is_valid_chapter_filename("CHAPTER.MD") is True
        assert file_operations.is_valid_chapter_filename("my-chapter.Md") is True

    def test_invalid_non_md_files(self):
        """Test that non-.md files are invalid."""
        assert file_operations.is_valid_chapter_filename("file.txt") is False
        assert file_operations.is_valid_chapter_filename("document.pdf") is False
        assert file_operations.is_valid_chapter_filename("script.py") is False
        assert file_operations.is_valid_chapter_filename("no_extension") is False

    def test_reserved_filenames_are_invalid(self):
        """Test that reserved filenames are invalid."""
        assert file_operations.is_valid_chapter_filename("_manifest.json") is False
        assert file_operations.is_valid_chapter_filename("_summary.md") is False
        assert file_operations.is_valid_chapter_filename("_index.md") is False
        assert file_operations.is_valid_chapter_filename("README.md") is False
        assert file_operations.is_valid_chapter_filename("readme.md") is False
        assert file_operations.is_valid_chapter_filename(".gitignore") is False
        assert file_operations.is_valid_chapter_filename(".mod_history.json") is False

    def test_case_insensitive_reserved_names(self):
        """Test that reserved name check is case insensitive."""
        assert file_operations.is_valid_chapter_filename("README.MD") is False
        assert file_operations.is_valid_chapter_filename("Readme.md") is False
        assert file_operations.is_valid_chapter_filename("_Summary.md") is False


class TestEnsureDirectoryExists:
    """Tests for ensure_directory_exists function."""

    def test_creates_directory(self, tmp_path):
        """Test that directory is created."""
        new_dir = tmp_path / "new_directory"
        assert not new_dir.exists()

        file_operations.ensure_directory_exists(new_dir)

        assert new_dir.exists()
        assert new_dir.is_dir()

    def test_creates_nested_directories(self, tmp_path):
        """Test that nested directories are created."""
        nested_dir = tmp_path / "level1" / "level2" / "level3"
        assert not nested_dir.exists()

        file_operations.ensure_directory_exists(nested_dir)

        assert nested_dir.exists()
        assert nested_dir.is_dir()

    def test_does_not_fail_if_exists(self, tmp_path):
        """Test that existing directory doesn't cause error."""
        existing_dir = tmp_path / "existing"
        existing_dir.mkdir()

        # Should not raise
        file_operations.ensure_directory_exists(existing_dir)

        assert existing_dir.exists()


class TestGetCurrentUser:
    """Tests for get_current_user function."""

    def test_returns_user_from_environment(self):
        """Test that USER environment variable is used."""
        with patch.dict(os.environ, {"USER": "test_user"}):
            result = file_operations.get_current_user()
            assert result == "test_user"

    def test_returns_default_when_no_user_env(self):
        """Test fallback when USER is not set."""
        env_copy = os.environ.copy()
        env_copy.pop("USER", None)
        with patch.dict(os.environ, env_copy, clear=True):
            result = file_operations.get_current_user()
            assert result == "system_user"


class TestSplitContentIntoParagraphs:
    """Tests for split_content_into_paragraphs function."""

    def test_splits_on_double_newlines(self):
        """Test that content is split on double newlines."""
        content = "Paragraph 1\n\nParagraph 2\n\nParagraph 3"
        result = file_operations.split_content_into_paragraphs(content)
        assert result == ["Paragraph 1", "Paragraph 2", "Paragraph 3"]

    def test_filters_empty_paragraphs(self):
        """Test that empty paragraphs are filtered out."""
        content = "Para 1\n\n\n\nPara 2\n\n\n\n\n\nPara 3"
        result = file_operations.split_content_into_paragraphs(content)
        assert result == ["Para 1", "Para 2", "Para 3"]

    def test_strips_whitespace(self):
        """Test that paragraphs are stripped of whitespace."""
        content = "  Para 1  \n\n  Para 2  "
        result = file_operations.split_content_into_paragraphs(content)
        assert result == ["Para 1", "Para 2"]

    def test_empty_content_returns_empty_list(self):
        """Test that empty content returns empty list."""
        assert file_operations.split_content_into_paragraphs("") == []
        assert file_operations.split_content_into_paragraphs("   ") == []

    def test_none_like_behavior(self):
        """Test handling of falsy content."""
        assert file_operations.split_content_into_paragraphs("") == []

    def test_single_paragraph(self):
        """Test content with single paragraph."""
        content = "Just one paragraph"
        result = file_operations.split_content_into_paragraphs(content)
        assert result == ["Just one paragraph"]


class TestJoinParagraphs:
    """Tests for join_paragraphs function."""

    def test_joins_with_double_newlines(self):
        """Test that paragraphs are joined with double newlines."""
        paragraphs = ["Para 1", "Para 2", "Para 3"]
        result = file_operations.join_paragraphs(paragraphs)
        assert result == "Para 1\n\nPara 2\n\nPara 3"

    def test_empty_list_returns_empty_string(self):
        """Test that empty list returns empty string."""
        result = file_operations.join_paragraphs([])
        assert result == ""

    def test_single_paragraph(self):
        """Test single paragraph returns itself."""
        result = file_operations.join_paragraphs(["Single"])
        assert result == "Single"


class TestGenerateContentDiff:
    """Tests for generate_content_diff function."""

    def test_no_changes(self):
        """Test diff when content is identical."""
        content = "Line 1\nLine 2"
        result = file_operations.generate_content_diff(content, content)

        assert result["has_changes"] is False
        assert result["old_line_count"] == 2
        assert result["new_line_count"] == 2
        assert result["lines_added"] == 0
        assert result["lines_removed"] == 0
        assert result["content_size_change"] == 0

    def test_lines_added(self):
        """Test diff when lines are added."""
        old = "Line 1"
        new = "Line 1\nLine 2\nLine 3"
        result = file_operations.generate_content_diff(old, new)

        assert result["has_changes"] is True
        assert result["old_line_count"] == 1
        assert result["new_line_count"] == 3
        assert result["lines_added"] == 2
        assert result["lines_removed"] == 0
        assert result["content_size_change"] > 0

    def test_lines_removed(self):
        """Test diff when lines are removed."""
        old = "Line 1\nLine 2\nLine 3"
        new = "Line 1"
        result = file_operations.generate_content_diff(old, new)

        assert result["has_changes"] is True
        assert result["old_line_count"] == 3
        assert result["new_line_count"] == 1
        assert result["lines_added"] == 0
        assert result["lines_removed"] == 2
        assert result["content_size_change"] < 0

    def test_empty_old_content(self):
        """Test diff with empty old content."""
        result = file_operations.generate_content_diff("", "New content")

        assert result["has_changes"] is True
        assert result["old_line_count"] == 0
        assert result["new_line_count"] == 1

    def test_empty_new_content(self):
        """Test diff with empty new content."""
        result = file_operations.generate_content_diff("Old content", "")

        assert result["has_changes"] is True
        assert result["old_line_count"] == 1
        assert result["new_line_count"] == 0

    def test_content_modified_same_lines(self):
        """Test diff when content modified but line count same."""
        old = "Hello World"
        new = "Hello Claude"
        result = file_operations.generate_content_diff(old, new)

        assert result["has_changes"] is True
        assert result["old_line_count"] == 1
        assert result["new_line_count"] == 1
        assert result["lines_added"] == 0
        assert result["lines_removed"] == 0
