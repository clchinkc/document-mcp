import pytest
from document_mcp.doc_tool_server import (
    create_document, delete_document,
    create_chapter, delete_chapter, write_chapter_content,
    append_paragraph_to_chapter,
    replace_text_in_chapter, replace_text_in_document
)

class TestDeleteChapter:
    def test_delete_chapter_success(self, mock_path_operations, mocker):
        doc_name, chapter_name = "doc1", "ch1.md"
        mock_chapter_path = mocker.Mock(); mock_chapter_path.is_file.return_value = True
        mock_path_operations.mock_chapter_path(doc_name, chapter_name, mock_chapter_path)
        mock_chapter_path.unlink.return_value = None
        result = delete_chapter(doc_name, chapter_name)
        assert result.success is True
        assert "deleted successfully" in result.message or result.success is True

    def test_delete_chapter_not_found(self, mock_path_operations, mocker):
        doc_name, chapter_name = "doc1", "ch1.md"
        mock_chapter_path = mocker.Mock(); mock_chapter_path.is_file.return_value = False
        mock_path_operations.mock_chapter_path(doc_name, chapter_name, mock_chapter_path)
        result = delete_chapter(doc_name, chapter_name)
        assert "not found" in result.message.lower()

    def test_delete_chapter_unlink_error(self, mock_path_operations, mocker):
        doc_name, chapter_name = "doc1", "ch1.md"
        mock_chapter_path = mocker.Mock(); mock_chapter_path.is_file.return_value = True
        mock_chapter_path.unlink.side_effect = Exception("unlink failed")
        mock_path_operations.mock_chapter_path(doc_name, chapter_name, mock_chapter_path)
        result = delete_chapter(doc_name, chapter_name)
        assert result.success is False
        assert "Error deleting chapter" in result.message or "unlink failed" in result.message 