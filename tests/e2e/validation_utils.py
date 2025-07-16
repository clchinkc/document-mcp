"""Validation utilities for E2E testing that focus on file system state verification.

This module provides reusable validation helpers that assert on concrete outcomes
rather than natural language text, implementing the "Assert on Reality" testing strategy.
"""

import ast
from pathlib import Path


class DocumentSystemValidator:
    """Validates document system state by checking file system directly."""

    def __init__(self, docs_root: Path):
        """Initialize validator with document root directory.

        Args:
            docs_root: Root directory where documents are stored
        """
        self.docs_root = Path(docs_root)

    def assert_document_exists(self, doc_name: str) -> Path:
        """Assert that a document directory exists and return its path.

        Args:
            doc_name: Name of the document to check

        Returns:
            Path to the document directory

        Raises:
            AssertionError: If document directory doesn't exist
        """
        doc_path = self.docs_root / doc_name
        assert doc_path.exists(), (
            f"Document directory '{doc_name}' does not exist at {doc_path}"
        )
        assert doc_path.is_dir(), (
            f"Document path '{doc_path}' exists but is not a directory"
        )
        return doc_path

    def assert_document_not_exists(self, doc_name: str) -> None:
        """Assert that a document directory does not exist.

        Args:
            doc_name: Name of the document to check

        Raises:
            AssertionError: If document directory exists
        """
        doc_path = self.docs_root / doc_name
        assert not doc_path.exists(), (
            f"Document directory '{doc_name}' should not exist but found at {doc_path}"
        )

    def assert_chapter_exists(self, doc_name: str, chapter_name: str) -> Path:
        """Assert that a chapter file exists and return its path.

        Args:
            doc_name: Name of the document containing the chapter
            chapter_name: Name of the chapter file (e.g., '01-intro.md')

        Returns:
            Path to the chapter file

        Raises:
            AssertionError: If chapter file doesn't exist
        """
        doc_path = self.assert_document_exists(doc_name)
        chapter_path = doc_path / chapter_name
        assert chapter_path.exists(), (
            f"Chapter file '{chapter_name}' does not exist at {chapter_path}"
        )
        assert chapter_path.is_file(), (
            f"Chapter path '{chapter_path}' exists but is not a file"
        )
        return chapter_path

    def assert_chapter_not_exists(self, doc_name: str, chapter_name: str) -> None:
        """Assert that a chapter file does not exist.

        Args:
            doc_name: Name of the document that should not contain the chapter
            chapter_name: Name of the chapter file

        Raises:
            AssertionError: If chapter file exists
        """
        doc_path = self.docs_root / doc_name
        if doc_path.exists():
            chapter_path = doc_path / chapter_name
            assert not chapter_path.exists(), (
                f"Chapter file '{chapter_name}' should not exist but found at {chapter_path}"
            )

    def assert_summary_exists(self, doc_name: str) -> Path:
        """Assert that a summary file (_SUMMARY.md) exists for a document.

        Args:
            doc_name: Name of the document

        Returns:
            Path to the summary file

        Raises:
            AssertionError: If summary file doesn't exist
        """
        doc_path = self.assert_document_exists(doc_name)
        summary_path = doc_path / "_SUMMARY.md"
        assert summary_path.exists(), (
            f"Summary file '_SUMMARY.md' does not exist at {summary_path}"
        )
        assert summary_path.is_file(), (
            f"Summary path '{summary_path}' exists but is not a file"
        )
        return summary_path

    def assert_chapter_content_contains(
        self, doc_name: str, chapter_name: str, expected_content: str
    ) -> None:
        """Assert that a chapter file contains specific content.

        Args:
            doc_name: Name of the document containing the chapter
            chapter_name: Name of the chapter file
            expected_content: Content that should be present in the chapter

        Raises:
            AssertionError: If chapter doesn't contain expected content
        """
        chapter_path = self.assert_chapter_exists(doc_name, chapter_name)
        actual_content = chapter_path.read_text(encoding="utf-8")
        assert expected_content in actual_content, (
            f"Chapter '{chapter_name}' in document '{doc_name}' does not contain expected content.\n"
            f"Expected to find: '{expected_content}'\n"
            f"Actual content: '{actual_content}'"
        )

    def assert_chapter_content_equals(
        self, doc_name: str, chapter_name: str, expected_content: str
    ) -> None:
        """Assert that a chapter file's content exactly matches expected content.

        Args:
            doc_name: Name of the document containing the chapter
            chapter_name: Name of the chapter file
            expected_content: Exact content that should be in the chapter

        Raises:
            AssertionError: If chapter content doesn't match exactly
        """
        chapter_path = self.assert_chapter_exists(doc_name, chapter_name)
        actual_content = chapter_path.read_text(encoding="utf-8").strip()
        expected_content = expected_content.strip()
        assert actual_content == expected_content, (
            f"Chapter '{chapter_name}' in document '{doc_name}' content mismatch.\n"
            f"Expected: '{expected_content}'\n"
            f"Actual: '{actual_content}'"
        )

    def get_document_names(self) -> set[str]:
        """Get names of all documents in the system.

        Returns:
            Set of document names (directory names)
        """
        if not self.docs_root.exists():
            return set()

        return {
            item.name
            for item in self.docs_root.iterdir()
            if item.is_dir() and not item.name.startswith(".")
        }

    def get_chapter_names(self, doc_name: str) -> set[str]:
        """Get names of all chapters in a document.

        Args:
            doc_name: Name of the document

        Returns:
            Set of chapter file names
        """
        doc_path = self.docs_root / doc_name
        if not doc_path.exists() or not doc_path.is_dir():
            return set()

        return {
            item.name
            for item in doc_path.iterdir()
            if item.is_file()
            and item.name.endswith(".md")
            and not item.name.startswith("_")
        }

    def assert_document_count(self, expected_count: int) -> None:
        """Assert that the system contains exactly the expected number of documents.

        Args:
            expected_count: Expected number of documents

        Raises:
            AssertionError: If document count doesn't match
        """
        actual_count = len(self.get_document_names())
        assert actual_count == expected_count, (
            f"Expected {expected_count} documents, but found {actual_count}. "
            f"Documents: {self.get_document_names()}"
        )

    def assert_chapter_count(self, doc_name: str, expected_count: int) -> None:
        """Assert that a document contains exactly the expected number of chapters.

        Args:
            doc_name: Name of the document
            expected_count: Expected number of chapters

        Raises:
            AssertionError: If chapter count doesn't match
        """
        actual_count = len(self.get_chapter_names(doc_name))
        assert actual_count == expected_count, (
            f"Expected {expected_count} chapters in document '{doc_name}', but found {actual_count}. "
            f"Chapters: {self.get_chapter_names(doc_name)}"
        )

    def get_debug_info(self) -> dict[str, any]:
        """Get debug information about the current file system state.

        Returns:
            Dictionary containing debug information about documents and chapters
        """
        debug_info = {
            "docs_root": str(self.docs_root),
            "docs_root_exists": self.docs_root.exists(),
            "documents": {},
        }

        if self.docs_root.exists():
            for doc_name in self.get_document_names():
                doc_path = self.docs_root / doc_name
                debug_info["documents"][doc_name] = {
                    "path": str(doc_path),
                    "chapters": list(self.get_chapter_names(doc_name)),
                    "all_files": [
                        item.name for item in doc_path.iterdir() if item.is_file()
                    ],
                }

        return debug_info


def safe_get_response_content(response: dict, field_name: str = "details") -> dict:
    """Safely extract content from agent response, handling various response formats.

    Args:
        response: Agent response dictionary
        field_name: Name of the field to extract ('details' or 'summary')

    Returns:
        Extracted content as dictionary, or empty dict if not found/invalid
        If parsing fails, includes error information for debugging
    """
    if not isinstance(response, dict):
        return {"_error": f"Response is not a dict, got {type(response).__name__}"}

    content = response.get(field_name)
    if content is None:
        return {"_error": f"Field '{field_name}' not found in response"}

    # Handle case where content is a string representation of a dict
    if isinstance(content, str):
        # Check if it looks like it should be parsed (starts with { or [)
        content_stripped = content.strip()
        if content_stripped.startswith(("{", "[")):
            try:
                return ast.literal_eval(content)
            except (ValueError, SyntaxError) as e:
                # This was likely meant to be parsed but failed - preserve error info
                return {
                    "content": content,
                    "_parse_error": f"Failed to parse as dict/list: {str(e)}",
                    "_original_content": content,
                }
        else:
            # Plain string content - this is expected
            return {"content": content}

    # Handle case where content is already a dict
    if isinstance(content, dict):
        return content

    # Handle other types
    return {"content": str(content), "_type_converted": type(content).__name__}


def ensure_proper_model_response(response: any) -> dict:
    """Ensure response is a proper dictionary, converting if necessary.

    Args:
        response: Response from agent (could be dict, model, or other type)

    Returns:
        Response as dictionary
    """
    if isinstance(response, dict):
        return response

    # Handle Pydantic models
    if hasattr(response, "model_dump"):
        return response.model_dump()

    # Handle other objects with dict conversion
    if hasattr(response, "__dict__"):
        return response.__dict__

    # Fallback - wrap in dict
    return {"raw_response": str(response)}
