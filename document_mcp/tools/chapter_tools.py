"""Chapter Management Tools.

This module contains MCP tools for managing chapter files within documents:
- list_chapters: List all chapter files within a document
- create_chapter: Create a new chapter file with optional initial content
- delete_chapter: Delete a chapter file from a document
- write_chapter_content: Overwrite entire chapter content
"""

import datetime
import difflib
from pathlib import Path
from typing import Any

from mcp.server import FastMCP

from ..helpers import _count_words
from ..helpers import _get_chapter_metadata
from ..helpers import _get_chapter_path
from ..helpers import _get_document_path
from ..helpers import _get_ordered_chapter_files
from ..helpers import _is_valid_chapter_filename
from ..helpers import _split_into_paragraphs
from ..logger_config import log_mcp_call
from ..models import ChapterContent
from ..models import ChapterMetadata
from ..models import OperationStatus
from ..utils.decorators import auto_snapshot
from ..utils.decorators import safety_enhanced_write_operation
from ..utils.validation import CHAPTER_MANIFEST_FILE
from ..utils.validation import validate_chapter_name
from ..utils.validation import validate_content
from ..utils.validation import validate_document_name

# Note: All helper functions are imported from helpers module above


def register_chapter_tools(mcp_server: FastMCP) -> None:
    """Register all chapter management tools with the MCP server."""

    @mcp_server.tool()
    @log_mcp_call
    def list_chapters(document_name: str) -> list[ChapterMetadata] | None:
        """List all chapter files within a specified document, ordered by filename.

        This tool retrieves metadata for all chapter files (.md) within a document directory.
        Chapters are automatically ordered alphanumerically by filename, which typically
        corresponds to their intended sequence (e.g., 01-intro.md, 02-setup.md).

        Parameters:
            document_name (str): Name of the document directory to list chapters from

        Returns:
            Optional[List[ChapterMetadata]]: List of chapter metadata objects if document exists,
            None if document not found. Each ChapterMetadata contains:
                - chapter_name (str): Filename of the chapter (e.g., "01-introduction.md")
                - title (Optional[str]): Chapter title (currently None, reserved for future use)
                - word_count (int): Total number of words in the chapter
                - paragraph_count (int): Total number of paragraphs in the chapter
                - last_modified (datetime): Timestamp of last file modification

            Returns empty list [] if document exists but contains no valid chapter files.
            Returns None if the document directory does not exist.

        Example Usage:
            ```json
            {
                "name": "list_chapters",
                "arguments": {
                    "document_name": "user_guide"
                }
            }
            ```

        Example Success Response:
            ```json
            [
                {
                    "chapter_name": "01-introduction.md",
                    "title": null,
                    "word_count": 342,
                    "paragraph_count": 8,
                    "last_modified": "2024-01-15T10:30:00Z"
                },
                {
                    "chapter_name": "02-getting-started.md",
                    "title": null,
                    "word_count": 567,
                    "paragraph_count": 15,
                    "last_modified": "2024-01-15T11:45:00Z"
                }
            ]
            ```

        Example Not Found Response:
            ```json
            null
            ```
        """
        doc_path = _get_document_path(document_name)
        if not doc_path.is_dir():
            print(f"Document '{document_name}' not found at {doc_path}")
            return None  # Or perhaps OperationStatus(success=False, message="Document not found")
            # For now, following Optional[List[...]] pattern for read lists

        ordered_chapter_files = _get_ordered_chapter_files(document_name)
        chapters_metadata_list = []
        for chapter_file_path in ordered_chapter_files:
            metadata = _get_chapter_metadata(document_name, chapter_file_path)
            if metadata:
                chapters_metadata_list.append(metadata)

        if not ordered_chapter_files and not chapters_metadata_list:
            # If the directory exists but has no valid chapter files, return empty list.
            return []

        return chapters_metadata_list

    @mcp_server.tool()
    @log_mcp_call
    @auto_snapshot("create_chapter")
    def create_chapter(document_name: str, chapter_name: str, initial_content: str = "") -> OperationStatus:
        r"""Create a new chapter file within an existing document directory.

        .. note::
           While this tool supports creating a chapter with initial content, for clarity it is recommended to create an empty chapter and then use a dedicated write/update tool to add content. This helps separate creation from modification operations.

        This tool adds a new Markdown chapter file to a document collection. The chapter
        will be ordered based on its filename when listed with other chapters. Supports
        optional initial content to bootstrap the chapter with starter text.

        Parameters:
            document_name (str): Name of the existing document directory to add chapter to
            chapter_name (str): Filename for the new chapter. Must be:
                - Valid .md filename (e.g., "03-advanced-features.md")
                - ≤100 characters
                - Valid filesystem filename
                - Cannot contain path separators (/ or \\)
                - Cannot be reserved name like "_manifest.json" or "_SUMMARY.md"
                - Must not already exist in the document
            initial_content (str, optional): Starting content for the new chapter. Can be:
                - Empty string (default): Creates empty chapter file
                - Any valid UTF-8 text content
                - Must be ≤1MB in size

        Returns:
            OperationStatus: Structured result object containing:
                - success (bool): True if chapter was created successfully, False otherwise
                - message (str): Human-readable description of the operation result
                - details (Dict[str, Any], optional): Additional context including:
                    - document_name (str): Name of the parent document (on success)
                    - chapter_name (str): Name of the created chapter (on success)

        Example Usage:
            ```json
            {
                "name": "create_chapter",
                "arguments": {
                    "document_name": "user_guide",
                    "chapter_name": "03-advanced-features.md",
                    "initial_content": "# Advanced Features\n\nThis chapter covers advanced functionality."
                }
            }
            ```

        Example Success Response:
            ```json
            {
                "success": true,
                "message": "Chapter '03-advanced-features.md' created successfully in document 'user_guide'.",
                "details": {
                    "document_name": "user_guide",
                    "chapter_name": "03-advanced-features.md"
                }
            }
            ```

        Example Error Response:
            ```json
            {
                "success": false,
                "message": "Chapter '03-advanced-features.md' already exists in document 'user_guide'.",
                "details": null
            }
            ```
        """
        # Validate inputs
        is_valid_doc, doc_error = validate_document_name(document_name)
        if not is_valid_doc:
            return OperationStatus(success=False, message=doc_error)

        is_valid_chapter, chapter_error = validate_chapter_name(chapter_name)
        if not is_valid_chapter:
            return OperationStatus(success=False, message=chapter_error)

        is_valid_content, content_error = validate_content(initial_content)
        if not is_valid_content:
            return OperationStatus(success=False, message=content_error)

        doc_path = _get_document_path(document_name)
        if not doc_path.is_dir():
            return OperationStatus(success=False, message=f"Document '{document_name}' not found.")

        if not _is_valid_chapter_filename(chapter_name):
            return OperationStatus(
                success=False,
                message=f"Invalid chapter name '{chapter_name}'. Must be a .md file and not a reserved name like '{CHAPTER_MANIFEST_FILE}'.",
            )

        chapter_path = _get_chapter_path(document_name, chapter_name)
        if chapter_path.exists():
            return OperationStatus(
                success=False,
                message=f"Chapter '{chapter_name}' already exists in document '{document_name}'.",
            )

        try:
            chapter_path.write_text(initial_content, encoding="utf-8")
            return OperationStatus(
                success=True,
                message=f"Chapter '{chapter_name}' created successfully in document '{document_name}'.",
                details={"document_name": document_name, "chapter_name": chapter_name},
            )
        except Exception as e:
            return OperationStatus(
                success=False,
                message=f"Error creating chapter '{chapter_name}' in document '{document_name}': {e}",
            )

    @mcp_server.tool()
    @log_mcp_call
    @auto_snapshot("delete_chapter")
    def delete_chapter(document_name: str, chapter_name: str) -> OperationStatus:
        """Delete a chapter file from a document directory.

        This tool permanently removes a chapter file from a document collection.
        The operation is irreversible and will delete the file from the filesystem.
        Use with caution as the chapter content will be lost.

        Parameters:
            document_name (str): Name of the document directory containing the chapter
            chapter_name (str): Filename of the chapter to delete. Must be:
                - Valid .md filename (e.g., "02-old-chapter.md")
                - Existing file within the specified document directory
                - Not a reserved filename like "_manifest.json"

        Returns:
            OperationStatus: Structured result object containing:
                - success (bool): True if chapter was deleted successfully, False otherwise
                - message (str): Human-readable description of the operation result
                - details (Dict[str, Any], optional): Additional context (currently None)

        Example Usage:
            ```json
            {
                "name": "delete_chapter",
                "arguments": {
                    "document_name": "user_guide",
                    "chapter_name": "02-outdated-section.md"
                }
            }
            ```

        Example Success Response:
            ```json
            {
                "success": true,
                "message": "Chapter '02-outdated-section.md' deleted successfully from document 'user_guide'.",
                "details": null
            }
            ```

        Example Error Response:
            ```json
            {
                "success": false,
                "message": "Chapter '02-outdated-section.md' not found in document 'user_guide'.",
                "details": null
            }
            ```
        """
        if not _is_valid_chapter_filename(chapter_name):  # Check early to avoid issues with non-MD files
            return OperationStatus(
                success=False,
                message=f"Invalid target '{chapter_name}'. Not a valid chapter Markdown file name.",
            )

        chapter_path = _get_chapter_path(document_name, chapter_name)
        if not chapter_path.is_file():
            return OperationStatus(
                success=False,
                message=f"Chapter '{chapter_name}' not found in document '{document_name}'.",
            )

        try:
            chapter_path.unlink()
            return OperationStatus(
                success=True,
                message=f"Chapter '{chapter_name}' deleted successfully from document '{document_name}'.",
            )
        except Exception as e:
            return OperationStatus(
                success=False,
                message=f"Error deleting chapter '{chapter_name}' from document '{document_name}': {e}",
            )

    @mcp_server.tool()
    @log_mcp_call
    @auto_snapshot("write_chapter_content")
    @safety_enhanced_write_operation("write_chapter_content")
    def write_chapter_content(
        document_name: str,
        chapter_name: str,
        new_content: str,
        last_known_modified: str | None = None,
        force_write: bool = False,
    ) -> OperationStatus:
        r"""Overwrite the entire content of a chapter file with new content.

        .. deprecated:: 0.18.0
           This tool's behavior of creating a chapter if it doesn't exist is deprecated and will be removed in a future version.
           In the future, this tool will only write to existing chapters. Please use `create_chapter` for new chapters.

        This tool completely replaces the content of an existing chapter file or creates
        a new chapter if it doesn't exist. The operation provides diff information showing
        exactly what changed between the original and new content.

        SAFETY FEATURES:
        - Checks file modification time before writing to detect external changes
        - Creates automatic micro-snapshots before destructive operations
        - Records all modifications in document history
        - Provides detailed safety warnings and recommendations

        Parameters:
            document_name (str): Name of the document directory containing the chapter
            chapter_name (str): Filename of the chapter to write. Must be:
                - Valid .md filename (e.g., "01-introduction.md")
                - Valid filesystem filename
                - Not a reserved filename like "_manifest.json"
            new_content (str): Complete new content for the chapter file. Can be:
                - Any valid UTF-8 text content
                - Empty string (creates empty chapter)
                - Must be ≤1MB in size
            last_known_modified (Optional[str]): ISO timestamp of last known modification
                for safety checking. If provided, will warn if file was modified externally.
            force_write (bool): If True, will proceed with write even if safety warnings exist.
                Default is False for maximum safety.

        Returns:
            OperationStatus: Enhanced result object containing:
                - success (bool): True if content was written successfully, False otherwise
                - message (str): Human-readable description of the operation result
                - details (Dict[str, Any], optional): Includes diff information:
                    - changed (bool): Whether content was actually modified
                    - diff (str): Unified diff showing changes made
                    - summary (str): Brief description of changes
                    - lines_added (int): Number of lines added
                    - lines_removed (int): Number of lines removed
                - safety_info (ContentFreshnessStatus): Safety check results
                - snapshot_created (Optional[str]): ID of safety snapshot if created
                - warnings (List[str]): List of safety warnings

        Example Usage:
            ```json
            {
                "name": "write_chapter_content",
                "arguments": {
                    "document_name": "user_guide",
                    "chapter_name": "01-introduction.md",
                    "new_content": "# Introduction\n\nWelcome to our comprehensive user guide.\n\nThis guide will help you get started."
                }
            }
            ```

        Example Success Response:
            ```json
            {
                "success": true,
                "message": "Content of chapter '01-introduction.md' in document 'user_guide' updated successfully.",
                "details": {
                    "changed": true,
                    "diff": "--- 01-introduction.md (before)\n+++ 01-introduction.md (after)\n@@ -1,3 +1,4 @@\n # Introduction\n \n-Welcome to our guide.\n+Welcome to our comprehensive user guide.\n+\n+This guide will help you get started.",
                    "summary": "Modified content: +3 lines, -1 lines",
                    "lines_added": 3,
                    "lines_removed": 1
                }
            }
            ```
        """
        doc_path = _get_document_path(document_name)
        if not doc_path.is_dir():
            return OperationStatus(success=False, message=f"Document '{document_name}' not found.")

        if not _is_valid_chapter_filename(chapter_name):
            return OperationStatus(success=False, message=f"Invalid chapter name '{chapter_name}'.")

        chapter_path = _get_chapter_path(document_name, chapter_name)

        try:
            # Read original content before overwriting
            original_content = ""
            if chapter_path.exists():
                original_content = chapter_path.read_text(encoding="utf-8")

            # Write new content
            chapter_path.write_text(new_content, encoding="utf-8")

            # Generate diff for details
            diff_info = _generate_content_diff(original_content, new_content, chapter_name)

            return OperationStatus(
                success=True,
                message=f"Content of chapter '{chapter_name}' in document '{document_name}' updated successfully.",
                details=diff_info,
            )
        except Exception as e:
            return OperationStatus(
                success=False,
                message=f"Error writing to chapter '{chapter_name}' in document '{document_name}': {e}",
            )


def read_chapter_content(document_name: str, chapter_name: str) -> ChapterContent | None:
    r"""Retrieve the complete content and metadata of a specific chapter within a document.

    This tool reads a chapter file (.md) from a document directory and returns both
    the raw content and associated metadata including word counts, paragraph counts,
    and modification timestamps. Returns None if the chapter or document is not found.

    Parameters:
        document_name (str): Name of the document directory containing the chapter
        chapter_name (str): Filename of the chapter to read. Must be:
            - Valid .md filename (e.g., "01-introduction.md")
            - Existing file within the specified document directory
            - Not a reserved filename like "_manifest.json"

    Returns:
        Optional[ChapterContent]: Chapter content object if found, None if not found.
        ChapterContent contains:
            - document_name (str): Name of the parent document
            - chapter_name (str): Filename of the chapter
            - content (str): Full raw text content of the chapter file
            - word_count (int): Total number of words in the chapter
            - paragraph_count (int): Total number of paragraphs in the chapter
            - last_modified (datetime): Timestamp of last file modification

        Returns None if document doesn't exist, chapter doesn't exist, or chapter
        filename is invalid.

    Example Usage:
        ```json
        {
            "name": "read_chapter_content",
            "arguments": {
                "document_name": "user_guide",
                "chapter_name": "01-introduction.md"
            }
        }
        ```

    Example Success Response:
        ```json
        {
            "document_name": "user_guide",
            "chapter_name": "01-introduction.md",
            "content": "# Introduction\n\nWelcome to our user guide...",
            "word_count": 342,
            "paragraph_count": 8,
            "last_modified": "2024-01-15T10:30:00Z"
        }
        ```

    Example Not Found Response:
        ```json
        null
        ```
    """
    chapter_file_path = _get_chapter_path(document_name, chapter_name)
    if not chapter_file_path.is_file() or not _is_valid_chapter_filename(chapter_name):
        print(
            f"Chapter '{chapter_name}' not found or invalid in document '{document_name}' at {chapter_file_path}"
        )
        return None
    return _read_chapter_content_details(document_name, chapter_file_path)


def _read_chapter_content_details(document_name: str, chapter_file_path: Path) -> ChapterContent | None:
    """Read the content and metadata of a chapter from its file path."""
    if not chapter_file_path.is_file():
        return None
    try:
        content = chapter_file_path.read_text(encoding="utf-8")
        paragraphs = _split_into_paragraphs(content)
        word_count = _count_words(content)
        stat = chapter_file_path.stat()
        return ChapterContent(
            document_name=document_name,
            chapter_name=chapter_file_path.name,
            content=content,
            word_count=word_count,
            paragraph_count=len(paragraphs),
            last_modified=datetime.datetime.fromtimestamp(stat.st_mtime, tz=datetime.timezone.utc),
        )
    except Exception as e:
        from ..logger_config import ErrorCategory
        from ..logger_config import log_structured_error

        log_structured_error(
            category=ErrorCategory.ERROR,
            message=f"Failed to read chapter file: {chapter_file_path.name}",
            exception=e,
            context={
                "document_name": document_name,
                "chapter_file_path": str(chapter_file_path),
                "file_exists": chapter_file_path.exists(),
            },
            operation="read_chapter_content",
        )
        return None


def _generate_content_diff(
    original_content: str, new_content: str, filename: str = "chapter"
) -> dict[str, Any]:
    """Generate a unified diff between original and new content.

    Compares two strings and produces a diff report including a summary,
    lines added/removed, and the full unified diff text.
    """
    if original_content == new_content:
        return {"changed": False, "diff": None, "summary": "No changes made to content"}

    original_lines = original_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)

    diff_lines = list(
        difflib.unified_diff(
            original_lines,
            new_lines,
            fromfile=f"{filename} (before)",
            tofile=f"{filename} (after)",
            lineterm="",
        )
    )

    added_lines = sum(1 for line in diff_lines if line.startswith("+") and not line.startswith("+++"))
    removed_lines = sum(1 for line in diff_lines if line.startswith("-") and not line.startswith("---"))

    diff_text = "\n".join(diff_lines)
    summary = f"Modified content: +{added_lines} lines, -{removed_lines} lines"

    return {
        "changed": True,
        "diff": diff_text,
        "summary": summary,
        "lines_added": added_lines,
        "lines_removed": removed_lines,
    }
