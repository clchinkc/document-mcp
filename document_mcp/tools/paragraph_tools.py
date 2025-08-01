"""Paragraph Management Tools.

This module contains MCP tools for precise paragraph-level operations within chapters:
- replace_paragraph: Replace specific paragraph content
- insert_paragraph_before: Insert paragraph before specified index
- insert_paragraph_after: Insert paragraph after specified index
- delete_paragraph: Delete specific paragraph
- append_paragraph_to_chapter: Add paragraph to end of chapter
- move_paragraph_before: Move paragraph to before another paragraph
- move_paragraph_to_end: Move paragraph to end of chapter
"""

from mcp.server import FastMCP

from ..helpers import _count_words
from ..helpers import _get_chapter_path
from ..helpers import _is_valid_chapter_filename
from ..helpers import _split_into_paragraphs
from ..logger_config import log_mcp_call
from ..models import OperationStatus
from ..models import ParagraphDetail
from ..utils.decorators import auto_snapshot
from ..utils.decorators import safety_enhanced_write_operation
from ..utils.validation import validate_chapter_name
from ..utils.validation import validate_content
from ..utils.validation import validate_document_name
from ..utils.validation import validate_paragraph_index


def register_paragraph_tools(mcp_server: FastMCP) -> None:
    """Register all paragraph management tools with the MCP server."""

    @mcp_server.tool()
    @log_mcp_call
    @auto_snapshot("replace_paragraph")
    @safety_enhanced_write_operation("replace_paragraph")
    def replace_paragraph(
        document_name: str,
        chapter_name: str,
        paragraph_index: int,
        new_content: str,
        last_known_modified: str | None = None,
        force_write: bool = False,
    ) -> OperationStatus:
        """Replace the content of a specific paragraph within a chapter.

        This atomic tool replaces an existing paragraph at the specified index with
        new content. The paragraph index is zero-based, and the operation will fail
        if the index is out of bounds.

        SAFETY FEATURES:
        - Checks file modification time before writing to detect external changes
        - Creates automatic micro-snapshots before destructive operations
        - Records all modifications in document history
        - Provides detailed safety warnings and recommendations

        Parameters:
            document_name (str): Name of the document directory containing the chapter
            chapter_name (str): Filename of the chapter to modify (must end with .md)
            paragraph_index (int): Zero-indexed position of the paragraph to replace (≥0)
            new_content (str): New content to replace the existing paragraph with
            last_known_modified (Optional[str]): ISO timestamp of last known modification
                for safety checking. If provided, will warn if file was modified externally.
            force_write (bool): If True, will proceed with write even if safety warnings exist.
                Default is False for maximum safety.

        Returns:
            OperationStatus: Enhanced result object with safety information

        Example Usage:
            ```json
            {
                "name": "replace_paragraph",
                "arguments": {
                    "document_name": "user_guide",
                    "chapter_name": "01-intro.md",
                    "paragraph_index": 2,
                    "new_content": "This is the updated paragraph content."
                }
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

        is_valid_index, index_error = validate_paragraph_index(paragraph_index)
        if not is_valid_index:
            return OperationStatus(success=False, message=index_error)

        is_valid_content, content_error = validate_content(new_content)
        if not is_valid_content:
            return OperationStatus(success=False, message=content_error)

        chapter_path = _get_chapter_path(document_name, chapter_name)
        if not chapter_path.is_file() or not _is_valid_chapter_filename(chapter_name):
            return OperationStatus(
                success=False,
                message=f"Chapter '{chapter_name}' not found in document '{document_name}'.",
            )

        try:
            # Read current content
            current_content = chapter_path.read_text(encoding="utf-8")
            paragraphs = _split_into_paragraphs(current_content)

            # Check if index is valid
            if paragraph_index >= len(paragraphs):
                return OperationStatus(
                    success=False,
                    message=f"Paragraph index {paragraph_index} is out of bounds. Chapter has {len(paragraphs)} paragraphs.",
                )

            # Replace the paragraph
            paragraphs[paragraph_index] = new_content.strip()

            # Join paragraphs back with double newlines
            new_content_full = "\n\n".join(paragraphs)

            # Write back to file
            chapter_path.write_text(new_content_full, encoding="utf-8")

            return OperationStatus(
                success=True,
                message=f"Paragraph {paragraph_index} in chapter '{chapter_name}' replaced successfully.",
                details={
                    "document_name": document_name,
                    "chapter_name": chapter_name,
                    "paragraph_index": paragraph_index,
                },
            )

        except Exception as e:
            return OperationStatus(success=False, message=f"Error replacing paragraph: {e}")

    @mcp_server.tool()
    @log_mcp_call
    @auto_snapshot("insert_paragraph_before")
    def insert_paragraph_before(
        document_name: str,
        chapter_name: str,
        paragraph_index: int,
        new_content: str,
        last_known_modified: str | None = None,
        force_write: bool = False,
    ) -> OperationStatus:
        """Insert a new paragraph before the specified index within a chapter.

        Parameters:
            document_name (str): Name of the document directory containing the chapter
            chapter_name (str): Filename of the chapter to modify (must end with .md)
            paragraph_index (int): Zero-indexed position to insert before (≥0)
            new_content (str): Content for the new paragraph
            last_known_modified (Optional[str]): ISO timestamp for safety checking
            force_write (bool): If True, proceed even with safety warnings

        Returns:
            OperationStatus: Result object with operation details
        """
        # Validate inputs (same as replace_paragraph)
        is_valid_doc, doc_error = validate_document_name(document_name)
        if not is_valid_doc:
            return OperationStatus(success=False, message=doc_error)

        is_valid_chapter, chapter_error = validate_chapter_name(chapter_name)
        if not is_valid_chapter:
            return OperationStatus(success=False, message=chapter_error)

        is_valid_index, index_error = validate_paragraph_index(paragraph_index)
        if not is_valid_index:
            return OperationStatus(success=False, message=index_error)

        is_valid_content, content_error = validate_content(new_content)
        if not is_valid_content:
            return OperationStatus(success=False, message=content_error)

        chapter_path = _get_chapter_path(document_name, chapter_name)
        if not chapter_path.is_file() or not _is_valid_chapter_filename(chapter_name):
            return OperationStatus(
                success=False,
                message=f"Chapter '{chapter_name}' not found in document '{document_name}'.",
            )

        try:
            # Read current content
            current_content = chapter_path.read_text(encoding="utf-8")
            paragraphs = _split_into_paragraphs(current_content)

            # Insert the paragraph
            paragraphs.insert(paragraph_index, new_content.strip())

            # Join paragraphs back with double newlines
            new_content_full = "\n\n".join(paragraphs)

            # Write back to file
            chapter_path.write_text(new_content_full, encoding="utf-8")

            return OperationStatus(
                success=True,
                message=f"Paragraph inserted before index {paragraph_index} in chapter '{chapter_name}' successfully.",
                details={
                    "document_name": document_name,
                    "chapter_name": chapter_name,
                    "paragraph_index": paragraph_index,
                },
            )

        except Exception as e:
            return OperationStatus(success=False, message=f"Error inserting paragraph: {e}")

    @mcp_server.tool()
    @log_mcp_call
    @auto_snapshot("insert_paragraph_after")
    def insert_paragraph_after(
        document_name: str, chapter_name: str, paragraph_index: int, new_content: str
    ) -> OperationStatus:
        """Insert a new paragraph after the specified index within a chapter.

        Parameters:
            document_name (str): Name of the document directory containing the chapter
            chapter_name (str): Filename of the chapter to modify (must end with .md)
            paragraph_index (int): Zero-indexed position to insert after (≥0)
            new_content (str): Content for the new paragraph

        Returns:
            OperationStatus: Result object with operation details
        """
        # Validate inputs
        is_valid_doc, doc_error = validate_document_name(document_name)
        if not is_valid_doc:
            return OperationStatus(success=False, message=doc_error)

        is_valid_chapter, chapter_error = validate_chapter_name(chapter_name)
        if not is_valid_chapter:
            return OperationStatus(success=False, message=chapter_error)

        is_valid_index, index_error = validate_paragraph_index(paragraph_index)
        if not is_valid_index:
            return OperationStatus(success=False, message=index_error)

        is_valid_content, content_error = validate_content(new_content)
        if not is_valid_content:
            return OperationStatus(success=False, message=content_error)

        chapter_path = _get_chapter_path(document_name, chapter_name)
        if not chapter_path.is_file() or not _is_valid_chapter_filename(chapter_name):
            return OperationStatus(
                success=False,
                message=f"Chapter '{chapter_name}' not found in document '{document_name}'.",
            )

        try:
            # Read current content
            current_content = chapter_path.read_text(encoding="utf-8")
            paragraphs = _split_into_paragraphs(current_content)

            # Check if index is valid
            if paragraph_index >= len(paragraphs):
                return OperationStatus(
                    success=False,
                    message=f"Paragraph index {paragraph_index} is out of bounds. Chapter has {len(paragraphs)} paragraphs.",
                )

            # Insert the paragraph after the specified index
            paragraphs.insert(paragraph_index + 1, new_content.strip())

            # Join paragraphs back with double newlines
            new_content_full = "\n\n".join(paragraphs)

            # Write back to file
            chapter_path.write_text(new_content_full, encoding="utf-8")

            return OperationStatus(
                success=True,
                message=f"Paragraph inserted after index {paragraph_index} in chapter '{chapter_name}' successfully.",
                details={
                    "document_name": document_name,
                    "chapter_name": chapter_name,
                    "paragraph_index": paragraph_index + 1,
                },
            )

        except Exception as e:
            return OperationStatus(success=False, message=f"Error inserting paragraph: {e}")

    @mcp_server.tool()
    @log_mcp_call
    @auto_snapshot("delete_paragraph")
    def delete_paragraph(
        document_name: str,
        chapter_name: str,
        paragraph_index: int,
        last_known_modified: str | None = None,
        force_write: bool = False,
    ) -> OperationStatus:
        """Delete a specific paragraph from a chapter.

        Parameters:
            document_name (str): Name of the document directory containing the chapter
            chapter_name (str): Filename of the chapter to modify (must end with .md)
            paragraph_index (int): Zero-indexed position of the paragraph to delete (≥0)
            last_known_modified (Optional[str]): ISO timestamp for safety checking
            force_write (bool): If True, proceed even with safety warnings

        Returns:
            OperationStatus: Result object with operation details
        """
        # Validate inputs
        is_valid_doc, doc_error = validate_document_name(document_name)
        if not is_valid_doc:
            return OperationStatus(success=False, message=doc_error)

        is_valid_chapter, chapter_error = validate_chapter_name(chapter_name)
        if not is_valid_chapter:
            return OperationStatus(success=False, message=chapter_error)

        is_valid_index, index_error = validate_paragraph_index(paragraph_index)
        if not is_valid_index:
            return OperationStatus(success=False, message=index_error)

        chapter_path = _get_chapter_path(document_name, chapter_name)
        if not chapter_path.is_file() or not _is_valid_chapter_filename(chapter_name):
            return OperationStatus(
                success=False,
                message=f"Chapter '{chapter_name}' not found in document '{document_name}'.",
            )

        try:
            # Read current content
            current_content = chapter_path.read_text(encoding="utf-8")
            paragraphs = _split_into_paragraphs(current_content)

            # Check if index is valid
            if paragraph_index >= len(paragraphs):
                return OperationStatus(
                    success=False,
                    message=f"Paragraph index {paragraph_index} is out of bounds. Chapter has {len(paragraphs)} paragraphs.",
                )

            # Delete the paragraph
            deleted_content = paragraphs.pop(paragraph_index)

            # Join paragraphs back with double newlines
            new_content_full = "\n\n".join(paragraphs)

            # Write back to file
            chapter_path.write_text(new_content_full, encoding="utf-8")

            return OperationStatus(
                success=True,
                message=f"Paragraph {paragraph_index} deleted from chapter '{chapter_name}' successfully.",
                details={
                    "document_name": document_name,
                    "chapter_name": chapter_name,
                    "paragraph_index": paragraph_index,
                    "deleted_content": deleted_content,
                },
            )

        except Exception as e:
            return OperationStatus(success=False, message=f"Error deleting paragraph: {e}")

    @mcp_server.tool()
    @log_mcp_call
    @auto_snapshot("append_paragraph_to_chapter")
    def append_paragraph_to_chapter(
        document_name: str, chapter_name: str, paragraph_content: str
    ) -> OperationStatus:
        """Add a new paragraph to the end of a chapter.

        Parameters:
            document_name (str): Name of the document directory containing the chapter
            chapter_name (str): Filename of the chapter to modify (must end with .md)
            paragraph_content (str): Content for the new paragraph

        Returns:
            OperationStatus: Result object with operation details
        """
        # Validate inputs
        is_valid_doc, doc_error = validate_document_name(document_name)
        if not is_valid_doc:
            return OperationStatus(success=False, message=doc_error)

        is_valid_chapter, chapter_error = validate_chapter_name(chapter_name)
        if not is_valid_chapter:
            return OperationStatus(success=False, message=chapter_error)

        is_valid_content, content_error = validate_content(paragraph_content)
        if not is_valid_content:
            return OperationStatus(success=False, message=content_error)

        chapter_path = _get_chapter_path(document_name, chapter_name)
        if not chapter_path.is_file() or not _is_valid_chapter_filename(chapter_name):
            return OperationStatus(
                success=False,
                message=f"Chapter '{chapter_name}' not found in document '{document_name}'.",
            )

        try:
            # Read current content
            current_content = chapter_path.read_text(encoding="utf-8")
            paragraphs = _split_into_paragraphs(current_content)

            # Append the paragraph
            paragraphs.append(paragraph_content.strip())

            # Join paragraphs back with double newlines
            new_content_full = "\n\n".join(paragraphs)

            # Write back to file
            chapter_path.write_text(new_content_full, encoding="utf-8")

            return OperationStatus(
                success=True,
                message=f"Paragraph appended to chapter '{chapter_name}' successfully.",
                details={
                    "document_name": document_name,
                    "chapter_name": chapter_name,
                    "paragraph_index": len(paragraphs) - 1,
                },
            )

        except Exception as e:
            return OperationStatus(success=False, message=f"Error appending paragraph: {e}")

    @mcp_server.tool()
    @log_mcp_call
    @auto_snapshot("move_paragraph_before")
    def move_paragraph_before(
        document_name: str,
        chapter_name: str,
        paragraph_to_move_index: int,
        target_paragraph_index: int,
    ) -> OperationStatus:
        """Move a paragraph to appear before another paragraph within the same chapter.

        This atomic tool reorders paragraphs within a chapter by moving the paragraph
        at the source index to appear before the paragraph at the target index.

        Parameters:
            document_name (str): Name of the document directory containing the chapter
            chapter_name (str): Filename of the chapter to modify (must end with .md)
            paragraph_to_move_index (int): Zero-indexed position of the paragraph to move (≥0)
            target_paragraph_index (int): Zero-indexed position to move before (≥0)

        Returns:
            OperationStatus: Result object with success status, message, and diff details
        """
        # Validate inputs
        is_valid_doc, doc_error = validate_document_name(document_name)
        if not is_valid_doc:
            return OperationStatus(success=False, message=doc_error)

        is_valid_chapter, chapter_error = validate_chapter_name(chapter_name)
        if not is_valid_chapter:
            return OperationStatus(success=False, message=chapter_error)

        is_valid_move_index, move_index_error = validate_paragraph_index(paragraph_to_move_index)
        if not is_valid_move_index:
            return OperationStatus(success=False, message=f"Move index: {move_index_error}")

        is_valid_target_index, target_index_error = validate_paragraph_index(target_paragraph_index)
        if not is_valid_target_index:
            return OperationStatus(success=False, message=f"Target index: {target_index_error}")

        if paragraph_to_move_index == target_paragraph_index:
            return OperationStatus(
                success=False,
                message="Cannot move a paragraph before itself.",
            )

        chapter_path = _get_chapter_path(document_name, chapter_name)
        if not chapter_path.is_file() or not _is_valid_chapter_filename(chapter_name):
            return OperationStatus(
                success=False,
                message=f"Chapter '{chapter_name}' not found in document '{document_name}'.",
            )

        try:
            original_full_content = chapter_path.read_text(encoding="utf-8")
            paragraphs = _split_into_paragraphs(original_full_content)
            total_paragraphs = len(paragraphs)

            if not (0 <= paragraph_to_move_index < total_paragraphs):
                return OperationStatus(
                    success=False,
                    message=f"Paragraph to move index {paragraph_to_move_index} is out of bounds (0-{total_paragraphs - 1}).",
                )

            if not (0 <= target_paragraph_index < total_paragraphs):
                return OperationStatus(
                    success=False,
                    message=f"Target paragraph index {target_paragraph_index} is out of bounds (0-{total_paragraphs - 1}).",
                )

            # Move the paragraph
            paragraph_to_move = paragraphs.pop(paragraph_to_move_index)

            # Adjust target index if necessary (if we moved from before the target)
            if paragraph_to_move_index < target_paragraph_index:
                target_paragraph_index -= 1

            paragraphs.insert(target_paragraph_index, paragraph_to_move)

            final_content = "\n\n".join(paragraphs)
            chapter_path.write_text(final_content, encoding="utf-8")

            # Generate diff for details
            from ..helpers import _generate_content_diff

            diff_info = _generate_content_diff(original_full_content, final_content, chapter_name)

            return OperationStatus(
                success=True,
                message=f"Paragraph {paragraph_to_move_index} moved before paragraph {target_paragraph_index} in '{chapter_name}' ({document_name}).",
                details=diff_info,
            )
        except Exception as e:
            return OperationStatus(
                success=False,
                message=f"Error moving paragraph in '{chapter_name}' ({document_name}): {str(e)}",
            )

    @mcp_server.tool()
    @log_mcp_call
    @auto_snapshot("move_paragraph_to_end")
    def move_paragraph_to_end(
        document_name: str, chapter_name: str, paragraph_to_move_index: int
    ) -> OperationStatus:
        """Move a paragraph to the end of a chapter.

        This atomic tool moves the paragraph at the specified index to the end of the
        chapter, after all other paragraphs.

        Parameters:
            document_name (str): Name of the document directory containing the chapter
            chapter_name (str): Filename of the chapter to modify (must end with .md)
            paragraph_to_move_index (int): Zero-indexed position of the paragraph to move (≥0)

        Returns:
            OperationStatus: Result object with success status, message, and diff details
        """
        # Validate inputs
        is_valid_doc, doc_error = validate_document_name(document_name)
        if not is_valid_doc:
            return OperationStatus(success=False, message=doc_error)

        is_valid_chapter, chapter_error = validate_chapter_name(chapter_name)
        if not is_valid_chapter:
            return OperationStatus(success=False, message=chapter_error)

        is_valid_index, index_error = validate_paragraph_index(paragraph_to_move_index)
        if not is_valid_index:
            return OperationStatus(success=False, message=index_error)

        chapter_path = _get_chapter_path(document_name, chapter_name)
        if not chapter_path.is_file() or not _is_valid_chapter_filename(chapter_name):
            return OperationStatus(
                success=False,
                message=f"Chapter '{chapter_name}' not found in document '{document_name}'.",
            )

        try:
            original_full_content = chapter_path.read_text(encoding="utf-8")
            paragraphs = _split_into_paragraphs(original_full_content)
            total_paragraphs = len(paragraphs)

            if not (0 <= paragraph_to_move_index < total_paragraphs):
                return OperationStatus(
                    success=False,
                    message=f"Paragraph index {paragraph_to_move_index} is out of bounds (0-{total_paragraphs - 1}).",
                )

            # If already at the end, no need to move
            if paragraph_to_move_index == total_paragraphs - 1:
                return OperationStatus(
                    success=True,
                    message=f"Paragraph {paragraph_to_move_index} is already at the end of '{chapter_name}' ({document_name}).",
                    details={
                        "changed": False,
                        "summary": "No changes made - paragraph already at end",
                    },
                )

            # Move the paragraph to the end
            paragraph_to_move = paragraphs.pop(paragraph_to_move_index)
            paragraphs.append(paragraph_to_move)

            final_content = "\n\n".join(paragraphs)
            chapter_path.write_text(final_content, encoding="utf-8")

            # Generate diff for details
            from ..helpers import _generate_content_diff

            diff_info = _generate_content_diff(original_full_content, final_content, chapter_name)

            return OperationStatus(
                success=True,
                message=f"Paragraph {paragraph_to_move_index} moved to end of '{chapter_name}' ({document_name}).",
                details=diff_info,
            )
        except Exception as e:
            return OperationStatus(
                success=False,
                message=f"Error moving paragraph to end in '{chapter_name}' ({document_name}): {str(e)}",
            )

    @mcp_server.tool()
    @log_mcp_call
    def read_paragraph_content(
        document_name: str, chapter_name: str, paragraph_index_in_chapter: int
    ) -> ParagraphDetail | None:
        """Retrieve the content and metadata of a specific paragraph within a chapter.

        This tool extracts a single paragraph from a chapter file using zero-indexed
        positioning. Paragraphs are defined as text blocks separated by blank lines.
        Useful for targeted content retrieval and editing operations.

        Parameters:
            document_name (str): Name of the document directory containing the chapter
            chapter_name (str): Filename of the chapter containing the paragraph
            paragraph_index_in_chapter (int): Zero-indexed position of the paragraph

        Returns:
            Optional[ParagraphDetail]: Paragraph content object if found, None if not found.
        """
        chapter_file_path = _get_chapter_path(document_name, chapter_name)
        if not chapter_file_path.is_file() or not _is_valid_chapter_filename(chapter_name):
            return None

        try:
            content = chapter_file_path.read_text(encoding="utf-8")
            paragraphs = _split_into_paragraphs(content)

            if paragraph_index_in_chapter < 0 or paragraph_index_in_chapter >= len(paragraphs):
                return None

            paragraph_content = paragraphs[paragraph_index_in_chapter]
            word_count = _count_words(paragraph_content)

            return ParagraphDetail(
                document_name=document_name,
                chapter_name=chapter_name,
                paragraph_index_in_chapter=paragraph_index_in_chapter,
                content=paragraph_content,
                word_count=word_count,
            )

        except Exception:
            return None
