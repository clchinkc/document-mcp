"""Paragraph Management Tools.

This module provides 4 tools for paragraph-level operations within chapters:
- add_paragraph: Add new paragraph (before index, after index, or at end)
- replace_paragraph: Replace content at specific index
- delete_paragraph: Remove paragraph at index
- move_paragraph: Reorder paragraphs (before target or to end)

Design rationale (from A/B testing):
- Replace kept separate: 50% failure rate when bundled with other ops
- Delete kept separate: Destructive operation needs clear intent
- Add combines insert before/after/append: Same semantics (adding new content)
- Move combines before/to_end: Same semantics (reordering)
"""

from typing import Literal

from mcp.server import FastMCP

from ..helpers import _generate_content_diff
from ..helpers import _get_chapter_path
from ..helpers import _is_valid_chapter_filename
from ..helpers import _split_into_paragraphs
from ..logger_config import log_mcp_call
from ..models import OperationStatus
from ..utils.decorators import auto_snapshot
from ..utils.validation import validate_chapter_name
from ..utils.validation import validate_content
from ..utils.validation import validate_document_name
from ..utils.validation import validate_paragraph_index


def register_paragraph_tools(mcp_server: FastMCP) -> None:
    """Register paragraph management tools with the MCP server."""

    @mcp_server.tool()
    @log_mcp_call
    @auto_snapshot("add_paragraph")
    def add_paragraph(
        document_name: str,
        chapter_name: str,
        content: str,
        position: Literal["before", "after", "end"] = "end",
        paragraph_index: int | None = None,
    ) -> OperationStatus:
        """Add a new paragraph at a specified position.

        Parameters:
            document_name (str): Name of the document directory
            chapter_name (str): Filename of the chapter (must end with .md)
            content (str): Content for the new paragraph
            position (str): "before" (above target), "after" (below target), or "end" (append)
            paragraph_index (int | None): Target index. Required for "before"/"after", omit for "end"

        Returns:
            OperationStatus: Result with success status and new paragraph index

        Examples:
            Insert before paragraph 2:
            {"content": "New text", "position": "before", "paragraph_index": 2}

            Insert after paragraph 3:
            {"content": "New text", "position": "after", "paragraph_index": 3}

            Append to end:
            {"content": "New text", "position": "end"}
        """
        # Validate inputs
        is_valid_doc, doc_error = validate_document_name(document_name)
        if not is_valid_doc:
            return OperationStatus(success=False, message=doc_error)

        is_valid_chapter, chapter_error = validate_chapter_name(chapter_name)
        if not is_valid_chapter:
            return OperationStatus(success=False, message=chapter_error)

        is_valid_content, content_error = validate_content(content)
        if not is_valid_content:
            return OperationStatus(success=False, message=content_error)

        # Validate index requirement
        if position in ("before", "after") and paragraph_index is None:
            return OperationStatus(
                success=False,
                message=f"paragraph_index is required when position is '{position}'",
            )

        if paragraph_index is not None:
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
            current_content = chapter_path.read_text(encoding="utf-8")
            paragraphs = _split_into_paragraphs(current_content)

            if position == "before":
                paragraphs.insert(paragraph_index, content.strip())
                new_index = paragraph_index
                result_message = f"Paragraph inserted before index {paragraph_index} in '{chapter_name}'"

            elif position == "after":
                if paragraph_index >= len(paragraphs):
                    return OperationStatus(
                        success=False,
                        message=f"Paragraph index {paragraph_index} is out of bounds. Chapter has {len(paragraphs)} paragraphs.",
                    )
                paragraphs.insert(paragraph_index + 1, content.strip())
                new_index = paragraph_index + 1
                result_message = f"Paragraph inserted after index {paragraph_index} in '{chapter_name}'"

            else:  # end
                paragraphs.append(content.strip())
                new_index = len(paragraphs) - 1
                result_message = f"Paragraph appended to '{chapter_name}'"

            new_content_full = "\n\n".join(paragraphs)
            chapter_path.write_text(new_content_full, encoding="utf-8")

            return OperationStatus(
                success=True,
                message=result_message,
                details={
                    "document_name": document_name,
                    "chapter_name": chapter_name,
                    "new_paragraph_index": new_index,
                    "position": position,
                },
            )

        except Exception as e:
            return OperationStatus(
                success=False,
                message=f"Error adding paragraph: {e}",
            )

    @mcp_server.tool()
    @log_mcp_call
    @auto_snapshot("replace_paragraph")
    def replace_paragraph(
        document_name: str,
        chapter_name: str,
        paragraph_index: int,
        new_content: str,
    ) -> OperationStatus:
        """Replace the content of an existing paragraph.

        Parameters:
            document_name (str): Name of the document directory
            chapter_name (str): Filename of the chapter (must end with .md)
            paragraph_index (int): Zero-indexed position of the paragraph to replace
            new_content (str): New content to replace the existing paragraph

        Returns:
            OperationStatus: Result with success status
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
            current_content = chapter_path.read_text(encoding="utf-8")
            paragraphs = _split_into_paragraphs(current_content)

            if paragraph_index >= len(paragraphs):
                return OperationStatus(
                    success=False,
                    message=f"Paragraph index {paragraph_index} is out of bounds. Chapter has {len(paragraphs)} paragraphs.",
                )

            paragraphs[paragraph_index] = new_content.strip()

            new_content_full = "\n\n".join(paragraphs)
            chapter_path.write_text(new_content_full, encoding="utf-8")

            return OperationStatus(
                success=True,
                message=f"Paragraph {paragraph_index} replaced in '{chapter_name}'",
                details={
                    "document_name": document_name,
                    "chapter_name": chapter_name,
                    "paragraph_index": paragraph_index,
                },
            )

        except Exception as e:
            return OperationStatus(
                success=False,
                message=f"Error replacing paragraph: {e}",
            )

    @mcp_server.tool()
    @log_mcp_call
    @auto_snapshot("delete_paragraph")
    def delete_paragraph(
        document_name: str,
        chapter_name: str,
        paragraph_index: int,
    ) -> OperationStatus:
        """Delete a paragraph from a chapter.

        Parameters:
            document_name (str): Name of the document directory
            chapter_name (str): Filename of the chapter (must end with .md)
            paragraph_index (int): Zero-indexed position of the paragraph to delete

        Returns:
            OperationStatus: Result with deleted content for reference
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
            current_content = chapter_path.read_text(encoding="utf-8")
            paragraphs = _split_into_paragraphs(current_content)

            if paragraph_index >= len(paragraphs):
                return OperationStatus(
                    success=False,
                    message=f"Paragraph index {paragraph_index} is out of bounds. Chapter has {len(paragraphs)} paragraphs.",
                )

            deleted_content = paragraphs.pop(paragraph_index)

            new_content_full = "\n\n".join(paragraphs)
            chapter_path.write_text(new_content_full, encoding="utf-8")

            return OperationStatus(
                success=True,
                message=f"Paragraph {paragraph_index} deleted from '{chapter_name}'",
                details={
                    "document_name": document_name,
                    "chapter_name": chapter_name,
                    "deleted_paragraph_index": paragraph_index,
                    "deleted_content": deleted_content,
                },
            )

        except Exception as e:
            return OperationStatus(
                success=False,
                message=f"Error deleting paragraph: {e}",
            )

    @mcp_server.tool()
    @log_mcp_call
    @auto_snapshot("move_paragraph")
    def move_paragraph(
        document_name: str,
        chapter_name: str,
        source_index: int,
        destination: Literal["before", "after"],
        target_index: int | None = None,
    ) -> OperationStatus:
        """Move a paragraph to a new position within the chapter.

        Parameters:
            document_name (str): Name of the document directory
            chapter_name (str): Filename of the chapter (must end with .md)
            source_index (int): Zero-indexed position of the paragraph to move
            destination (str): "before" or "after" the target position
            target_index (int | None): Target position. If None with "after", moves to end.

        Returns:
            OperationStatus: Result with diff information

        Examples:
            Move paragraph 3 before paragraph 1:
            {"source_index": 3, "destination": "before", "target_index": 1}

            Move paragraph 0 after paragraph 2:
            {"source_index": 0, "destination": "after", "target_index": 2}

            Move paragraph 2 to end (after last):
            {"source_index": 2, "destination": "after", "target_index": null}
        """
        # Validate inputs
        is_valid_doc, doc_error = validate_document_name(document_name)
        if not is_valid_doc:
            return OperationStatus(success=False, message=doc_error)

        is_valid_chapter, chapter_error = validate_chapter_name(chapter_name)
        if not is_valid_chapter:
            return OperationStatus(success=False, message=chapter_error)

        is_valid_source, source_error = validate_paragraph_index(source_index)
        if not is_valid_source:
            return OperationStatus(success=False, message=f"source_index: {source_error}")

        if destination == "before":
            if target_index is None:
                return OperationStatus(
                    success=False,
                    message="target_index is required when destination is 'before'",
                )
            is_valid_target, target_error = validate_paragraph_index(target_index)
            if not is_valid_target:
                return OperationStatus(success=False, message=f"target_index: {target_error}")
            if source_index == target_index:
                return OperationStatus(
                    success=False,
                    message="Cannot move a paragraph before itself.",
                )
        elif destination == "after" and target_index is not None:
            is_valid_target, target_error = validate_paragraph_index(target_index)
            if not is_valid_target:
                return OperationStatus(success=False, message=f"target_index: {target_error}")
            if source_index == target_index:
                return OperationStatus(
                    success=False,
                    message="Cannot move a paragraph after itself.",
                )

        chapter_path = _get_chapter_path(document_name, chapter_name)
        if not chapter_path.is_file() or not _is_valid_chapter_filename(chapter_name):
            return OperationStatus(
                success=False,
                message=f"Chapter '{chapter_name}' not found in document '{document_name}'.",
            )

        try:
            original_content = chapter_path.read_text(encoding="utf-8")
            paragraphs = _split_into_paragraphs(original_content)
            total_paragraphs = len(paragraphs)

            # Validate source index
            if not (0 <= source_index < total_paragraphs):
                return OperationStatus(
                    success=False,
                    message=f"source_index {source_index} is out of bounds (0-{total_paragraphs - 1}).",
                )

            if destination == "before":
                # Validate target index
                if not (0 <= target_index < total_paragraphs):
                    return OperationStatus(
                        success=False,
                        message=f"target_index {target_index} is out of bounds (0-{total_paragraphs - 1}).",
                    )

                # Move the paragraph
                paragraph_to_move = paragraphs.pop(source_index)

                # Adjust target if source was before target
                adjusted_target = target_index
                if source_index < target_index:
                    adjusted_target -= 1

                paragraphs.insert(adjusted_target, paragraph_to_move)
                result_message = (
                    f"Paragraph {source_index} moved before paragraph {target_index} in '{chapter_name}'"
                )

            elif destination == "after":
                if target_index is None:
                    # Move to end (after last paragraph)
                    if source_index == total_paragraphs - 1:
                        return OperationStatus(
                            success=True,
                            message=f"Paragraph {source_index} is already at the end of '{chapter_name}'.",
                            details={
                                "changed": False,
                                "summary": "No changes made - paragraph already at end",
                            },
                        )
                    paragraph_to_move = paragraphs.pop(source_index)
                    paragraphs.append(paragraph_to_move)
                    result_message = f"Paragraph {source_index} moved to end of '{chapter_name}'"
                else:
                    # Move after specific target
                    if not (0 <= target_index < total_paragraphs):
                        return OperationStatus(
                            success=False,
                            message=f"target_index {target_index} is out of bounds (0-{total_paragraphs - 1}).",
                        )
                    paragraph_to_move = paragraphs.pop(source_index)

                    # Adjust target if source was before target
                    adjusted_target = target_index + 1  # Insert after target
                    if source_index < target_index:
                        adjusted_target -= 1

                    paragraphs.insert(adjusted_target, paragraph_to_move)
                    result_message = (
                        f"Paragraph {source_index} moved after paragraph {target_index} in '{chapter_name}'"
                    )

            else:
                return OperationStatus(
                    success=False,
                    message=f"Unknown destination: {destination}. Use 'before' or 'after'.",
                )

            final_content = "\n\n".join(paragraphs)
            chapter_path.write_text(final_content, encoding="utf-8")

            diff_info = _generate_content_diff(original_content, final_content, chapter_name)

            return OperationStatus(
                success=True,
                message=result_message,
                details=diff_info,
            )

        except Exception as e:
            return OperationStatus(
                success=False,
                message=f"Error moving paragraph: {e}",
            )
