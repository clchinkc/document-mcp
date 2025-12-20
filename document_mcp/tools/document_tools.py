"""Document Management Tools.

This module contains MCP tools for managing document collections:
- list_documents: List all available document collections
- read_summary: Read summary files with flexible scope (document, chapter, section)
- write_summary: Write or update summary files with flexible scope
- list_summaries: List all available summary files for a document
- create_document: Create new document directory
- delete_document: Delete entire document and all chapters
"""

import datetime
import shutil
from pathlib import Path

from mcp.server import FastMCP

from ..helpers import _get_chapter_metadata
from ..helpers import _get_document_path
from ..helpers import _get_ordered_chapter_files
from ..helpers import _get_summaries_path
from ..helpers import _get_summary_file_path
from ..logger_config import log_mcp_call
from ..models import DocumentInfo
from ..models import DocumentSummary
from ..models import OperationStatus
from ..utils.decorators import auto_snapshot
from ..utils.file_operations import DOCS_ROOT_PATH
from ..utils.validation import validate_document_name


def register_document_tools(mcp_server: FastMCP) -> None:
    """Register all document management tools with the MCP server."""

    @mcp_server.tool()
    @log_mcp_call
    def list_documents(include_chapters: bool = False) -> list[DocumentInfo]:
        """List all available document collections in the document management system.

        This tool retrieves metadata for all document directories, where each document
        is a collection of ordered Markdown chapter files (.md). Provides comprehensive
        information including chapter counts, word counts, and modification timestamps.

        Parameters:
            include_chapters (bool): Whether to include detailed chapter metadata (default: False).
                - False: Fast response with empty chapters list, suitable for document overview
                - True: Complete response including all chapter metadata, slower but comprehensive

        Returns:
            List[DocumentInfo]: A list of document metadata objects. Each DocumentInfo contains:
                - document_name (str): Directory name of the document
                - total_chapters (int): Number of chapter files in the document
                - total_word_count (int): Sum of words across all chapters
                - total_paragraph_count (int): Sum of paragraphs across all chapters
                - last_modified (datetime): Most recent modification time across all chapters
                - chapters (List[ChapterMetadata]): Ordered list of chapter metadata
                - has_summary (bool): Whether a summary file exists in summaries/document.md

            Returns empty list [] if no documents exist or documents directory is not found.

        Example Usage:
            ```json
            // Fast document overview (recommended for most use cases)
            {
                "name": "list_documents",
                "arguments": {}
            }

            // Complete document details with all chapter metadata
            {
                "name": "list_documents",
                "arguments": {"include_chapters": true}
            }
            ```

        Example Response:
            ```json
            [
                {
                    "document_name": "user_guide",
                    "total_chapters": 3,
                    "total_word_count": 1250,
                    "total_paragraph_count": 45,
                    "last_modified": "2024-01-15T10:30:00Z",
                    "chapters": [
                        {
                            "chapter_name": "01-introduction.md",
                            "word_count": 300,
                            "paragraph_count": 12,
                            "last_modified": "2024-01-15T10:30:00Z"
                        }
                    ],
                    "has_summary": true
                }
            ]
            ```
        """
        docs_info = []
        # Use runtime environment variable check for test compatibility
        import os

        docs_root_name = os.environ.get("DOCUMENT_ROOT_DIR", str(DOCS_ROOT_PATH))
        root_path = Path(docs_root_name)

        if not root_path.exists() or not root_path.is_dir():
            return []

        for doc_dir in root_path.iterdir():
            if doc_dir.is_dir():  # Each subdirectory is a potential document
                document_name = doc_dir.name
                ordered_chapter_files = _get_ordered_chapter_files(document_name)

                chapters_metadata_list = []
                doc_total_word_count = 0
                doc_total_paragraph_count = 0
                latest_mod_time = datetime.datetime.min.replace(
                    tzinfo=datetime.timezone.utc
                )  # Ensure timezone aware for comparison

                for chapter_file_path in ordered_chapter_files:
                    metadata = _get_chapter_metadata(document_name, chapter_file_path)
                    if metadata:
                        if include_chapters:
                            chapters_metadata_list.append(metadata)
                        doc_total_word_count += metadata.word_count
                        doc_total_paragraph_count += metadata.paragraph_count
                        # Ensure metadata.last_modified is offset-aware before comparison
                        current_mod_time_aware = metadata.last_modified
                        if current_mod_time_aware > latest_mod_time:
                            latest_mod_time = current_mod_time_aware

                # Handle case where document has no chapter files
                if not ordered_chapter_files:  # No chapter files at all
                    stat_dir = doc_dir.stat()
                    latest_mod_time = datetime.datetime.fromtimestamp(
                        stat_dir.st_mtime, tz=datetime.timezone.utc
                    )

                # Check for summary in new organized location
                new_summary_path = _get_summaries_path(document_name) / "document.md"
                has_summary_file = new_summary_path.is_file()

                docs_info.append(
                    DocumentInfo(
                        document_name=document_name,
                        total_chapters=len(ordered_chapter_files),  # Always show actual chapter count
                        total_word_count=doc_total_word_count,
                        total_paragraph_count=doc_total_paragraph_count,
                        last_modified=(
                            latest_mod_time
                            if latest_mod_time != datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)
                            else datetime.datetime.fromtimestamp(
                                doc_dir.stat().st_mtime, tz=datetime.timezone.utc
                            )
                        ),
                        chapters=chapters_metadata_list,  # Empty list when include_chapters=False
                        has_summary=has_summary_file,
                    )
                )
        return docs_info

    @mcp_server.tool()
    @log_mcp_call
    @auto_snapshot("create_document")
    def create_document(document_name: str) -> OperationStatus:
        r"""Create a new document collection as a directory in the document management system.

        This tool initializes a new document by creating a directory that will contain
        chapter files (.md). The document name must be valid for filesystem usage and
        will serve as the directory name for organizing chapters.

        Parameters:
            document_name (str): Name for the new document directory. Must be:
                - Non-empty string
                - ≤100 characters
                - Valid filesystem directory name
                - Cannot contain path separators (/ or \\)
                - Cannot start with a dot (.)
                - Cannot conflict with existing document names

        Returns:
            OperationStatus: Structured result object containing:
                - success (bool): True if document was created successfully, False otherwise
                - message (str): Human-readable description of the operation result
                - details (Dict[str, Any], optional): Additional context including:
                    - document_name (str): Name of the created document (on success)

        Example Usage:
            ```json
            {
                "name": "create_document",
                "arguments": {
                    "document_name": "user_manual"
                }
            }
            ```

        Example Success Response:
            ```json
            {
                "success": true,
                "message": "Document 'user_manual' created successfully.",
                "details": {
                    "document_name": "user_manual"
                }
            }
            ```

        Example Error Response:
            ```json
            {
                "success": false,
                "message": "Document 'user_manual' already exists.",
                "details": null
            }
            ```
        """
        # Validate input
        is_valid, error_msg = validate_document_name(document_name)
        if not is_valid:
            return OperationStatus(success=False, message=error_msg)

        doc_path = _get_document_path(document_name)

        if doc_path.exists():
            return OperationStatus(success=False, message=f"Document '{document_name}' already exists.")

        try:
            doc_path.mkdir(parents=True, exist_ok=False)
            return OperationStatus(
                success=True,
                message=f"Document '{document_name}' created successfully.",
                details={"document_name": document_name},
            )
        except Exception as e:
            return OperationStatus(success=False, message=f"Error creating document '{document_name}': {e}")

    @mcp_server.tool()
    @log_mcp_call
    @auto_snapshot("delete_document")
    def delete_document(document_name: str) -> OperationStatus:
        """Permanently deletes a document directory and all its chapter files.

        This tool removes an entire document collection including all chapter files
        and any associated metadata. This operation is irreversible and should be
        used with caution. All content within the document directory will be lost.

        Parameters:
            document_name (str): Name of the document directory to delete

        Returns:
            OperationStatus: Structured result object containing:
                - success (bool): True if document was deleted successfully, False otherwise
                - message (str): Human-readable description of the operation result
                - details (Dict[str, Any], optional): Additional context (currently None)

        Example Usage:
            ```json
            {
                "name": "delete_document",
                "arguments": {
                    "document_name": "old_manual"
                }
            }
            ```

        Example Success Response:
            ```json
            {
                "success": true,
                "message": "Document 'old_manual' and its contents deleted successfully.",
                "details": null
            }
            ```

        Example Error Response:
            ```json
            {
                "success": false,
                "message": "Document 'old_manual' not found.",
                "details": null
            }
            ```
        """
        doc_path = _get_document_path(document_name)
        if not doc_path.is_dir():
            return OperationStatus(success=False, message=f"Document '{document_name}' not found.")
        try:
            shutil.rmtree(doc_path)
            return OperationStatus(
                success=True,
                message=f"Document '{document_name}' and its contents deleted successfully.",
            )
        except Exception as e:
            return OperationStatus(success=False, message=f"Error deleting document '{document_name}': {e}")

    @mcp_server.tool()
    @log_mcp_call
    def read_summary(
        document_name: str, scope: str = "document", target_name: str | None = None
    ) -> DocumentSummary | None:
        """Read a summary file with flexible scope (document, chapter, or section).

        This tool reads summary content based on the specified scope. It supports:
        - Document summaries: Overall document overview
        - Chapter summaries: Summary of specific chapters
        - Section summaries: Summary of thematic sections

        Parameters:
            document_name (str): Name of the document directory
            scope (str): Scope of the summary ("document", "chapter", "section")
            target_name (str, optional): Required for chapter/section scope.
                For chapters: chapter filename (e.g., "01-intro.md")
                For sections: section name (e.g., "introduction")

        Returns:
            DocumentSummary | None: Summary object with content, scope and target_name,
            or None if summary doesn't exist

        Example Usage:
            # Read document summary
            read_summary("my_book", scope="document")

            # Read chapter summary
            read_summary("my_book", scope="chapter", target_name="01-intro.md")

            # Read section summary
            read_summary("my_book", scope="section", target_name="introduction")
        """
        if not validate_document_name(document_name):
            return None

        doc_path = _get_document_path(document_name)
        if not doc_path.is_dir():
            return None

        try:
            summary_file_path = _get_summary_file_path(document_name, scope, target_name)
            if not summary_file_path.exists():
                return None

            content = summary_file_path.read_text(encoding="utf-8")
            return DocumentSummary(
                document_name=document_name, content=content, scope=scope, target_name=target_name
            )
        except Exception:
            return None

    @mcp_server.tool()
    @log_mcp_call
    @auto_snapshot("write_summary")
    def write_summary(
        document_name: str, summary_content: str, scope: str = "document", target_name: str | None = None
    ) -> OperationStatus:
        """Write or update a summary file with flexible scope.

        Creates or updates summary files for documents, chapters, or sections.
        Automatically creates the summaries directory structure if needed.

        Parameters:
            document_name (str): Name of the document directory
            summary_content (str): Content to write to the summary file
            scope (str): Scope of the summary ("document", "chapter", "section")
            target_name (str, optional): Required for chapter/section scope.
                For chapters: chapter filename (e.g., "01-intro.md")
                For sections: section name (e.g., "introduction")

        Returns:
            OperationStatus: Result of the write operation

        Example Usage:
            # Write document summary
            write_summary("my_book", "This book covers...", scope="document")

            # Write chapter summary
            write_summary("my_book", "Chapter introduces...", scope="chapter", target_name="01-intro.md")
        """
        if not validate_document_name(document_name):
            return OperationStatus(success=False, message="Invalid document name")

        doc_path = _get_document_path(document_name)
        if not doc_path.is_dir():
            return OperationStatus(success=False, message=f"Document '{document_name}' not found")

        try:
            # Create summaries directory if it doesn't exist
            summaries_path = _get_summaries_path(document_name)
            summaries_path.mkdir(exist_ok=True)

            summary_file_path = _get_summary_file_path(document_name, scope, target_name)
            summary_file_path.write_text(summary_content, encoding="utf-8")

            scope_desc = f"{scope} summary"
            if target_name:
                scope_desc += f" for '{target_name}'"

            return OperationStatus(
                success=True, message=f"Successfully wrote {scope_desc} for document '{document_name}'"
            )
        except ValueError as e:
            return OperationStatus(success=False, message=str(e))
        except Exception as e:
            return OperationStatus(success=False, message=f"Error writing summary: {e}")

    @mcp_server.tool()
    @log_mcp_call
    def list_summaries(document_name: str) -> list[str]:
        """List all available summary files for a document.

        Returns a list of all summary files in the document's summaries directory,
        providing an overview of what summaries exist across all scopes.

        Parameters:
            document_name (str): Name of the document directory

        Returns:
            list[str]: List of summary filenames (e.g., ["document.md", "chapter-01-intro.md"])

        Example Usage:
            list_summaries("my_book")
        """
        if not validate_document_name(document_name):
            return []

        doc_path = _get_document_path(document_name)
        if not doc_path.is_dir():
            return []

        summaries_path = _get_summaries_path(document_name)
        if not summaries_path.is_dir():
            return []

        try:
            summary_files = [f.name for f in summaries_path.iterdir() if f.is_file() and f.suffix == ".md"]
            return sorted(summary_files)
        except Exception:
            return []


# Helper functions are now imported from centralized helpers module
