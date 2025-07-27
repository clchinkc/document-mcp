"""Document Management Tools.

This module contains MCP tools for managing document collections:
- list_documents: List all available document collections
- read_document_summary: Read document summary file (_SUMMARY.md)
- create_document: Create new document directory
- delete_document: Delete entire document and all chapters
"""

import datetime
import shutil
from pathlib import Path

from mcp.server import FastMCP

from ..batch import register_batchable_operation
from ..helpers import DOCUMENT_SUMMARY_FILE
from ..helpers import _get_chapter_metadata
from ..helpers import _get_document_path
from ..helpers import _get_ordered_chapter_files
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
    def list_documents() -> list[DocumentInfo]:
        """List all available document collections in the document management system.

        This tool retrieves metadata for all document directories, where each document
        is a collection of ordered Markdown chapter files (.md). Provides comprehensive
        information including chapter counts, word counts, and modification timestamps.

        Parameters:
            None

        Returns:
            List[DocumentInfo]: A list of document metadata objects. Each DocumentInfo contains:
                - document_name (str): Directory name of the document
                - total_chapters (int): Number of chapter files in the document
                - total_word_count (int): Sum of words across all chapters
                - total_paragraph_count (int): Sum of paragraphs across all chapters
                - last_modified (datetime): Most recent modification time across all chapters
                - chapters (List[ChapterMetadata]): Ordered list of chapter metadata
                - has_summary (bool): Whether a _SUMMARY.md file exists

            Returns empty list [] if no documents exist or documents directory is not found.

        Example Usage:
            ```json
            {
                "name": "list_documents",
                "arguments": {}
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
                        chapters_metadata_list.append(metadata)
                        doc_total_word_count += metadata.word_count
                        doc_total_paragraph_count += metadata.paragraph_count
                        # Ensure metadata.last_modified is offset-aware before comparison
                        current_mod_time_aware = metadata.last_modified
                        if current_mod_time_aware > latest_mod_time:
                            latest_mod_time = current_mod_time_aware

                if (
                    not chapters_metadata_list
                ):  # If no valid chapters, maybe don't list as a doc or list with 0s
                    # Or, use directory's mtime if no chapters. For now, only list if chapters exist.
                    # Or list if it's an empty initialized doc.
                    # Let's list it even if empty, using the folder's mtime.
                    if not ordered_chapter_files:  # No chapter files at all
                        stat_dir = doc_dir.stat()
                        latest_mod_time = datetime.datetime.fromtimestamp(
                            stat_dir.st_mtime, tz=datetime.timezone.utc
                        )

                summary_file_path = doc_dir / DOCUMENT_SUMMARY_FILE
                has_summary_file = summary_file_path.is_file()

                docs_info.append(
                    DocumentInfo(
                        document_name=document_name,
                        total_chapters=len(chapters_metadata_list),
                        total_word_count=doc_total_word_count,
                        total_paragraph_count=doc_total_paragraph_count,
                        last_modified=(
                            latest_mod_time
                            if latest_mod_time != datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)
                            else datetime.datetime.fromtimestamp(
                                doc_dir.stat().st_mtime, tz=datetime.timezone.utc
                            )
                        ),
                        chapters=chapters_metadata_list,
                        has_summary=has_summary_file,
                    )
                )
        return docs_info

    @mcp_server.tool()
    @log_mcp_call
    def read_document_summary(document_name: str) -> DocumentSummary | None:
        r"""Retrieve the content of a document's summary file (_SUMMARY.md).

        This tool reads the special _SUMMARY.md file that can be used to provide
        an overview or table of contents for a document collection. The summary
        file is optional and may not exist for all documents.

        Parameters:
            document_name (str): Name of the document directory to read summary from

        Returns:
            Optional[DocumentSummary]: A DocumentSummary object containing the document name
            and summary content if it exists, None if the summary file doesn't exist or
            cannot be read.

            Returns None if:
            - Document directory doesn't exist
            - _SUMMARY.md file doesn't exist in the document

        Example Usage:
            ```json
            {
                "name": "read_document_summary",
                "arguments": {
                    "document_name": "user_guide"
                }
            }
            ```

        Example Success Response:
            ```json
            {
                "document_name": "user_guide",
                "content": "# Document Summary\n\n- Chapter 1\n- Chapter 2"
            }
            ```

        Example Not Found Response:
            ```json
            null
            ```
        """
        # Validate document name (optional, but good practice if it can be invalid)
        is_valid_doc, doc_error = validate_document_name(document_name)
        if not is_valid_doc:
            from ..logger_config import ErrorCategory
            from ..logger_config import log_structured_error

            log_structured_error(
                category=ErrorCategory.WARNING,
                message="Invalid document name provided",
                context={"document_name": document_name, "validation_error": doc_error},
                operation="read_document_summary",
            )
            # Depending on desired strictness, could return None or raise error
            return None  # For now, let's be lenient if the path check below handles it

        doc_path = _get_document_path(document_name)
        if not doc_path.is_dir():
            from ..logger_config import ErrorCategory
            from ..logger_config import log_structured_error

            log_structured_error(
                category=ErrorCategory.INFO,
                message="Document not found",
                context={
                    "document_name": document_name,
                    "attempted_path": str(doc_path),
                },
                operation="read_document_summary",
            )
            return None

        summary_file_path = doc_path / DOCUMENT_SUMMARY_FILE
        if not summary_file_path.is_file():
            from ..logger_config import ErrorCategory
            from ..logger_config import log_structured_error

            log_structured_error(
                category=ErrorCategory.INFO,
                message="Summary file not found in document",
                context={
                    "document_name": document_name,
                    "summary_file_name": DOCUMENT_SUMMARY_FILE,
                    "summary_file_path": str(summary_file_path),
                },
                operation="read_document_summary",
            )
            return None

        try:
            summary_content = summary_file_path.read_text(encoding="utf-8")
            return DocumentSummary(document_name=document_name, content=summary_content)
        except Exception as e:
            from ..logger_config import ErrorCategory
            from ..logger_config import log_structured_error

            log_structured_error(
                category=ErrorCategory.ERROR,
                message="Failed to read summary file",
                exception=e,
                context={
                    "document_name": document_name,
                    "summary_file_path": str(summary_file_path),
                },
                operation="read_document_summary",
            )
            return None

    @mcp_server.tool()
    @register_batchable_operation("create_document")
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
        # Debug: Track if create_document tool is called
        print(f"[CREATE_DOC_DEBUG] create_document tool called with name: '{document_name}'")
        
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


# Helper functions are now imported from centralized helpers module
