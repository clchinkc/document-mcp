"""Context Management Tools.

This module contains MCP tools for managing document context using OneContext patterns:
- store_memory: Store key-value memories with tags and metadata
- recall_memory: Retrieve stored memories with update tracking
- export_context: Export all context to json/yaml/markdown format
- import_context: Import context from external sources with validation
"""

import datetime
from pathlib import Path
from typing import Any
from typing import Optional

from mcp.server import FastMCP

from ..logger_config import log_mcp_call
from ..models import ExportStatus
from ..models import ImportStatus
from ..models import MemoryEntry
from ..models import OperationStatus
from ..utils.context_manager import delete_memory
from ..utils.context_manager import export_context as export_context_impl
from ..utils.context_manager import get_context_path
from ..utils.context_manager import import_context as import_context_impl
from ..utils.context_manager import list_memories
from ..utils.context_manager import recall_memory as recall_memory_impl
from ..utils.context_manager import store_memory as store_memory_impl
from ..utils.file_operations import get_document_path
from ..utils.validation import validate_document_name


def register_context_tools(mcp_server: FastMCP) -> None:
    """Register all context management tools with the MCP server."""

    @mcp_server.tool()
    @log_mcp_call
    def store_memory(
        document_name: str,
        key: str,
        value: Any,
        tags: Optional[list[str]] = None,
        expires_days: Optional[int] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> MemoryEntry:
        """Store a key-value memory entry in document context.

        Stores memories with automatic session initialization. Memories are organized
        by document and can be tagged for easy retrieval. Supports optional expiration
        and custom metadata.

        This tool implements OneContext-inspired context management for persistent
        session state, goals, blockers, and decisions.

        Parameters:
            document_name (str): Name of the document to store memory in.
                Must exist in the document management system.
            key (str): Unique identifier for this memory entry.
                Used as primary lookup key. Examples: "current_goal", "blocker_type_a"
            value (Any): The value to store (string, JSON object, list, etc).
                Automatically serialized to JSON format.
            tags (list[str]): Optional tags for organizing and filtering memories.
                Useful for grouping related entries. Examples: ["urgent", "blocking"]
            expires_days (int): Optional number of days before memory expires.
                If specified, memory will be marked as expired but not deleted.
            metadata (dict): Optional custom metadata dict attached to the entry.
                Useful for storing context-specific data alongside the value.

        Returns:
            MemoryEntry: The stored memory entry with:
                - key: The memory key
                - value: The stored value
                - stored_at: ISO timestamp of when it was stored
                - retrieved_at: None (first retrieval not yet tracked)
                - tags: The provided tags
                - expires: Expiration timestamp if specified
                - metadata: The provided metadata

        Example Usage:
            ```json
            // Store a simple goal
            {
                "name": "store_memory",
                "arguments": {
                    "document_name": "novel_draft",
                    "key": "current_chapter",
                    "value": "Chapter 5: The Discovery",
                    "tags": ["progress", "current"]
                }
            }

            // Store a complex decision with expiration
            {
                "name": "store_memory",
                "arguments": {
                    "document_name": "novel_draft",
                    "key": "character_arc_decision",
                    "value": {
                        "character": "Marcus",
                        "decision": "Change backstory to military",
                        "reasons": ["Better motivation", "Stronger character"]
                    },
                    "tags": ["decision", "major"],
                    "expires_days": 30,
                    "metadata": {"priority": "high", "reviewed": false}
                }
            }
            ```

        Raises:
            ValueError: If document_name is invalid or value cannot be serialized
        """
        validate_document_name(document_name)

        # Check document exists
        doc_path = get_document_path(document_name)
        if not doc_path.exists():
            raise ValueError(f"Document not found: {document_name}")

        expires = None
        if expires_days:
            expires = datetime.datetime.utcnow() + datetime.timedelta(days=expires_days)

        return store_memory_impl(
            document_name=document_name,
            key=key,
            value=value,
            tags=tags,
            expires=expires,
            metadata=metadata,
        )

    @mcp_server.tool()
    @log_mcp_call
    def recall_memory(
        document_name: str,
        key: str,
        pattern: Optional[str] = None,
    ) -> Optional[MemoryEntry]:
        """Retrieve a stored memory entry from document context.

        Retrieves a memory by key and updates the retrieved_at timestamp.
        Supports pattern matching for searching across multiple keys.

        This tool tracks when memories are accessed, useful for understanding
        which context is actively being used.

        Parameters:
            document_name (str): Name of the document to recall memory from.
            key (str): The memory key to retrieve.
                Examples: "current_goal", "main_character_notes"
            pattern (str): Optional pattern for fuzzy matching.
                Currently reserved for future use.

        Returns:
            MemoryEntry | None: The retrieved memory entry if found with:
                - All stored data (key, value, tags, etc)
                - retrieved_at: Updated to current time
                Returns None if memory key not found.

        Example Usage:
            ```json
            // Retrieve a specific memory
            {
                "name": "recall_memory",
                "arguments": {
                    "document_name": "novel_draft",
                    "key": "current_chapter"
                }
            }

            // Retrieve and check structure
            {
                "name": "recall_memory",
                "arguments": {
                    "document_name": "novel_draft",
                    "key": "character_arc_decision"
                }
            }
            ```

        Example Response:
            ```json
            {
                "key": "current_chapter",
                "value": "Chapter 5: The Discovery",
                "stored_at": "2026-02-25T14:30:00.000000",
                "retrieved_at": "2026-02-25T15:45:00.000000",
                "tags": ["progress", "current"],
                "expires": null,
                "metadata": {}
            }
            ```
        """
        validate_document_name(document_name)

        doc_path = get_document_path(document_name)
        if not doc_path.exists():
            raise ValueError(f"Document not found: {document_name}")

        return recall_memory_impl(
            document_name=document_name,
            key=key,
            pattern=pattern,
        )

    @mcp_server.tool()
    @log_mcp_call
    def list_memories(
        document_name: str,
        tags: Optional[list[str]] = None,
    ) -> list[MemoryEntry]:
        """List all memories for a document, optionally filtered by tags.

        Retrieves all memory entries without updating their retrieved_at timestamps.
        Useful for reviewing what context has been stored.

        Parameters:
            document_name (str): Name of the document.
            tags (list[str]): Optional list of tags to filter by.
                Returns memories that have ANY of the specified tags.

        Returns:
            list[MemoryEntry]: List of matching memory entries.
                Empty list if no memories exist or no tags match.

        Example Usage:
            ```json
            // List all memories for a document
            {
                "name": "list_memories",
                "arguments": {
                    "document_name": "novel_draft"
                }
            }

            // List memories with specific tag
            {
                "name": "list_memories",
                "arguments": {
                    "document_name": "novel_draft",
                    "tags": ["decision"]
                }
            }
            ```
        """
        validate_document_name(document_name)

        doc_path = get_document_path(document_name)
        if not doc_path.exists():
            raise ValueError(f"Document not found: {document_name}")

        return list_memories(
            document_name=document_name,
            tags=tags,
        )

    @mcp_server.tool()
    @log_mcp_call
    def delete_memory(
        document_name: str,
        key: str,
    ) -> OperationStatus:
        """Delete a memory entry from document context.

        Permanently removes a memory entry. Cannot be undone.

        Parameters:
            document_name (str): Name of the document.
            key (str): The memory key to delete.

        Returns:
            OperationStatus: Success/failure status.

        Example Usage:
            ```json
            {
                "name": "delete_memory",
                "arguments": {
                    "document_name": "novel_draft",
                    "key": "outdated_note"
                }
            }
            ```
        """
        validate_document_name(document_name)

        doc_path = get_document_path(document_name)
        if not doc_path.exists():
            raise ValueError(f"Document not found: {document_name}")

        success = delete_memory(document_name=document_name, key=key)

        return OperationStatus(
            success=success,
            message=(
                f"Memory '{key}' deleted successfully"
                if success
                else f"Memory '{key}' not found"
            ),
        )

    @mcp_server.tool()
    @log_mcp_call
    def export_context(
        document_name: str,
        format_type: str = "json",
        export_filename: Optional[str] = None,
    ) -> ExportStatus:
        """Export document context to a file.

        Exports all memories, session metadata, goals, decisions, and blockers
        to a portable format. Useful for backing up context or sharing between
        documents.

        Supports three formats:
        - json: Machine-readable JSON with full structure
        - yaml: Human-readable YAML format
        - markdown: Markdown document for review

        Parameters:
            document_name (str): Name of the document to export context from.
            format_type (str): Export format - 'json', 'yaml', or 'markdown'.
                Default: 'json'
            export_filename (str): Optional custom filename.
                If not provided, uses format-specific default.
                Examples: 'context_backup.json', 'session_export.yaml'

        Returns:
            ExportStatus: Details about the export including:
                - success: Whether export completed successfully
                - exported_file_path: Path to the exported file
                - file_size: Size in bytes
                - entry_count: Number of memories exported
                - format_used: Actual format used
                - message: Human-readable status message

        Example Usage:
            ```json
            // Export to JSON (default)
            {
                "name": "export_context",
                "arguments": {
                    "document_name": "novel_draft"
                }
            }

            // Export to YAML with custom filename
            {
                "name": "export_context",
                "arguments": {
                    "document_name": "novel_draft",
                    "format_type": "yaml",
                    "export_filename": "context_backup_2026.yaml"
                }
            }

            // Export to Markdown for review
            {
                "name": "export_context",
                "arguments": {
                    "document_name": "novel_draft",
                    "format_type": "markdown",
                    "export_filename": "context_review.md"
                }
            }
            ```

        Example Response (JSON):
            ```json
            {
                "success": true,
                "exported_file_path": "/path/to/.context/exports/context.json",
                "file_size": 2048,
                "entry_count": 5,
                "format_used": "json",
                "message": "Context exported successfully to /path/to/.context/exports/context.json",
                "timestamp": "2026-02-25T14:30:00.000000"
            }
            ```
        """
        validate_document_name(document_name)

        doc_path = get_document_path(document_name)
        if not doc_path.exists():
            raise ValueError(f"Document not found: {document_name}")

        if format_type not in ["json", "yaml", "markdown"]:
            raise ValueError(f"Unsupported format: {format_type}")

        # Generate default filename if not provided
        if not export_filename:
            suffix = {"json": ".json", "yaml": ".yaml", "markdown": ".md"}.get(
                format_type, ".json"
            )
            export_filename = f"context{suffix}"

        context_path = get_context_path(document_name)
        export_path = context_path / "exports" / export_filename

        return export_context_impl(
            document_name=document_name,
            export_path=export_path,
            format_type=format_type,
        )

    @mcp_server.tool()
    @log_mcp_call
    def import_context(
        document_name: str,
        context_file: str,
        merge: bool = False,
    ) -> ImportStatus:
        """Import context from an exported context file.

        Imports memories, session metadata, and other context from a previously
        exported context file. Can merge with existing context or replace it entirely.

        Validates file format and detects conflicts before importing.

        Parameters:
            document_name (str): Name of the document to import context into.
            context_file (str): Path to the context file to import.
                Must be a valid json/yaml file with proper structure.
            merge (bool): If True, merge imported context with existing.
                If False (default), skip entries that already exist (safer).

        Returns:
            ImportStatus: Details about the import including:
                - success: Whether all imports succeeded
                - entries_imported: Number of memories imported
                - conflicts_detected: Number of conflicts found
                - conflict_details: List of specific conflicts
                - merge_mode_used: Whether merge mode was applied
                - message: Human-readable status message

        Example Usage:
            ```json
            // Import and merge with existing context
            {
                "name": "import_context",
                "arguments": {
                    "document_name": "novel_draft",
                    "context_file": "/path/to/backup.json",
                    "merge": true
                }
            }

            // Import from another document's export
            {
                "name": "import_context",
                "arguments": {
                    "document_name": "novel_draft_v2",
                    "context_file": ".context/exports/context.yaml",
                    "merge": false
                }
            }
            ```

        Example Response:
            ```json
            {
                "success": true,
                "entries_imported": 5,
                "conflicts_detected": 0,
                "conflict_details": [],
                "merge_mode_used": true,
                "message": "Imported 5 memories",
                "timestamp": "2026-02-25T14:30:00.000000"
            }
            ```
        """
        validate_document_name(document_name)

        doc_path = get_document_path(document_name)
        if not doc_path.exists():
            raise ValueError(f"Document not found: {document_name}")

        context_file_path = Path(context_file)
        if not context_file_path.exists():
            raise ValueError(f"Context file not found: {context_file}")

        return import_context_impl(
            document_name=document_name,
            context_file=context_file_path,
            merge=merge,
        )
