"""Safety and version control MCP tools for document management.

This module provides unified safety tools for snapshot management,
content status checking, and diff generation.
"""

from typing import Any

from ..batch import register_batchable_operation
from ..logger_config import ErrorCategory
from ..logger_config import log_mcp_call
from ..logger_config import log_structured_error
from ..utils.validation import validate_chapter_name as _validate_chapter_name
from ..utils.validation import validate_document_name as _validate_document_name


def register_safety_tools(mcp_server):
    """Register all safety-related tools with the MCP server."""

    @mcp_server.tool()
    @log_mcp_call
    @register_batchable_operation("manage_snapshots")
    def manage_snapshots(
        document_name: str,
        action: str,  # "create", "list", "restore"
        snapshot_id: str | None = None,
        message: str | None = None,
        auto_cleanup: bool = True,
    ) -> dict[str, Any]:
        """Unified snapshot management tool with action-based interface.

        This consolidated tool replaces snapshot_document, list_snapshots, and
        restore_snapshot with a single interface that supports all snapshot operations.
        Reduces tool count while maintaining full functionality.

        Parameters:
            document_name (str): Name of the document directory
            action (str): Operation to perform - "create", "list", or "restore"
            snapshot_id (Optional[str]): Snapshot ID for restore action (required for restore)
            message (Optional[str]): Message for create action (optional)
            auto_cleanup (bool): Auto-cleanup old snapshots for create action (default: True)

        Returns:
            Dict[str, Any]: Action-specific result data:
                - create: OperationStatus with snapshot_id
                - list: SnapshotsList with all snapshots
                - restore: OperationStatus with restoration details

        Example Usage:
            ```json
            {
                "name": "manage_snapshots",
                "arguments": {
                    "document_name": "user_guide",
                    "action": "create",
                    "message": "Before major revision"
                }
            }
            ```

            ```json
            {
                "name": "manage_snapshots",
                "arguments": {
                    "document_name": "user_guide",
                    "action": "list"
                }
            }
            ```

            ```json
            {
                "name": "manage_snapshots",
                "arguments": {
                    "document_name": "user_guide",
                    "action": "restore",
                    "snapshot_id": "20240115_103045_snapshot"
                }
            }
            ```
        """
        # Import here to avoid circular imports
        from ..doc_tool_server import list_snapshots
        from ..doc_tool_server import restore_snapshot
        from ..doc_tool_server import snapshot_document

        # Validate action parameter
        valid_actions = ["create", "list", "restore"]
        if action not in valid_actions:
            return {
                "success": False,
                "message": f"Invalid action '{action}'. Must be one of: {', '.join(valid_actions)}",
                "action": action,
                "valid_actions": valid_actions,
            }

        # Validate document name
        is_valid, error_msg = _validate_document_name(document_name)
        if not is_valid:
            return {
                "success": False,
                "message": f"Invalid document name: {error_msg}",
                "action": action,
            }

        try:
            if action == "create":
                # Create snapshot using existing functionality
                result = snapshot_document(document_name, message, auto_cleanup)
                return {
                    "success": result.success,
                    "message": result.message,
                    "action": "create",
                    "snapshot_id": result.details.get("snapshot_id")
                    if result.details
                    else None,
                    "details": result.details,
                }

            elif action == "list":
                # List snapshots using existing functionality
                result = list_snapshots(document_name)
                return {
                    "success": True,
                    "message": f"Retrieved {result.total_snapshots} snapshots for document '{document_name}'",
                    "action": "list",
                    "document_name": result.document_name,
                    "snapshots": [
                        snapshot.model_dump() for snapshot in result.snapshots
                    ],
                    "total_snapshots": result.total_snapshots,
                    "total_size_bytes": result.total_size_bytes,
                }

            elif action == "restore":
                # Validate snapshot_id is provided
                if not snapshot_id:
                    return {
                        "success": False,
                        "message": "snapshot_id is required for restore action",
                        "action": "restore",
                    }

                # Restore snapshot using existing functionality
                result = restore_snapshot(document_name, snapshot_id)
                return {
                    "success": result.success,
                    "message": result.message,
                    "action": "restore",
                    "document_name": document_name,
                    "snapshot_id": snapshot_id,
                    "details": result.details,
                }

        except Exception as e:
            log_structured_error(
                ErrorCategory.OPERATION_FAILED,
                f"Failed to {action} snapshot for document '{document_name}': {e}",
                {
                    "operation": "manage_snapshots",
                    "action": action,
                    "document_name": document_name,
                },
            )
            return {
                "success": False,
                "message": f"Failed to {action} snapshot: {str(e)}",
                "action": action,
                "error": str(e),
            }

    @mcp_server.tool()
    @log_mcp_call
    @register_batchable_operation("check_content_status")
    def check_content_status(
        document_name: str,
        chapter_name: str | None = None,
        include_history: bool = False,
        time_window: str = "24h",
        last_known_modified: str | None = None,
    ) -> dict[str, Any]:
        """Unified content status and modification history checker.

        This consolidated tool combines check_content_freshness and get_modification_history
        into a single interface that provides comprehensive content status information.
        Reduces tool count while offering enhanced functionality.

        Parameters:
            document_name (str): Name of the document directory
            chapter_name (Optional[str]): Specific chapter to check (if None, checks entire document)
            include_history (bool): Whether to include modification history (default: False)
            time_window (str): Time window for history if included ("1h", "24h", "7d", "30d", "all")
            last_known_modified (Optional[str]): ISO timestamp for freshness check

        Returns:
            Dict[str, Any]: Comprehensive content status including:
                - freshness: ContentFreshnessStatus data
                - history: ModificationHistory data (if include_history=True)
                - summary: Human-readable status summary

        Example Usage:
            ```json
            {
                "name": "check_content_status",
                "arguments": {
                    "document_name": "user_guide",
                    "chapter_name": "01-intro.md",
                    "include_history": true,
                    "time_window": "7d",
                    "last_known_modified": "2024-01-15T10:30:00Z"
                }
            }
            ```
        """
        # Import here to avoid circular imports
        from ..doc_tool_server import check_content_freshness
        from ..doc_tool_server import get_modification_history

        # Validate document name
        is_valid, error_msg = _validate_document_name(document_name)
        if not is_valid:
            return {
                "success": False,
                "message": f"Invalid document name: {error_msg}",
                "operation": "check_content_status",
            }

        # Validate chapter name if provided
        if chapter_name:
            is_valid, error_msg = _validate_chapter_name(chapter_name)
            if not is_valid:
                return {
                    "success": False,
                    "message": f"Invalid chapter name: {error_msg}",
                    "operation": "check_content_status",
                }

        try:
            # Get freshness status
            freshness_result = check_content_freshness(
                document_name, chapter_name, last_known_modified
            )

            # Get modification history if requested
            history_result = None
            if include_history:
                history_result = get_modification_history(
                    document_name, chapter_name, time_window
                )

            # Generate summary
            scope = f"chapter '{chapter_name}'" if chapter_name else "document"
            freshness_status = "fresh" if freshness_result.is_fresh else "stale"

            summary_parts = [
                f"Content status for {scope} in '{document_name}': {freshness_status}"
            ]
            if history_result:
                summary_parts.append(
                    f"Found {history_result.total_modifications} modifications in {time_window}"
                )

            return {
                "success": True,
                "message": ". ".join(summary_parts),
                "operation": "check_content_status",
                "document_name": document_name,
                "chapter_name": chapter_name,
                "freshness": {
                    "is_fresh": freshness_result.is_fresh,
                    "last_modified": freshness_result.last_modified.isoformat()
                    if freshness_result.last_modified
                    else None,
                    "safety_status": freshness_result.safety_status,
                    "message": freshness_result.message,
                    "recommendations": freshness_result.recommendations,
                },
                "history": {
                    "total_modifications": history_result.total_modifications,
                    "time_window": history_result.time_window,
                    "entries": [entry.model_dump() for entry in history_result.entries],
                }
                if history_result
                else None,
                "summary": ". ".join(summary_parts),
            }

        except Exception as e:
            log_structured_error(
                ErrorCategory.OPERATION_FAILED,
                f"Failed to check content status for '{document_name}': {e}",
                {
                    "operation": "check_content_status",
                    "document_name": document_name,
                    "chapter_name": chapter_name,
                },
            )
            return {
                "success": False,
                "message": f"Failed to check content status: {str(e)}",
                "operation": "check_content_status",
                "error": str(e),
            }

    @mcp_server.tool()
    @log_mcp_call
    @register_batchable_operation("diff_content")
    def diff_content(
        document_name: str,
        source_type: str = "snapshot",  # "snapshot", "current", "file"
        source_id: str | None = None,
        target_type: str = "current",  # "snapshot", "current", "file"
        target_id: str | None = None,
        output_format: str = "unified",  # "unified", "context", "summary"
        chapter_name: str | None = None,
    ) -> dict[str, Any]:
        """Unified content comparison and diff generation tool.

        This consolidated tool replaces diff_snapshots and provides enhanced diff
        capabilities between any combination of snapshots, current content, and files.
        Supports multiple output formats and flexible source/target specification.

        Parameters:
            document_name (str): Name of the document directory
            source_type (str): Type of source content - "snapshot", "current", or "file"
            source_id (Optional[str]): ID/name for source (snapshot_id for snapshots, file path for files)
            target_type (str): Type of target content - "snapshot", "current", or "file"
            target_id (Optional[str]): ID/name for target (snapshot_id for snapshots, file path for files)
            output_format (str): Diff output format - "unified", "context", or "summary"
            chapter_name (Optional[str]): Specific chapter to compare (if None, compares full documents)

        Returns:
            Dict[str, Any]: Comprehensive diff results including:
                - diff_text: Generated diff in requested format
                - summary: Human-readable change summary
                - statistics: Change statistics (lines added/removed/modified)
                - metadata: Source and target information

        Example Usage:
            ```json
            {
                "name": "diff_content",
                "arguments": {
                    "document_name": "user_guide",
                    "source_type": "snapshot",
                    "source_id": "20240115_103045_snapshot",
                    "target_type": "current",
                    "output_format": "unified",
                    "chapter_name": "01-intro.md"
                }
            }
            ```
        """
        # Import here to avoid circular imports
        from ..doc_tool_server import _validate_chapter_name
        from ..doc_tool_server import _validate_document_name
        from ..doc_tool_server import diff_snapshots

        # Validate document name
        is_valid, error_msg = _validate_document_name(document_name)
        if not is_valid:
            return {
                "success": False,
                "message": f"Invalid document name: {error_msg}",
                "operation": "diff_content",
            }

        # Validate chapter name if provided
        if chapter_name:
            is_valid, error_msg = _validate_chapter_name(chapter_name)
            if not is_valid:
                return {
                    "success": False,
                    "message": f"Invalid chapter name: {error_msg}",
                    "operation": "diff_content",
                }

        # Validate source and target types
        valid_types = ["snapshot", "current", "file"]
        if source_type not in valid_types:
            return {
                "success": False,
                "message": f"Invalid source_type '{source_type}'. Must be one of: {', '.join(valid_types)}",
                "operation": "diff_content",
            }

        if target_type not in valid_types:
            return {
                "success": False,
                "message": f"Invalid target_type '{target_type}'. Must be one of: {', '.join(valid_types)}",
                "operation": "diff_content",
            }

        # Validate output format
        valid_formats = ["unified", "context", "summary"]
        if output_format not in valid_formats:
            return {
                "success": False,
                "message": f"Invalid output_format '{output_format}'. Must be one of: {', '.join(valid_formats)}",
                "operation": "diff_content",
            }

        # Validate required IDs
        if source_type in ["snapshot", "file"] and not source_id:
            return {
                "success": False,
                "message": f"source_id is required for source_type '{source_type}'",
                "operation": "diff_content",
            }

        if target_type in ["snapshot", "file"] and not target_id:
            return {
                "success": False,
                "message": f"target_id is required for target_type '{target_type}'",
                "operation": "diff_content",
            }

        try:
            # For now, use the existing diff_snapshots function as a starting point
            # This is a simplified implementation that works with snapshot comparisons
            if source_type == "snapshot" and target_type == "current":
                # Use existing diff_snapshots functionality
                result = diff_snapshots(
                    document_name=document_name,
                    snapshot_id_1=source_id,
                    snapshot_id_2=None,  # None means compare with current
                    output_format=output_format,
                    chapter_name=chapter_name,
                )
                return result.model_dump()

            elif source_type == "snapshot" and target_type == "snapshot":
                # Compare two snapshots
                result = diff_snapshots(
                    document_name=document_name,
                    snapshot_id_1=source_id,
                    snapshot_id_2=target_id,
                    output_format=output_format,
                    chapter_name=chapter_name,
                )
                return result.model_dump()

            else:
                # For other combinations, return a not implemented message
                return {
                    "success": False,
                    "message": f"Diff between {source_type} and {target_type} not yet implemented",
                    "operation": "diff_content",
                    "source_type": source_type,
                    "target_type": target_type,
                    "note": "Currently only snapshot-to-current and snapshot-to-snapshot comparisons are supported",
                }

        except Exception as e:
            log_structured_error(
                ErrorCategory.OPERATION_FAILED,
                f"Failed to diff content for '{document_name}': {e}",
                {
                    "operation": "diff_content",
                    "document_name": document_name,
                    "chapter_name": chapter_name,
                },
            )
            return {
                "success": False,
                "message": f"Failed to diff content: {str(e)}",
                "operation": "diff_content",
                "error": str(e),
            }
