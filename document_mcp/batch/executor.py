"""Simple batch execution engine for document operations.

This module provides sequential execution of batch operations with basic
conflict detection for document management workflows.
"""

import time

from .models import BatchApplyResult
from .models import BatchOperation
from .models import ConflictInfo
from .registry import execute_batch_operation


class BatchExecutor:
    """Simple executor for batch operations with basic conflict detection."""

    def __init__(self):
        """Initialize the batch executor with operation categorization."""
        # Operations that modify content and cannot run on the same resource
        self._write_operations = {
            "create_document",
            "delete_document",
            "create_chapter",
            "delete_chapter",
            "write_chapter_content",
            "replace_paragraph",
            "insert_paragraph_before",
            "insert_paragraph_after",
            "delete_paragraph",
            "move_paragraph_before",
            "move_paragraph_to_end",
            "append_paragraph_to_chapter",
            "replace_text",
        }
        # Operations that only read content
        self._read_operations = {
            "read_content",
            "find_text",
            "get_statistics",
            "list_documents",
            "list_chapters",
            "read_document_summary",
        }

    def execute_batch(
        self,
        operations: list[BatchOperation],
        continue_on_error: bool = False,
        snapshot_id: str = None,
    ) -> BatchApplyResult:
        """Execute a batch of operations sequentially with conflict detection.

        Args:
            operations: List of operations to execute
            continue_on_error: If True, continue executing operations after failures
            snapshot_id: Optional snapshot ID for atomic rollback on failure

        Returns:
            BatchApplyResult: Results of batch execution
        """
        start_time = time.time()

        # Check for conflicts first - only abort on error-level conflicts
        conflicts = self._detect_conflicts(operations)
        error_conflicts = [c for c in conflicts if c.severity == "error"]
        if error_conflicts:
            return BatchApplyResult(
                success=False,
                total_operations=len(operations),
                successful_operations=0,
                failed_operations=len(operations),
                execution_time_ms=(time.time() - start_time) * 1000,
                operation_results=[],
                summary="Batch execution aborted due to conflicts",
                error_summary=f"Detected {len(error_conflicts)} error-level conflicts in batch operations",
            )

        # Execute operations sequentially
        results = []
        successful_operations = 0
        failed_operations = 0
        rollback_performed = False

        for operation in sorted(operations, key=lambda op: op.order):
            operation_start = time.time()
            result = execute_batch_operation(operation)
            result.execution_time_ms = (time.time() - operation_start) * 1000

            results.append(result)

            if result.success:
                successful_operations += 1
            else:
                failed_operations += 1
                if not continue_on_error:
                    # For atomic operations, perform appropriate rollback
                    if successful_operations > 0:
                        if snapshot_id:
                            # Use snapshot-based rollback for existing documents
                            rollback_performed = self._perform_snapshot_rollback(
                                snapshot_id
                            )
                        else:
                            # Use manual rollback for new document creation
                            rollback_performed = self._perform_manual_rollback(
                                results[:successful_operations]
                            )
                    break

        execution_time = (time.time() - start_time) * 1000

        return BatchApplyResult(
            success=failed_operations == 0,
            total_operations=len(operations),
            successful_operations=successful_operations,
            failed_operations=failed_operations,
            execution_time_ms=execution_time,
            rollback_performed=rollback_performed,
            operation_results=results,
            summary=f"Executed {successful_operations}/{len(operations)} operations successfully"
            + (" (stopped early due to failure)" if rollback_performed else ""),
            error_summary=f"Batch failed: {failed_operations} operation(s) failed"
            if failed_operations > 0
            else None,
        )

    def _detect_conflicts(self, operations: list[BatchOperation]) -> list[ConflictInfo]:
        """Detect basic conflicts between operations.

        Args:
            operations: List of operations to check

        Returns:
            List of detected conflicts
        """
        conflicts = []

        # Group operations by resource (document/chapter)
        document_operations = {}
        chapter_operations = {}

        for op in operations:
            # Determine resource being accessed - check both target and parameters
            doc_name = op.target.get("document_name") or op.parameters.get(
                "document_name"
            )
            if doc_name:
                if doc_name not in document_operations:
                    document_operations[doc_name] = []
                document_operations[doc_name].append(op)

                # Also track chapter-level operations
                chapter_name = op.target.get("chapter_name") or op.parameters.get(
                    "chapter_name"
                )
                if chapter_name:
                    chapter_key = f"{doc_name}::{chapter_name}"
                    if chapter_key not in chapter_operations:
                        chapter_operations[chapter_key] = []
                    chapter_operations[chapter_key].append(op)

        # Check for write-write conflicts on same document
        for doc_name, doc_ops in document_operations.items():
            write_ops = [
                op for op in doc_ops if op.operation_type in self._write_operations
            ]
            if len(write_ops) > 1:
                conflicts.append(
                    ConflictInfo(
                        conflict_type="same_document",
                        affected_operations=[op.operation_id for op in write_ops],
                        severity="warning",
                        resolution="Will execute sequentially",
                    )
                )

        # Check for write-write conflicts on same chapter
        for chapter_key, chapter_ops in chapter_operations.items():
            write_ops = [
                op for op in chapter_ops if op.operation_type in self._write_operations
            ]
            if len(write_ops) > 1:
                # Check if operations have dependencies that would resolve the conflict
                has_dependencies = any(op.depends_on for op in write_ops)
                if has_dependencies:
                    # If there are dependencies, treat as warning - they'll execute sequentially
                    severity = "warning"
                    resolution = "Will execute sequentially due to dependencies"
                else:
                    # Only flag as error if there are truly concurrent writes with no dependencies
                    severity = "error"
                    resolution = "Cannot execute multiple writes on same chapter"

                conflicts.append(
                    ConflictInfo(
                        conflict_type="same_chapter",
                        affected_operations=[op.operation_id for op in write_ops],
                        severity=severity,
                        resolution=resolution,
                    )
                )

        return conflicts

    def _perform_snapshot_rollback(self, snapshot_id: str) -> bool:
        """Perform rollback using snapshot restoration.

        Args:
            snapshot_id: ID of the snapshot to restore

        Returns:
            bool: True if rollback was attempted, False otherwise
        """
        try:
            # Import snapshot management function
            from ..mcp_client import manage_snapshots

            # Restore the snapshot
            result = manage_snapshots(action="restore", snapshot_id=snapshot_id)

            # Check if restoration was successful
            if hasattr(result, "success"):
                return result.success
            elif isinstance(result, dict) and "success" in result:
                return result["success"]
            else:
                # If we got a result without error, assume success
                return True

        except Exception as e:
            # Log rollback failure but don't crash
            print(f"Warning: Snapshot rollback failed: {e}")
            return False

    def _perform_manual_rollback(self, successful_results: list) -> bool:
        """Perform manual rollback for operations that created new resources.

        Args:
            successful_results: List of successful operation results

        Returns:
            bool: True if rollback was attempted, False otherwise
        """
        try:
            # Import deletion functions directly
            from ..mcp_client import delete_chapter
            from ..mcp_client import delete_document

            # Rollback in reverse order
            for result in reversed(successful_results):
                operation_type = result.operation_type

                if operation_type == "create_document":
                    # Extract document name from result
                    if result.result and isinstance(result.result, dict):
                        details = result.result.get("details", {})
                        doc_name = details.get("document_name")
                        if doc_name:
                            delete_document(doc_name)

                elif operation_type == "create_chapter":
                    # Extract document and chapter names from result
                    if result.result and isinstance(result.result, dict):
                        details = result.result.get("details", {})
                        doc_name = details.get("document_name")
                        chapter_name = details.get("chapter_name")
                        if doc_name and chapter_name:
                            delete_chapter(doc_name, chapter_name)

            return True

        except Exception as e:
            print(f"Warning: Manual rollback failed: {e}")
            return False

    def _get_operation_resource(self, operation: BatchOperation) -> str:
        """Get the resource identifier for an operation."""
        # Check both target and parameters for document_name
        doc_name = operation.target.get("document_name") or operation.parameters.get(
            "document_name"
        )
        if doc_name:
            # Check both target and parameters for chapter_name
            chapter_name = operation.target.get(
                "chapter_name"
            ) or operation.parameters.get("chapter_name")
            if chapter_name:
                return f"{doc_name}::{chapter_name}"
            return doc_name
        return "global"
