"""Batch processing MCP tools for document management.

This module provides the batch operation execution tool for performing
multiple document operations atomically.
"""

import time
from typing import Any

from ..batch import BatchExecutor
from ..batch import BatchOperation
from ..logger_config import ErrorCategory
from ..logger_config import log_mcp_call
from ..logger_config import log_structured_error
from ..models import BatchApplyResult


def register_batch_tools(mcp_server):
    """Register all batch-related tools with the MCP server."""

    @mcp_server.tool()
    @log_mcp_call
    def batch_apply_operations(
        operations: list[dict[str, Any]],
        atomic: bool = True,
        validate_only: bool = False,
        snapshot_before: bool = False,
        continue_on_error: bool = False,
    ) -> dict[str, Any]:
        r"""Execute multiple operations as a batch with sequential execution.

        This tool enables atomic execution of multiple document operations with
        automatic rollback on failure through snapshots.

        Parameters:
            operations (List[Dict]): List of operations to execute. Each operation must contain:
                - operation_type (str): Name of the operation/tool to execute
                - target (Dict[str, str]): Target identifiers (document_name, chapter_name, etc.)
                - parameters (Dict[str, Any]): Operation-specific parameters
                - order (int): Execution sequence number
                - operation_id (str, optional): Unique identifier for tracking

            atomic (bool): If True, all operations succeed or all rollback (default: True)
            validate_only (bool): If True, only validate operations without executing (default: False)
            snapshot_before (bool): If True, create snapshot before execution for rollback (default: False)
            continue_on_error (bool): If True and atomic=False, continue after individual failures (default: False)

        Returns:
            BatchApplyResult: Complete batch execution results including:
                - success (bool): Overall batch success status
                - total_operations (int): Number of operations in batch
                - successful_operations (int): Number of successfully executed operations
                - failed_operations (int): Number of failed operations
                - execution_time_ms (float): Total execution time in milliseconds
                - rollback_performed (bool): Whether rollback was triggered
                - operation_results (List[OperationResult]): Individual operation results
                - snapshot_id (str, optional): ID of created snapshot if snapshot_before=True
                - error_summary (str, optional): Summary of errors if any occurred
                - summary (str): Human-readable execution summary

        Example Usage:
            ```json
            {
                "name": "batch_apply_operations",
                "arguments": {
                    "operations": [
                        {
                            "operation_type": "create_document",
                            "target": {},
                            "parameters": {"document_name": "My Novel"},
                            "order": 1,
                            "operation_id": "create_doc"
                        },
                        {
                            "operation_type": "create_chapter",
                            "target": {"document_name": "My Novel"},
                            "parameters": {
                                "chapter_name": "01-intro.md",
                                "initial_content": "# Chapter 1\\n\\nOnce upon a time..."
                            },
                            "order": 2,
                            "operation_id": "create_intro",
                            "depends_on": ["create_doc"]
                        }
                    ],
                    "atomic": true,
                    "snapshot_before": true
                }
            }
            ```
        """
        # Import here to avoid circular imports
        from ..batch.global_registry import get_batch_registry
        from ..helpers import _get_document_path
        from ..helpers import _resolve_operation_dependencies
        from ..mcp_client import manage_snapshots

        _batch_registry = get_batch_registry()

        start_time = time.time()

        try:
            # Convert dict operations to BatchOperation objects
            batch_ops = []
            for i, op_dict in enumerate(operations):
                try:
                    batch_op = BatchOperation(
                        operation_type=op_dict.get("operation_type", ""),
                        target=op_dict.get("target", {}),
                        parameters=op_dict.get("parameters", {}),
                        order=op_dict.get("order", i),
                        operation_id=op_dict.get("operation_id", f"op_{i}"),
                        depends_on=op_dict.get("depends_on", []),
                    )
                    batch_ops.append(batch_op)
                except Exception as e:
                    return BatchApplyResult(
                        success=False,
                        total_operations=len(operations),
                        successful_operations=0,
                        failed_operations=len(operations),
                        execution_time_ms=0,
                        operation_results=[],
                        error_summary=f"Failed to parse operation {i}: {str(e)}",
                        summary="Batch validation failed",
                    )

            # Resolve dependencies and sort operations in executable order
            try:
                sorted_ops = _resolve_operation_dependencies(batch_ops)
            except ValueError as e:
                return BatchApplyResult(
                    success=False,
                    total_operations=len(operations),
                    successful_operations=0,
                    failed_operations=len(operations),
                    execution_time_ms=0,
                    operation_results=[],
                    error_summary=f"Dependency resolution failed: {str(e)}",
                    summary="Batch dependency resolution failed",
                )

            # If validate_only mode, validate operations and return
            if validate_only:
                validation_errors = []
                for op in sorted_ops:
                    if not _batch_registry.is_valid_operation(op.operation_type):
                        validation_errors.append(f"Unknown operation type: {op.operation_type}")

                execution_time = (time.time() - start_time) * 1000
                if validation_errors:
                    return BatchApplyResult(
                        success=False,
                        total_operations=len(operations),
                        successful_operations=0,
                        failed_operations=len(operations),
                        execution_time_ms=execution_time,
                        operation_results=[],
                        error_summary="; ".join(validation_errors),
                        summary="Batch validation failed",
                    )
                else:
                    return BatchApplyResult(
                        success=True,
                        total_operations=len(operations),
                        successful_operations=0,
                        failed_operations=0,
                        execution_time_ms=execution_time,
                        operation_results=[],
                        summary="Validation successful - no operations executed",
                    )

            # Create snapshot before execution if requested
            snapshot_id = None
            if snapshot_before:
                # Extract existing documents that will be modified from operations
                affected_docs = set()
                for op in sorted_ops:
                    doc_name = op.target.get("document_name") or op.parameters.get("document_name")
                    if doc_name and op.operation_type != "create_document":
                        # Only snapshot existing documents (skip create_document operations)
                        doc_path = _get_document_path(doc_name)
                        if doc_path.exists():
                            affected_docs.add(doc_name)

                # Create snapshots for existing affected documents
                if affected_docs:
                    try:
                        # Use first existing document for snapshot
                        doc_name = list(affected_docs)[0]
                        snapshot_result = manage_snapshots(
                            doc_name,
                            action="create",
                            message=f"Batch operation snapshot before {len(operations)} operations",
                        )
                        if snapshot_result.success:
                            snapshot_id = snapshot_result.details.get("snapshot_id")
                    except Exception as e:
                        log_structured_error(
                            ErrorCategory.OPERATION_FAILED,
                            f"Failed to create snapshot before batch execution: {e}",
                            {
                                "operation": "batch_apply_operations",
                                "snapshot_before": True,
                            },
                        )

            # Execute operations using simplified batch executor
            executor = BatchExecutor()
            result = executor.execute_batch(
                sorted_ops,
                continue_on_error=continue_on_error,
                snapshot_id=snapshot_id,
            )

            # Add snapshot information if created
            if snapshot_id:
                result.snapshot_id = snapshot_id
            return result

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return BatchApplyResult(
                success=False,
                total_operations=len(operations) if operations else 0,
                successful_operations=0,
                failed_operations=len(operations) if operations else 0,
                execution_time_ms=execution_time,
                operation_results=[],
                summary=f"Batch execution failed with error: {str(e)}",
                error_summary=str(e),
            )
