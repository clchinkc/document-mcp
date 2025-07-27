"""Simplified batch operation models for document management.

This module contains the essential data structures for batch operations,
focused on document management workflows without premature optimization.
"""

import time
from typing import Any

from pydantic import BaseModel
from pydantic import Field


class BatchApplyRequest(BaseModel):
    """Batch execution request."""

    operations: list[dict[str, Any]]  # Will be converted to BatchOperation objects
    atomic: bool = True  # All succeed or all fail
    validate_only: bool = False  # Dry-run mode
    snapshot_before: bool = False  # Auto-snapshot before execution
    continue_on_error: bool = False  # Continue despite individual failures
    execution_mode: str = "sequential"  # sequential, parallel_safe


class BatchOperation(BaseModel):
    """Represents a single operation within a batch.

    This is the fundamental unit of work that can be executed individually
    or as part of a batch sequence.
    """

    operation_type: str = Field(..., description="Type of operation (e.g., 'create_document')")
    target: dict[str, Any] = Field(default_factory=dict, description="Target specification")
    parameters: dict[str, Any] = Field(..., description="Operation parameters")
    order: int = Field(..., description="Order of execution in batch")
    operation_id: str = Field(..., description="Unique identifier for this operation")
    depends_on: list[str] | None = Field(default=None, description="List of operation IDs this depends on")


class OperationResult(BaseModel):
    """Result of executing a single batch operation.

    Contains both success/failure information and the actual result data
    from the operation execution.
    """

    operation_id: str = Field(..., description="ID of the executed operation")
    operation_type: str = Field(..., description="Type of operation executed")
    success: bool = Field(..., description="Whether operation succeeded")
    result: dict[str, Any] | None = Field(default=None, description="Operation result data")
    error: str | None = Field(default=None, description="Error message if failed")
    execution_time_ms: float = Field(default=0.0, description="Execution time in milliseconds")
    timestamp: str = Field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))


class BatchApplyResult(BaseModel):
    """Complete result of executing a batch of operations.

    This is the top-level result returned by the batch execution system,
    containing summary information and individual operation results.
    """

    success: bool = Field(..., description="Overall batch success")
    total_operations: int = Field(..., description="Total number of operations")
    successful_operations: int = Field(default=0, description="Number of successful operations")
    failed_operations: int = Field(default=0, description="Number of failed operations")
    execution_time_ms: float = Field(default=0.0, description="Total execution time")
    rollback_performed: bool = Field(default=False, description="Whether rollback was performed")
    operation_results: list[OperationResult] = Field(
        default_factory=list, description="Individual operation results"
    )
    summary: str = Field(default="", description="Human-readable execution summary")
    error_summary: str | None = Field(default=None, description="Error summary if batch failed")


class ConflictInfo(BaseModel):
    """Information about conflicts between operations.

    Used for simple conflict detection in batch operations.
    """

    conflict_type: str = Field(..., description="Type of conflict ('same_document', 'same_chapter')")
    affected_operations: list[str] = Field(..., description="Operation IDs affected by this conflict")
    severity: str = Field(..., description="Severity level ('warning', 'error')")
    resolution: str = Field(..., description="How the conflict should be resolved")
