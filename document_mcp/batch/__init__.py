"""Simplified batch processing for document operations.

This module provides essential batch processing capabilities for document
management workflows without premature optimization or complexity.

Key Components:
- BatchExecutor: Simple sequential execution with conflict detection
- BatchOperationRegistry: Maps operation types to tool functions
- Essential data models for batch operations and results

The system focuses on:
- Sequential execution with basic conflict detection
- Simple operation registry and tool mapping
- Clear error handling and result reporting
- Integration with existing document operations
"""

from .executor import BatchExecutor
from .models import BatchApplyResult
from .models import BatchOperation
from .models import ConflictInfo
from .models import OperationResult
from .registry import BatchOperationRegistry
from .registry import execute_batch_operation
from .registry import register_batchable_operation

__all__ = [
    # Models
    "BatchOperation",
    "OperationResult",
    "BatchApplyResult",
    "ConflictInfo",
    # Core Components
    "BatchExecutor",
    "BatchOperationRegistry",
    # Utilities
    "execute_batch_operation",
    "register_batchable_operation",
]

__version__ = "1.0.0"
__author__ = "Document MCP Development Team"
