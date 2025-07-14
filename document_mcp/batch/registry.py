"""
Registry for batch operations and tool mapping.

This module provides a simple registry that manages the mapping
between operation types and their corresponding tool functions.
"""

from typing import Any, Dict, List, Optional

from .models import BatchOperation, OperationResult


class BatchOperationRegistry:
    """Simple registry for batchable operations."""
    
    def __init__(self):
        self._operations: Dict[str, str] = {}  # operation_type -> tool_function_name
        
    def register_operation(self, operation_type: str, tool_function_name: str):
        """Register a tool as batchable operation."""
        self._operations[operation_type] = tool_function_name
        
    def get_tool_function_name(self, operation_type: str) -> Optional[str]:
        """Get the actual tool function name for an operation type."""
        return self._operations.get(operation_type)
        
    def is_valid_operation(self, operation_type: str) -> bool:
        """Check if operation type is registered."""
        return operation_type in self._operations
    
    def get_batchable_operations(self) -> List[str]:
        """Return list of all registered operation types."""
        return list(self._operations.keys())


def register_batchable_operation(operation_type: str, tool_function_name: str = None):
    """Decorator to register a tool as batchable."""
    def decorator(func):
        # Use global registry from main module
        from ..doc_tool_server import _batch_registry
        _batch_registry.register_operation(
            operation_type, 
            tool_function_name or func.__name__
        )
        return func
    return decorator


def execute_batch_operation(operation: BatchOperation) -> OperationResult:
    """Execute a single operation within a batch using safe_operation."""
    # Import here to avoid circular imports
    from ..doc_tool_server import _batch_registry
    from ..logger_config import safe_operation, ErrorCategory
    
    # Get the tool function name
    tool_function_name = _batch_registry.get_tool_function_name(operation.operation_type)
    if not tool_function_name:
        return OperationResult(
            success=False,
            operation_id=operation.operation_id,
            operation_type=operation.operation_type,
            error=f"Unknown operation type: {operation.operation_type}"
        )
    
    # Get the actual function from main module globals
    import sys
    doc_tool_module = sys.modules.get('document_mcp.doc_tool_server')
    if not doc_tool_module:
        return OperationResult(
            success=False,
            operation_id=operation.operation_id,
            operation_type=operation.operation_type,
            error="Document tool server module not found"
        )
    
    tool_function = getattr(doc_tool_module, tool_function_name, None)
    if not tool_function:
        return OperationResult(
            success=False,
            operation_id=operation.operation_id,
            operation_type=operation.operation_type,
            error=f"Tool function not found: {tool_function_name}"
        )
    
    # Prepare arguments - merge target and parameters
    all_args = {**operation.target, **operation.parameters}
    
    # Execute using safe_operation for robust error handling
    success, result, error = safe_operation(
        operation_name=operation.operation_type,
        operation_func=tool_function,
        error_category=ErrorCategory.ERROR,
        context={
            "batch_operation_id": operation.operation_id,
            "operation_type": operation.operation_type,
            "target": operation.target,
            "parameters": operation.parameters
        },
        **all_args
    )
    
    # Convert result to OperationResult and determine actual success
    result_data = None
    actual_success = success  # Start with safe_operation success
    
    if result:
        if hasattr(result, 'model_dump'):
            result_data = result.model_dump()
        elif hasattr(result, 'dict'):
            result_data = result.dict()
        elif isinstance(result, dict):
            result_data = result
        else:
            result_data = {"result": str(result)}
        
        # Check if the result indicates success (for OperationStatus-like objects)
        if isinstance(result_data, dict) and 'success' in result_data:
            actual_success = success and result_data['success']
        # For unified tools that return plain dicts, non-None result generally means success
        elif result_data is not None and not error:
            actual_success = True
    
    return OperationResult(
        success=actual_success,
        operation_id=operation.operation_id,
        operation_type=operation.operation_type,
        result=result_data,
        error=error if not actual_success else None
    )


def _get_function_for_operation(operation_type: str):
    """Get the actual function for an operation type."""
    from ..doc_tool_server import _batch_registry
    
    # Get the tool function name
    tool_function_name = _batch_registry.get_tool_function_name(operation_type)
    if not tool_function_name:
        return None
    
    # Get the actual function from main module globals
    import sys
    doc_tool_module = sys.modules.get('document_mcp.doc_tool_server')
    if not doc_tool_module:
        return None
    
    return getattr(doc_tool_module, tool_function_name, None)