"""Centralized error handling utilities for the Document MCP system.

This module provides utilities for consistent error handling, logging,
and response formatting across all components.
"""

import functools
import logging
import traceback
from collections.abc import Callable
from typing import Any
from typing import TypeVar

from .exceptions import DocumentMCPError
from .exceptions import OperationError
from .models import OperationStatus

logger = logging.getLogger(__name__)

T = TypeVar("T")


def safe_operation(
    operation_name: str,
    default_return: Any = None,
    log_errors: bool = True,
    raise_on_error: bool = False,
) -> Callable:
    """Decorator for safe operation execution with consistent error handling.

    Args:
        operation_name: Name of the operation for logging and error reporting
        default_return: Value to return if operation fails (when not raising)
        log_errors: Whether to log errors when they occur
        raise_on_error: Whether to re-raise exceptions after handling

    Returns:
        Decorated function with error handling
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T | Any]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T | Any:
            try:
                return func(*args, **kwargs)
            except DocumentMCPError as e:
                # Document MCP errors are already well-formed
                if log_errors:
                    logger.error(
                        f"Operation '{operation_name}' failed",
                        extra={
                            "operation": operation_name,
                            "error_type": e.__class__.__name__,
                            "error_code": e.error_code,
                            "error_details": e.details,
                        },
                    )

                if raise_on_error:
                    raise
                return default_return

            except Exception as e:
                # Convert generic exceptions to DocumentMCPError
                error = OperationError(
                    operation=operation_name,
                    reason=str(e),
                    details={
                        "original_error_type": e.__class__.__name__,
                        "traceback": traceback.format_exc(),
                    },
                )

                if log_errors:
                    logger.error(
                        f"Unexpected error in operation '{operation_name}'",
                        extra={
                            "operation": operation_name,
                            "error_type": e.__class__.__name__,
                            "error_message": str(e),
                            "traceback": traceback.format_exc(),
                        },
                    )

                if raise_on_error:
                    raise error from e
                return default_return

        return wrapper

    return decorator


def create_error_response(
    error: DocumentMCPError | Exception,
    operation_name: str | None = None,
) -> OperationStatus:
    """Create a standardized error response from an exception.

    Args:
        error: The exception that occurred
        operation_name: Name of the operation that failed

    Returns:
        OperationStatus with error information
    """
    if isinstance(error, DocumentMCPError):
        # Use the structured error information
        return OperationStatus(
            success=False,
            message=error.user_message,
            details={
                "error_code": error.error_code,
                "error_type": error.__class__.__name__,
                "technical_message": error.message,
                "error_details": error.details,
                "operation": operation_name,
            },
        )
    else:
        # Handle generic exceptions
        return OperationStatus(
            success=False,
            message=f"An unexpected error occurred: {str(error)}",
            details={
                "error_code": "UNEXPECTED_ERROR",
                "error_type": error.__class__.__name__,
                "technical_message": str(error),
                "operation": operation_name,
            },
        )


def handle_mcp_tool_error(
    tool_name: str,
    error: Exception,
    context: dict[str, Any] | None = None,
) -> OperationStatus:
    """Handle errors from MCP tool execution.

    Args:
        tool_name: Name of the MCP tool that failed
        error: The exception that occurred
        context: Additional context for error reporting

    Returns:
        OperationStatus with error information
    """
    logger.error(
        f"MCP tool '{tool_name}' failed",
        extra={
            "tool_name": tool_name,
            "error_type": error.__class__.__name__,
            "error_message": str(error),
            "context": context or {},
        },
    )

    return create_error_response(error, f"mcp_tool_{tool_name}")


def log_operation_start(operation_name: str, **context) -> None:
    """Log the start of an operation with context.

    Args:
        operation_name: Name of the operation
        **context: Additional context to log
    """
    logger.info(
        f"Starting operation: {operation_name}", extra={"operation": operation_name, "context": context}
    )


def log_operation_success(operation_name: str, result: Any = None, **context) -> None:
    """Log successful completion of an operation.

    Args:
        operation_name: Name of the operation
        result: Operation result (will be summarized for logging)
        **context: Additional context to log
    """
    # Summarize result for logging (avoid logging large objects)
    result_summary = None
    if result is not None:
        if hasattr(result, "__dict__"):
            result_summary = f"{result.__class__.__name__} object"
        elif isinstance(result, list | dict):
            result_summary = f"{type(result).__name__} with {len(result)} items"
        else:
            result_summary = str(result)[:100]  # Truncate long strings

    logger.info(
        f"Operation completed successfully: {operation_name}",
        extra={
            "operation": operation_name,
            "result_summary": result_summary,
            "context": context,
        },
    )


def validate_required_fields(data: dict[str, Any], required_fields: list[str]) -> None:
    """Validate that required fields are present in data.

    Args:
        data: Data dictionary to validate
        required_fields: List of required field names

    Raises:
        ValidationError: If any required fields are missing
    """
    from .exceptions import ValidationError

    missing_fields = [field for field in required_fields if field not in data or data[field] is None]

    if missing_fields:
        raise ValidationError(
            f"Missing required fields: {', '.join(missing_fields)}",
            details={"missing_fields": missing_fields, "provided_fields": list(data.keys())},
        )


def validate_field_type(
    data: dict[str, Any],
    field_name: str,
    expected_type: type,
    required: bool = True,
) -> None:
    """Validate that a field has the expected type.

    Args:
        data: Data dictionary to validate
        field_name: Name of the field to validate
        expected_type: Expected type for the field
        required: Whether the field is required

    Raises:
        ValidationError: If field type is invalid
    """
    from .exceptions import ValidationError

    if field_name not in data:
        if required:
            raise ValidationError(f"Missing required field: {field_name}", field=field_name)
        return

    value = data[field_name]
    if value is None and not required:
        return

    if not isinstance(value, expected_type):
        raise ValidationError(
            f"Field '{field_name}' must be of type {expected_type.__name__}, got {type(value).__name__}",
            field=field_name,
            value=value,
            details={"expected_type": expected_type.__name__, "actual_type": type(value).__name__},
        )


class ErrorContext:
    """Context manager for operation error handling with automatic logging."""

    def __init__(
        self,
        operation_name: str,
        log_start: bool = True,
        log_success: bool = True,
        raise_on_error: bool = True,
        **context,
    ):
        """Initialize error context.

        Args:
            operation_name: Name of the operation
            log_start: Whether to log operation start
            log_success: Whether to log operation success
            raise_on_error: Whether to re-raise exceptions after handling
            **context: Additional context for logging
        """
        self.operation_name = operation_name
        self.log_start = log_start
        self.log_success = log_success
        self.raise_on_error = raise_on_error
        self.context = context
        self.result = None

    def __enter__(self):
        if self.log_start:
            log_operation_start(self.operation_name, **self.context)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            # Success case
            if self.log_success:
                log_operation_success(self.operation_name, self.result, **self.context)
            return False

        # Error case
        if isinstance(exc_val, DocumentMCPError):
            logger.error(
                f"Operation '{self.operation_name}' failed with DocumentMCPError",
                extra={
                    "operation": self.operation_name,
                    "error_code": exc_val.error_code,
                    "error_type": exc_val.__class__.__name__,
                    "error_details": exc_val.details,
                    "context": self.context,
                },
            )
        else:
            logger.error(
                f"Operation '{self.operation_name}' failed with unexpected error",
                extra={
                    "operation": self.operation_name,
                    "error_type": exc_type.__name__ if exc_type else "Unknown",
                    "error_message": str(exc_val) if exc_val else "Unknown error",
                    "context": self.context,
                },
                exc_info=True,
            )

        # Return False to re-raise the exception, True to suppress it
        return not self.raise_on_error
