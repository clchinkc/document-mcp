import functools  # Added import
import json
import logging
import traceback
from enum import Enum
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

# Import metrics functionality (will gracefully handle if not available)
try:
    from .metrics_config import record_tool_call_error
    from .metrics_config import record_tool_call_start
    from .metrics_config import record_tool_call_success

    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False


class ErrorCategory(Enum):
    """Categories for different types of errors to help with analysis and alerting."""

    CRITICAL = "CRITICAL"  # System-breaking errors that require immediate attention
    ERROR = "ERROR"  # Functional errors that prevent operation completion
    WARNING = "WARNING"  # Non-blocking issues that should be monitored
    INFO = "INFO"  # Informational messages for debugging


class StructuredLogFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging."""

    def format(self, record):
        # Create base log entry
        log_entry = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add exception information if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info),
            }

        # Add any extra context from the log record
        for key, value in record.__dict__.items():
            if key not in [
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "getMessage",
                "exc_info",
                "exc_text",
                "stack_info",
            ]:
                log_entry[key] = value

        return json.dumps(log_entry, default=str, ensure_ascii=False)


# --- Enhanced Logging Setup ---
mcp_call_logger = logging.getLogger("mcp_call_logger")
mcp_call_logger.setLevel(logging.INFO)

# Structured error logger
error_logger = logging.getLogger("error_logger")
error_logger.setLevel(logging.INFO)

# Use Path(__file__).resolve().parent to ensure the path is correct even if the script is called from elsewhere.
log_file_path = Path(__file__).resolve().parent / "mcp_calls.log"
error_log_path = Path(__file__).resolve().parent / "errors.log"

# Use RotatingFileHandler for log rotation
# maxBytes: 10MB per file, backupCount: 5 files (total ~50MB)
file_handler = RotatingFileHandler(
    log_file_path,
    maxBytes=10 * 1024 * 1024,
    backupCount=5,  # 10MB
)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
mcp_call_logger.addHandler(file_handler)

# Structured error log handler
error_file_handler = RotatingFileHandler(
    error_log_path,
    maxBytes=10 * 1024 * 1024,
    backupCount=5,  # 10MB
)
error_file_handler.setFormatter(StructuredLogFormatter())
error_logger.addHandler(error_file_handler)

# Prevent logs from propagating to the root logger if not desired
mcp_call_logger.propagate = False
error_logger.propagate = False


def log_structured_error(
    category: ErrorCategory,
    message: str,
    exception: Exception | None = None,
    context: dict[str, Any] | None = None,
    operation: str | None = None,
    **kwargs,
):
    """Log a structured error with comprehensive context information.

    Args:
        category: Error severity category
        message: Human-readable error description
        exception: Exception object if applicable
        context: Additional context dictionary (e.g., file paths, user inputs)
        operation: Name of the operation being performed
        **kwargs: Additional context fields
    """
    # Determine log level based on category
    level_map = {
        ErrorCategory.CRITICAL: logging.CRITICAL,
        ErrorCategory.ERROR: logging.ERROR,
        ErrorCategory.WARNING: logging.WARNING,
        ErrorCategory.INFO: logging.INFO,
    }
    level = level_map.get(category, logging.ERROR)

    # Build structured context
    log_context = {
        "error_category": category.value,
        "operation": operation,
        **(context or {}),
        **kwargs,
    }

    # Log with structured context
    error_logger.log(level, message, exc_info=exception is not None, extra=log_context)


def safe_operation(
    operation_name: str,
    operation_func,
    *args,
    error_category: ErrorCategory = ErrorCategory.ERROR,
    context: dict[str, Any] | None = None,
    **kwargs,
):
    """Execute an operation with enhanced error handling and logging.

    Args:
        operation_name: Name of the operation for logging
        operation_func: Function to execute
        *args: Arguments for the operation function
        error_category: Category to assign to any errors
        context: Additional context for error logging
        **kwargs: Additional keyword arguments for the operation function

    Returns:
        Tuple of (success: bool, result: Any, error: Optional[Exception])
    """
    try:
        result = operation_func(*args, **kwargs)
        return True, result, None
    except Exception as e:
        log_structured_error(
            category=error_category,
            message=f"Operation '{operation_name}' failed: {str(e)}",
            exception=e,
            context=context,
            operation=operation_name,
            function_args=repr(args),
            function_kwargs=repr(kwargs),
        )
        return False, None, e


# --- Decorator for Logging MCP Calls with Metrics ---
def log_mcp_call(func):
    @functools.wraps(func)  # Use functools.wraps to preserve function metadata
    def wrapper(*args, **kwargs):
        # Try to get the actual function name if it's a bound method or has other wrappers
        func_name = getattr(func, "__name__", "unknown_function")

        # Start metrics recording
        start_time = None
        if METRICS_AVAILABLE:
            try:
                start_time = record_tool_call_start(func_name, args, kwargs)
            except Exception as e:
                # Don't let metrics errors break the function call
                log_structured_error(
                    category=ErrorCategory.WARNING,
                    message=f"Metrics recording failed for {func_name}",
                    exception=e,
                    operation="metrics_start",
                    function=func_name,
                )

        try:
            logged_args = []
            for arg in args:
                if hasattr(arg, "model_dump_json"):  # Check for Pydantic v2 model
                    logged_args.append(arg.model_dump_json(indent=None, exclude_none=True))  # Compact JSON
                elif hasattr(arg, "json"):  # Check for Pydantic v1 model
                    logged_args.append(arg.json(indent=None, exclude_none=True))  # Compact JSON
                else:
                    logged_args.append(repr(arg))

            logged_kwargs = {}
            for k, v in kwargs.items():
                if hasattr(v, "model_dump_json"):
                    logged_kwargs[k] = v.model_dump_json(indent=None, exclude_none=True)
                elif hasattr(v, "json"):
                    logged_kwargs[k] = v.json(indent=None, exclude_none=True)
                else:
                    logged_kwargs[k] = repr(v)

            arg_str = f"args={logged_args}, kwargs={logged_kwargs}"
        except Exception as e:
            arg_str = f"args/kwargs logging error: {e}"
            log_structured_error(
                category=ErrorCategory.WARNING,
                message="Failed to serialize function arguments for logging",
                exception=e,
                operation="argument_serialization",
                function=func_name,
            )

        mcp_call_logger.info(f"Calling tool: {func_name} with {arg_str}")
        try:
            result = func(*args, **kwargs)

            # Record successful completion in metrics
            if METRICS_AVAILABLE:
                try:
                    # Calculate result size for metrics
                    result_size = 0
                    try:
                        if hasattr(result, "__len__") and not isinstance(result, str):
                            result_size = len(str(result))
                        elif isinstance(result, str):
                            result_size = len(result.encode("utf-8"))
                        else:
                            result_size = len(str(result))
                    except Exception:
                        result_size = 0

                    record_tool_call_success(func_name, start_time, result_size)
                except Exception as e:
                    log_structured_error(
                        category=ErrorCategory.WARNING,
                        message=f"Metrics success recording failed for {func_name}",
                        exception=e,
                        operation="metrics_success",
                        function=func_name,
                    )

            try:
                if isinstance(result, list):
                    # Check if it's a list of Pydantic-like objects
                    if result and (hasattr(result[0], "model_dump_json") or hasattr(result[0], "json")):
                        logged_list = []
                        for item in result:
                            if hasattr(item, "model_dump_json"):
                                logged_list.append(item.model_dump_json(indent=None, exclude_none=True))
                            elif hasattr(item, "json"):  # Pydantic v1 fallback
                                logged_list.append(item.json(indent=None, exclude_none=True))
                            else:
                                logged_list.append(repr(item))
                        result_str = "[" + ", ".join(logged_list) + "]"
                    else:
                        # It's a list, but not of Pydantic models (or empty)
                        result_str = repr(result)
                elif hasattr(result, "model_dump_json"):  # Single Pydantic v2 model
                    result_str = result.model_dump_json(indent=None, exclude_none=True)
                elif hasattr(result, "json"):  # Single Pydantic v1 model
                    result_str = result.json(indent=None, exclude_none=True)
                else:  # Other types
                    result_str = repr(result)
                # Removed the 500-character truncation
            except Exception as e:
                result_str = f"Result logging error: {e}"
                log_structured_error(
                    category=ErrorCategory.WARNING,
                    message="Failed to serialize function result for logging",
                    exception=e,
                    operation="result_serialization",
                    function=func_name,
                )

            mcp_call_logger.info(f"Tool {func_name} returned: {result_str}")
            return result
        except Exception as e:
            # Record error in metrics
            if METRICS_AVAILABLE:
                try:
                    record_tool_call_error(func_name, start_time, e)
                except Exception as metrics_error:
                    log_structured_error(
                        category=ErrorCategory.WARNING,
                        message=f"Metrics error recording failed for {func_name}",
                        exception=metrics_error,
                        operation="metrics_error",
                        function=func_name,
                        original_error=str(e),
                    )

            # Enhanced error logging with structured context
            log_structured_error(
                category=ErrorCategory.ERROR,
                message=f"Tool execution failed: {func_name}",
                exception=e,
                operation="tool_execution",
                function=func_name,
                arguments=arg_str,
            )

            raise

    return wrapper
