"""Safety decorators for MCP tools.

This module provides decorators for automatic safety and snapshot features:
- auto_snapshot: Automatic snapshot creation before edit operations
- safety_enhanced_write_operation: Automatic snapshots, freshness checks, and result enrichment for write operations
"""

import datetime
from functools import wraps
from pathlib import Path

from ..logger_config import ErrorCategory
from ..logger_config import log_structured_error
from ..models import ContentFreshnessStatus
from ..models import OperationStatus
from ..utils.file_operations import get_current_user


def auto_snapshot(operation_name: str):
    """Decorator for automatic snapshot creation before edit operations."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract document name from arguments (first positional or keyword)
            document_name = None
            if args and isinstance(args[0], str):
                document_name = args[0]
            elif "document_name" in kwargs:
                document_name = kwargs["document_name"]

            # Create snapshot if we have a document name
            if document_name:
                try:
                    user_id = get_current_user()
                    message = f"Auto-snapshot before {operation_name} by {user_id}"

                    # Import locally to avoid circular imports
                    from ..tools.safety_tools import _create_snapshot

                    _create_snapshot(document_name, message, auto_cleanup=True)
                except Exception as e:
                    # Log warning but don't fail the operation
                    log_structured_error(
                        category=ErrorCategory.WARNING,
                        message=f"Auto-snapshot failed for {operation_name}",
                        exception=e,
                        operation="auto_snapshot",
                    )

            # Execute the original operation
            return func(*args, **kwargs)

        return wrapper

    return decorator


# Removed create_automatic_snapshot - auto_snapshot decorator now handles snapshot creation directly


def safety_enhanced_write_operation(
    operation_name: str, create_snapshot: bool = False, check_freshness: bool = True
):
    """Comprehensive safety decorator combining automatic snapshots, freshness validation, and result enrichment.

    Note: @auto_snapshot decorator handles snapshot creation separately.
    This decorator focuses on file freshness checking and result enhancement.
    """

    def decorator(func):
        protected_func = func

        if check_freshness:
            protected_func = check_file_freshness(protected_func)

        protected_func = enhance_operation_result(protected_func)
        return protected_func

    return decorator


def check_file_freshness(func):
    """Decorator to check file freshness before write operations."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        document_name, chapter_name, last_known_modified, force_write = _extract_operation_parameters(
            args, kwargs
        )

        # Parse timestamp
        last_known_dt = _parse_timestamp(last_known_modified)
        if last_known_modified and last_known_dt is None:
            return OperationStatus(
                success=False,
                message=f"Invalid timestamp format: {last_known_modified}",
                warnings=[f"Invalid timestamp format: {last_known_modified}"],
            )

        # Check freshness
        operation_path = _get_operation_path(document_name, chapter_name)
        safety_info = _check_file_freshness(operation_path, last_known_dt)

        # Handle conflicts
        if safety_info.safety_status in ["warning", "conflict"] and not force_write:
            warnings = [f"File {safety_info.safety_status} detected: {safety_info.message}"]
            warnings.extend(safety_info.recommendations)
            return OperationStatus(
                success=False,
                message=f"Safety check failed: {safety_info.message}. Use force_write=True to proceed.",
                safety_info=safety_info,
                warnings=warnings,
            )

        # Store safety info for later use by enhance_operation_result
        kwargs["_safety_info"] = safety_info

        # Call the function with original kwargs (but remove the _safety_info since it's not a real parameter)
        original_kwargs = {k: v for k, v in kwargs.items() if k != "_safety_info"}
        result = func(*args, **original_kwargs)

        # Re-add safety info to result if needed for enhance_operation_result
        if hasattr(result, "__dict__") or isinstance(result, dict):
            kwargs["_safety_info"] = safety_info

        return result

    return wrapper


def enhance_operation_result(func):
    """Decorator to enhance operation results with safety information."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)

        # Enhance result with safety information if successful
        if result and hasattr(result, "success") and result.success:
            safety_info = kwargs.get("_safety_info")

            # Add safety info to result
            if safety_info:
                # Ensure safety_info is properly serialized
                if hasattr(safety_info, "model_dump"):
                    result.safety_info = safety_info
                else:
                    result.safety_info = safety_info
            else:
                result.safety_info = None

            # Add warnings if safety info has them
            if safety_info and hasattr(safety_info, "recommendations"):
                if not hasattr(result, "warnings"):
                    result.warnings = []
                if safety_info.safety_status == "warning":
                    result.warnings.append(f"File was modified externally: {safety_info.message}")

        return result

    return wrapper


def _extract_operation_parameters(args, kwargs):
    """Extract operation parameters from function arguments."""
    document_name = None
    chapter_name = None
    last_known_modified = None
    force_write = False

    # Extract document_name
    if args and isinstance(args[0], str):
        document_name = args[0]
    elif "document_name" in kwargs:
        document_name = kwargs["document_name"]

    # Extract chapter_name
    if len(args) > 1 and isinstance(args[1], str):
        chapter_name = args[1]
    elif "chapter_name" in kwargs:
        chapter_name = kwargs["chapter_name"]

    # Extract last_known_modified and force_write - depends on function signature
    if "last_known_modified" in kwargs:
        last_known_modified = kwargs["last_known_modified"]
    if "force_write" in kwargs:
        force_write = kwargs["force_write"]

    # Handle positional arguments based on function signature
    if len(args) == 5:  # write_chapter_content: (doc, chapter, content, last_known_modified, force_write)
        if last_known_modified is None:
            last_known_modified = args[3]
        if not force_write:
            force_write = args[4] if args[4] is not None else False
    elif len(args) == 6:  # replace_paragraph: (doc, chapter, idx, content, last_known_modified, force_write)
        if last_known_modified is None:
            last_known_modified = args[4]
        if not force_write:
            force_write = args[5] if args[5] is not None else False
    elif (
        len(args) == 4
    ):  # write_chapter_content without force_write: (doc, chapter, content, last_known_modified)
        if last_known_modified is None:
            last_known_modified = args[3]

    return document_name, chapter_name, last_known_modified, force_write


def _parse_timestamp(timestamp_str: str | None) -> datetime.datetime | None:
    """Parse ISO timestamp string to datetime object."""
    if not timestamp_str:
        return None

    try:
        # Handle ISO format with Z suffix
        if timestamp_str.endswith("Z"):
            timestamp_str = timestamp_str[:-1] + "+00:00"

        dt = datetime.datetime.fromisoformat(timestamp_str)

        # Convert to naive datetime for consistent comparison
        if dt.tzinfo:
            dt = dt.replace(tzinfo=None)

        return dt
    except (ValueError, AttributeError):
        return None


def _get_operation_path(document_name: str, chapter_name: str | None = None) -> Path:
    """Get the file path for the operation."""
    import os

    from ..utils.file_operations import DOCS_ROOT_PATH

    # Use environment variable if available for test isolation
    if "PYTEST_CURRENT_TEST" in os.environ:
        docs_root_name = os.environ.get("DOCUMENT_ROOT_DIR", str(DOCS_ROOT_PATH))
        root_path = Path(docs_root_name)
    else:
        root_path = DOCS_ROOT_PATH

    if chapter_name:
        return root_path / document_name / chapter_name
    else:
        return root_path / document_name


def _check_file_freshness(
    file_path: Path, last_known_modified: datetime.datetime | None = None
) -> ContentFreshnessStatus:
    """Check if a file has been modified since last known modification time."""
    if not file_path.exists():
        return ContentFreshnessStatus(
            is_fresh=False,
            last_modified=datetime.datetime.now(),
            last_known_modified=last_known_modified,
            safety_status="conflict",
            message="File no longer exists",
            recommendations=[
                "Verify file was not accidentally deleted",
                "Consider restoring from snapshot",
            ],
        )

    current_modified = datetime.datetime.fromtimestamp(file_path.stat().st_mtime)

    if last_known_modified is None:
        return ContentFreshnessStatus(
            is_fresh=True,
            last_modified=current_modified,
            last_known_modified=None,
            safety_status="safe",
            message="No previous modification time to compare against",
            recommendations=[],
        )

    time_diff = abs((current_modified - last_known_modified).total_seconds())

    if time_diff < 1:  # Within 1 second tolerance
        return ContentFreshnessStatus(
            is_fresh=True,
            last_modified=current_modified,
            last_known_modified=last_known_modified,
            safety_status="safe",
            message="Content is fresh and safe to modify",
            recommendations=[],
        )
    else:
        return ContentFreshnessStatus(
            is_fresh=False,
            last_modified=current_modified,
            last_known_modified=last_known_modified,
            safety_status="warning",
            message=f"Content was modified {time_diff:.1f} seconds ago by external source",
            recommendations=[
                "Re-read content before proceeding",
                "Consider creating a snapshot before modifying",
                "Use force_write=True to proceed anyway",
            ],
        )
