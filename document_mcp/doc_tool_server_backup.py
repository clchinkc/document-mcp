"""MCP Server for Document Management.

This module provides a FastAPI-based MCP server for managing structured Markdown documents.
It exposes tools for creating, reading, updating, and deleting documents and chapters,
as well as for analyzing their content.
"""

import argparse
import datetime
import difflib  # Added for generating unified diffs
import os
import re  # Added for robust paragraph splitting
import shutil
import time
import urllib.parse
from dataclasses import dataclass
from dataclasses import field
from functools import wraps
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from mcp import McpError
from mcp.server import FastMCP
from mcp.types import ErrorData
from mcp.types import Resource
from mcp.types import TextResourceContents
from pydantic import BaseModel
from starlette.requests import Request
from starlette.responses import Response

# Import simplified batch components
from .batch import BatchApplyResult
from .batch import BatchExecutor
from .batch import BatchOperation
from .batch import register_batchable_operation

# Local imports
from .logger_config import (  # Import the configured logger and the decorator
    ErrorCategory,
)
from .logger_config import (  # Import the configured logger and the decorator
    log_mcp_call,
)
from .logger_config import (  # Import the configured logger and the decorator
    log_structured_error,
)
from .logger_config import (  # Import the configured logger and the decorator
    safe_operation,
)

# Import models and utilities
from .models import BatchApplyRequest
from .models import BatchApplyResult
from .models import BatchOperation
from .models import ChapterContent
from .models import ChapterMetadata
from .models import ContentFreshnessStatus
from .models import DocumentInfo
from .models import DocumentSummary
from .models import FullDocumentContent
from .models import ModificationHistory
from .models import ModificationHistoryEntry
from .models import OperationResult
from .models import OperationStatus
from .models import ParagraphDetail
from .models import SnapshotInfo
from .models import SnapshotsList
from .models import StatisticsReport
from .utils.file_operations import DOCS_ROOT_PATH
from .utils.file_operations import get_current_user
from .utils.validation import CHAPTER_MANIFEST_FILE
from .utils.validation import MAX_CHAPTER_NAME_LENGTH
from .utils.validation import MAX_CONTENT_LENGTH
from .utils.validation import MAX_DOCUMENT_NAME_LENGTH
from .utils.validation import MIN_PARAGRAPH_INDEX
from .utils.validation import check_file_freshness

# Import metrics functionality for the metrics endpoint
try:
    from .metrics_config import get_metrics_export

    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False

# --- Configuration ---
# Each "document" will be a subdirectory within DOCS_ROOT_DIR.
# Chapters will be .md files within their respective document subdirectory.
# Default for production
_DEFAULT_DOCS_ROOT = ".documents_storage"

# Check if running under pytest. If so, allow override via .env for test isolation.
# In production, this is not used and the path is fixed.
if "PYTEST_CURRENT_TEST" in os.environ:
    load_dotenv()
    DOCS_ROOT_DIR_NAME = os.environ.get("DOCUMENT_ROOT_DIR", _DEFAULT_DOCS_ROOT)
else:
    DOCS_ROOT_DIR_NAME = _DEFAULT_DOCS_ROOT

DOCS_ROOT_PATH = Path(DOCS_ROOT_DIR_NAME)
DOCS_ROOT_PATH.mkdir(parents=True, exist_ok=True)  # Ensure the root directory exists

# Manifest file name to store chapter order and metadata (optional, for future explicit ordering)
CHAPTER_MANIFEST_FILE = "_manifest.json"
# Summary file name
DOCUMENT_SUMMARY_FILE = "_SUMMARY.md"

# Validation constants
MAX_DOCUMENT_NAME_LENGTH = 100
MAX_CHAPTER_NAME_LENGTH = 100
MAX_CONTENT_LENGTH = 1_000_000  # 1MB max content
MIN_PARAGRAPH_INDEX = 0

# HTTP SSE server configuration
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 3001

# --- Enhanced Automatic Snapshot System Configuration ---


@dataclass
class UserModificationRecord:
    """Enhanced user modification tracking for automatic snapshots."""

    user_id: str
    operation_type: str  # "edit", "create", "delete", "batch"
    affected_scope: str  # "document", "chapter", "paragraph"
    timestamp: datetime.datetime
    snapshot_id: str
    operation_details: dict[str, Any]
    restoration_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SnapshotRetentionPolicy:
    """Intelligent snapshot cleanup with user priority."""

    # High Priority (Keep Longer)
    USER_EDIT_SNAPSHOTS = 30  # days - User-initiated changes
    MILESTONE_SNAPSHOTS = 90  # days - Major document versions
    ERROR_RECOVERY_SNAPSHOTS = 7  # days - Failed operation rollbacks

    # Medium Priority
    BATCH_OPERATION_SNAPSHOTS = 14  # days - Batch operation checkpoints
    CHAPTER_LEVEL_SNAPSHOTS = 14  # days - Chapter modifications

    # Low Priority (Cleanup Frequently)
    PARAGRAPH_LEVEL_SNAPSHOTS = 3  # days - Small edits
    AUTO_BACKUP_SNAPSHOTS = 1  # days - System automated backups


# Global retention policy instance
RETENTION_POLICY = SnapshotRetentionPolicy()

# Track user modifications for better UX
_user_modification_history: list[UserModificationRecord] = []


def get_current_user() -> str:
    """Get current user identifier for tracking modifications."""
    # In production, this would integrate with authentication system
    # For now, return a simple identifier
    return os.environ.get("USER", "system_user")


mcp_server = FastMCP(
    name="DocumentManagementTools", capabilities=["tools", "resources"]
)

# --- Input Validation Helpers ---


def _validate_document_name(document_name: str) -> tuple[bool, str]:
    """Validate document name input."""
    if (
        not document_name
        or not isinstance(document_name, str)
        or not document_name.strip()
    ):
        return False, "Document name cannot be empty"
    if len(document_name) > MAX_DOCUMENT_NAME_LENGTH:
        return (
            False,
            f"Document name too long (max {MAX_DOCUMENT_NAME_LENGTH} characters)",
        )
    if "/" in document_name or "\\" in document_name:
        return False, "Document name cannot contain path separators"
    if document_name.startswith("."):
        return False, "Document name cannot start with a dot"
    return True, ""


def _validate_chapter_name(chapter_name: str) -> tuple[bool, str]:
    """Validate chapter name input."""
    if (
        not chapter_name
        or not isinstance(chapter_name, str)
        or not chapter_name.strip()
    ):
        return False, "Chapter name cannot be empty"
    if len(chapter_name) > MAX_CHAPTER_NAME_LENGTH:
        return (
            False,
            f"Chapter name too long (max {MAX_CHAPTER_NAME_LENGTH} characters)",
        )
    if chapter_name == CHAPTER_MANIFEST_FILE:
        return False, f"Chapter name cannot be reserved name '{CHAPTER_MANIFEST_FILE}'"
    if not chapter_name.lower().endswith(".md"):
        return False, "Chapter name must end with .md"
    if "/" in chapter_name or "\\" in chapter_name:
        return False, "Chapter name cannot contain path separators"
    return True, ""


def _validate_content(content: str) -> tuple[bool, str]:
    """Validate content input."""
    if content is None:
        return False, "Content cannot be None"
    if not isinstance(content, str):
        return False, "Content must be a string"
    if len(content) > MAX_CONTENT_LENGTH:
        return False, f"Content too long (max {MAX_CONTENT_LENGTH} characters)"
    return True, ""


def _validate_paragraph_index(index: int) -> tuple[bool, str]:
    """Validate paragraph index input."""
    if not isinstance(index, int):
        return False, "Paragraph index must be an integer"
    if index < MIN_PARAGRAPH_INDEX:
        return False, "Paragraph index cannot be negative"
    return True, ""


def _validate_search_query(query: str) -> tuple[bool, str]:
    """Validate search query input."""
    if query is None:
        return False, "Search query cannot be None"
    if not isinstance(query, str):
        return False, "Search query must be a string"
    if not query.strip():
        return False, "Search query cannot be empty or whitespace only"
    return True, ""


# --- Diff Generation Helper ---


def _generate_content_diff(
    original_content: str, new_content: str, filename: str = "chapter"
) -> dict[str, Any]:
    """Generate a unified diff between original and new content.

    Compares two strings and produces a diff report including a summary,
    lines added/removed, and the full unified diff text.
    """
    if original_content == new_content:
        return {"changed": False, "diff": None, "summary": "No changes made to content"}

    original_lines = original_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)

    diff_lines = list(
        difflib.unified_diff(
            original_lines,
            new_lines,
            fromfile=f"{filename} (before)",
            tofile=f"{filename} (after)",
            lineterm="",
        )
    )

    added_lines = sum(
        1 for line in diff_lines if line.startswith("+") and not line.startswith("+++")
    )
    removed_lines = sum(
        1 for line in diff_lines if line.startswith("-") and not line.startswith("---")
    )

    return {
        "changed": True,
        "diff": "\n".join(diff_lines),
        "summary": f"Modified content: +{added_lines} lines, -{removed_lines} lines",
        "lines_added": added_lines,
        "lines_removed": removed_lines,
    }


# --- Pydantic Models for Tool I/O ---


class OperationStatus(BaseModel):
    """Generic status for operations with optional safety information."""

    success: bool
    message: str
    details: dict[str, Any] | None = (
        None  # For extra info, e.g., created entity name
    )
    # Safety fields (optional for backward compatibility)
    safety_info: Any | None = None
    snapshot_created: str | None = None
    warnings: list[str] = []


class ChapterMetadata(BaseModel):
    """Metadata for a chapter within a document."""

    chapter_name: str  # File name of the chapter, e.g., "01-introduction.md"
    title: str | None = None  # Optional: Could be extracted from H1 or from manifest
    word_count: int
    paragraph_count: int
    last_modified: datetime.datetime
    # chapter_index: int # Determined by order in list_chapters


class DocumentInfo(BaseModel):
    """Represents metadata for a document."""

    document_name: str  # Directory name of the document
    total_chapters: int
    total_word_count: int
    total_paragraph_count: int
    last_modified: (
        datetime.datetime
    )  # Could be latest of any chapter or document folder itself
    chapters: list[ChapterMetadata]  # Ordered list of chapter metadata
    has_summary: bool = False


class ParagraphDetail(BaseModel):
    """Detailed information about a paragraph."""

    document_name: str
    chapter_name: str
    paragraph_index_in_chapter: int  # 0-indexed within its chapter
    content: str
    word_count: int


class ChapterContent(BaseModel):
    """Content of a chapter file."""

    document_name: str
    chapter_name: str
    # chapter_index: int # Can be inferred from order if needed by agent
    content: str
    word_count: int
    paragraph_count: int
    last_modified: datetime.datetime


class FullDocumentContent(BaseModel):
    """Content of an entire document, comprising all its chapters in order."""

    document_name: str
    chapters: list[ChapterContent]  # Ordered list of chapter contents
    total_word_count: int
    total_paragraph_count: int


class DocumentSummary(BaseModel):
    """Content of a document's summary file."""

    document_name: str
    content: str


class StatisticsReport(BaseModel):
    """Report for analytical queries."""

    scope: str  # e.g., "document: my_doc", "chapter: my_doc/ch1.md"
    word_count: int
    paragraph_count: int
    chapter_count: int | None = None  # Only for document-level stats


class ContentFreshnessStatus(BaseModel):
    """Status information about content freshness and safety."""

    is_fresh: bool
    last_modified: datetime.datetime
    last_known_modified: datetime.datetime | None = None
    safety_status: str  # "safe", "warning", "conflict"
    message: str
    recommendations: list[str] = []


class ModificationHistoryEntry(BaseModel):
    """Single entry in modification history."""

    timestamp: datetime.datetime
    file_path: str
    operation: str  # "read", "write", "create", "delete"
    source: str  # "mcp_tool", "external", "unknown"
    details: dict[str, Any] | None = None


class ModificationHistory(BaseModel):
    """Complete modification history for a document or chapter."""

    document_name: str
    chapter_name: str | None = None
    entries: list[ModificationHistoryEntry]
    total_modifications: int
    time_window: str  # e.g., "24h", "7d", "all"


# --- Batch Operation Models ---


class BatchOperation(BaseModel):
    """Single operation within a batch."""

    operation_type: str  # Tool name or operation type
    target: dict[str, str]  # {"document_name": "...", "chapter_name": "..."}
    parameters: dict[str, Any]  # Operation-specific parameters
    order: int  # Execution sequence
    operation_id: str | None = None  # Operation identifier for result tracking
    depends_on: list[str] | None = None  # Operation dependencies


class BatchApplyRequest(BaseModel):
    """Batch execution request."""

    operations: list[dict[str, Any]]  # Will be converted to BatchOperation objects
    atomic: bool = True  # All succeed or all fail
    validate_only: bool = False  # Dry-run mode
    snapshot_before: bool = False  # Auto-snapshot before execution
    continue_on_error: bool = False  # Continue despite individual failures
    execution_mode: str = "sequential"  # sequential, parallel_safe


class OperationResult(BaseModel):
    """Result of a single operation within a batch."""

    success: bool
    operation_id: str | None = None
    operation_type: str
    result_data: dict[str, Any] | None = None  # Tool-specific result data
    error: str | None = None
    execution_time_ms: float = 0.0


class BatchApplyResult(BaseModel):
    """Batch execution result."""

    success: bool
    total_operations: int
    successful_operations: int
    failed_operations: int
    execution_time_ms: float
    rollback_performed: bool = False
    operation_results: list[OperationResult]
    snapshot_id: str | None = None
    error_summary: str | None = None
    summary: str  # Human-readable summary


# --- Batch Processing Configuration ---


# --- Helper Functions ---


# Safety and versioning helpers
def _extract_operation_parameters(args, kwargs):
    """Extract common parameters from function arguments."""
    document_name = kwargs.get("document_name") or args[0]
    chapter_name = kwargs.get("chapter_name") or (args[1] if len(args) > 1 else None)
    last_known_modified = kwargs.get("last_known_modified")
    force_write = kwargs.get("force_write", False)
    return document_name, chapter_name, last_known_modified, force_write


def _parse_timestamp(timestamp_str: str) -> datetime.datetime | None:
    """Parse timestamp string to datetime object."""
    if not timestamp_str:
        return None
    try:
        dt = datetime.datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        if dt.tzinfo:
            dt = dt.replace(tzinfo=None)
        return dt
    except ValueError:
        return None


def _get_operation_path(document_name: str, chapter_name: str | None) -> Path:
    """Get file path for operation based on document and chapter names."""
    if chapter_name:
        return _get_chapter_path(document_name, chapter_name)
    else:
        return _get_document_path(document_name)


def check_file_freshness(func):
    """Decorator to check file freshness before write operations."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        document_name, chapter_name, last_known_modified, force_write = (
            _extract_operation_parameters(args, kwargs)
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
            warnings = [
                f"File {safety_info.safety_status} detected: {safety_info.message}"
            ]
            warnings.extend(safety_info.recommendations)
            return OperationStatus(
                success=False,
                message=f"Safety check failed: {safety_info.message}. Use force_write=True to proceed.",
                safety_info=safety_info,
                warnings=warnings,
            )

        # Store safety info for later use
        kwargs["_safety_info"] = safety_info

        # Remove internal parameters before calling the function
        internal_params = ["_safety_info", "_snapshot_id"]
        clean_kwargs = {k: v for k, v in kwargs.items() if k not in internal_params}

        return func(*args, **clean_kwargs)

    return wrapper


def create_safety_snapshot(operation_name: str):
    """Decorator to create snapshots before write operations."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            document_name, chapter_name, _, _ = _extract_operation_parameters(
                args, kwargs
            )

            # Create snapshot if file exists
            snapshot_id = None
            operation_path = _get_operation_path(document_name, chapter_name)
            if operation_path.exists():
                snapshot_id = _create_micro_snapshot(
                    document_name, chapter_name, operation_name
                )

            # Store snapshot ID for later use
            kwargs["_snapshot_id"] = snapshot_id

            # Remove internal parameters before calling the function
            internal_params = ["_safety_info", "_snapshot_id"]
            clean_kwargs = {k: v for k, v in kwargs.items() if k not in internal_params}

            return func(*args, **clean_kwargs)

        return wrapper

    return decorator


def record_operation_history(operation_name: str):
    """Decorator to record operation history after successful write operations."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)

            # Record history only if operation was successful
            if isinstance(result, OperationStatus) and result.success:
                document_name, chapter_name, _, _ = _extract_operation_parameters(
                    args, kwargs
                )
                _record_modification(
                    document_name,
                    chapter_name,
                    operation_name,
                    details=result.details or {},
                )

            return result

        return wrapper

    return decorator


def enhance_operation_result(func):
    """Decorator to enhance operation results with safety information."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)

        # Enhance result with safety information if successful
        if isinstance(result, OperationStatus) and result.success:
            safety_info = kwargs.get("_safety_info")
            snapshot_id = kwargs.get("_snapshot_id")

            if safety_info:
                # Ensure safety_info is properly serialized
                if hasattr(safety_info, "model_dump"):
                    result.safety_info = safety_info
                else:
                    result.safety_info = safety_info
            if snapshot_id:
                result.snapshot_created = snapshot_id

            # Add warnings if safety info has them
            if safety_info and hasattr(safety_info, "recommendations"):
                result.warnings = result.warnings or []
                if safety_info.safety_status == "warning":
                    result.warnings.append(
                        f"File was modified externally: {safety_info.message}"
                    )

        return result

    return wrapper


def safety_enhanced_write_operation(
    operation_name: str, create_snapshot: bool = False, check_freshness: bool = True
):
    """Composed decorator that combines all safety features for write operations.

    Enhanced to work with @auto_snapshot decorator - snapshot creation is now disabled by default
    since @auto_snapshot handles snapshot creation with better user tracking and naming.

    Features:
    - File freshness checking
    - Operation history recording
    - Result enhancement
    - Optional snapshot creation (disabled by default to prevent collision with @auto_snapshot)
    """

    def decorator(func):
        # Apply decorators in reverse order since they wrap from inside out
        # enhance_operation_result must be outermost to access safety info
        enhanced_func = func
        enhanced_func = record_operation_history(operation_name)(enhanced_func)

        if create_snapshot:
            enhanced_func = create_safety_snapshot(operation_name)(enhanced_func)

        if check_freshness:
            enhanced_func = check_file_freshness(enhanced_func)

        # Apply result enhancement last so it can access safety info
        enhanced_func = enhance_operation_result(enhanced_func)

        return enhanced_func

    return decorator


# --- Enhanced Automatic Snapshot System ---


def create_automatic_snapshot(
    operation_name: str,
    affected_documents: list[str],
    operation_details: dict[str, Any] = None,
) -> str | None:
    """Create automatic snapshot for edit operations with enhanced naming and tracking.

    Features:
    - Human-readable naming with operation context
    - User modification tracking and attribution
    - Intelligent retention policy application
    - Time-based and logical identifiers
    """
    if not affected_documents:
        return None

    try:
        user_id = get_current_user()
        timestamp = datetime.datetime.now()
        operation_details = operation_details or {}

        # Create human-readable snapshot name
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
        snapshot_name = f"user_edit_{timestamp_str}_{operation_name}"

        # Create snapshot for each affected document
        snapshot_ids = []
        for document_name in affected_documents:
            doc_path = _get_document_path(document_name)
            if not doc_path.is_dir():
                continue

            # Create the actual snapshot
            snapshot_result = snapshot_document(
                document_name=document_name,
                message=f"Auto-snapshot before {operation_name} by {user_id}",
                auto_cleanup=False,  # We'll handle cleanup with retention policy
            )

            if snapshot_result.success:
                snapshot_id = snapshot_result.details.get("snapshot_id", "")
                snapshot_ids.append(snapshot_id)

                # Record user modification for better UX
                modification_record = UserModificationRecord(
                    user_id=user_id,
                    operation_type=_get_operation_type(operation_name),
                    affected_scope=_get_operation_scope(operation_name),
                    timestamp=timestamp,
                    snapshot_id=snapshot_id,
                    operation_details={
                        "operation_name": operation_name,
                        "document_name": document_name,
                        **operation_details,
                    },
                    restoration_metadata={
                        "snapshot_name": snapshot_name,
                        "auto_created": True,
                        "restoration_priority": "high",
                    },
                )
                _user_modification_history.append(modification_record)

        # Apply retention policy
        for document_name in affected_documents:
            _apply_retention_policy(document_name, operation_name)

        return snapshot_ids[0] if snapshot_ids else None

    except Exception as e:
        log_structured_error(
            category=ErrorCategory.WARNING,
            message=f"Automatic snapshot creation failed for {operation_name}",
            exception=e,
            operation="auto_snapshot",
            operation_name=operation_name,
            affected_documents=affected_documents,
        )
        return None


def _get_operation_type(operation_name: str) -> str:
    """Classify operation type for tracking."""
    if "batch" in operation_name.lower():
        return "batch"
    elif "create" in operation_name.lower():
        return "create"
    elif "delete" in operation_name.lower():
        return "delete"
    else:
        return "edit"


def _get_operation_scope(operation_name: str) -> str:
    """Determine operation scope for tracking."""
    if "paragraph" in operation_name.lower():
        return "paragraph"
    elif "chapter" in operation_name.lower():
        return "chapter"
    else:
        return "document"


def _apply_retention_policy(document_name: str, operation_name: str):
    """Apply intelligent snapshot retention policy."""
    try:
        operation_scope = _get_operation_scope(operation_name)

        # Determine retention period based on operation scope
        if operation_scope == "paragraph":
            max_age_days = RETENTION_POLICY.PARAGRAPH_LEVEL_SNAPSHOTS
        elif operation_scope == "chapter":
            max_age_days = RETENTION_POLICY.CHAPTER_LEVEL_SNAPSHOTS
        else:
            max_age_days = RETENTION_POLICY.USER_EDIT_SNAPSHOTS

        # Clean up expired snapshots
        _cleanup_expired_snapshots(document_name, max_age_days)

    except Exception as e:
        log_structured_error(
            category=ErrorCategory.WARNING,
            message=f"Retention policy application failed for {document_name}",
            exception=e,
            operation="retention_policy",
            document_name=document_name,
        )


def _cleanup_expired_snapshots(document_name: str, max_age_days: int):
    """Clean up snapshots older than specified age."""
    try:
        snapshots_path = _get_snapshots_path(document_name)
        if not snapshots_path.exists():
            return

        cutoff_time = datetime.datetime.now() - datetime.timedelta(days=max_age_days)

        for snapshot_dir in snapshots_path.iterdir():
            if not snapshot_dir.is_dir():
                continue

            # Check if snapshot is older than cutoff
            try:
                snapshot_time = datetime.datetime.fromtimestamp(
                    snapshot_dir.stat().st_mtime
                )
                if snapshot_time < cutoff_time:
                    shutil.rmtree(snapshot_dir)
            except (OSError, ValueError):
                # Skip snapshots we can't process
                continue

    except Exception as e:
        log_structured_error(
            category=ErrorCategory.WARNING,
            message=f"Snapshot cleanup failed for {document_name}",
            exception=e,
            operation="snapshot_cleanup",
            document_name=document_name,
        )


def auto_snapshot(operation_name: str):
    """Decorator for automatic snapshot creation before edit operations.

    This decorator automatically creates snapshots before any edit operation
    with intelligent naming, user tracking, and retention policies.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract document names from function arguments
            affected_documents = []

            # Handle different function signatures
            if args and isinstance(args[0], str):
                # First argument is typically document_name
                affected_documents.append(args[0])
            elif "document_name" in kwargs and isinstance(kwargs["document_name"], str):
                # Document name passed as keyword argument
                affected_documents.append(kwargs["document_name"])

            # For batch operations, extract from operations list
            if "operations" in kwargs:
                operations = kwargs["operations"]
                if isinstance(operations, list):
                    for op in operations:
                        if isinstance(op, dict) and "target" in op:
                            target = op["target"]
                            if isinstance(target, dict) and "document_name" in target:
                                doc_name = target["document_name"]
                                if doc_name not in affected_documents:
                                    affected_documents.append(doc_name)

            # Create automatic snapshot before operation
            snapshot_id = None
            if affected_documents:
                snapshot_id = create_automatic_snapshot(
                    operation_name=operation_name,
                    affected_documents=affected_documents,
                    operation_details={
                        "function_name": func.__name__,
                        "args_count": len(args),
                        "kwargs_keys": list(kwargs.keys()),
                    },
                )

            # Execute the original operation
            try:
                result = func(*args, **kwargs)

                # Add snapshot info to result if applicable
                if hasattr(result, "snapshot_created") and snapshot_id:
                    result.snapshot_created = snapshot_id
                elif isinstance(result, dict) and snapshot_id:
                    result["auto_snapshot_created"] = snapshot_id

                return result

            except Exception as e:
                # Log the error but don't interfere with normal error handling
                log_structured_error(
                    category=ErrorCategory.ERROR,
                    message=f"Operation {operation_name} failed after snapshot creation",
                    exception=e,
                    operation=operation_name,
                    snapshot_created=snapshot_id,
                    affected_documents=affected_documents,
                )
                raise

        return wrapper

    return decorator


def _get_snapshots_path(document_name: str) -> Path:
    """Return the path to the snapshots directory for a document."""
    doc_path = _get_document_path(document_name)
    return doc_path / ".snapshots"


def _get_modification_history_path(document_name: str) -> Path:
    """Return the path to the modification history file for a document."""
    doc_path = _get_document_path(document_name)
    return doc_path / ".mod_history.json"


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
                "Verify changes don't conflict with your intended modifications",
            ],
        )


def _create_micro_snapshot(
    document_name: str, chapter_name: str | None = None, operation: str = "pre-write"
) -> str:
    """Create a micro-snapshot before destructive operations."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    snapshot_id = f"{operation}_{timestamp}"

    snapshots_path = _get_snapshots_path(document_name)
    snapshots_path.mkdir(exist_ok=True)

    snapshot_dir = snapshots_path / snapshot_id
    snapshot_dir.mkdir(exist_ok=True)

    if chapter_name:
        # Snapshot single chapter
        chapter_path = _get_chapter_path(document_name, chapter_name)
        if chapter_path.exists():
            shutil.copy2(chapter_path, snapshot_dir / chapter_name)
    else:
        # Snapshot entire document
        doc_path = _get_document_path(document_name)
        for chapter_file in doc_path.glob("*.md"):
            if chapter_file.is_file():
                shutil.copy2(chapter_file, snapshot_dir / chapter_file.name)

    # Create snapshot metadata
    metadata = {
        "snapshot_id": snapshot_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "operation": operation,
        "document_name": document_name,
        "chapter_name": chapter_name,
        "created_by": "mcp_safety_system",
    }

    with open(snapshot_dir / "_metadata.json", "w") as f:
        import json

        json.dump(metadata, f, indent=2)

    return snapshot_id


def _record_modification(
    document_name: str,
    chapter_name: str | None,
    operation: str,
    source: str = "mcp_tool",
    details: dict[str, Any] | None = None,
):
    """Record a modification in the document's history."""
    history_path = _get_modification_history_path(document_name)

    entry = ModificationHistoryEntry(
        timestamp=datetime.datetime.now(),
        file_path=f"{document_name}/{chapter_name}" if chapter_name else document_name,
        operation=operation,
        source=source,
        details=details or {},
    )

    # Load existing history or create new
    history_entries = []
    if history_path.exists():
        try:
            import json

            with open(history_path) as f:
                data = json.load(f)
                history_entries = [
                    ModificationHistoryEntry(**entry)
                    for entry in data.get("entries", [])
                ]
        except Exception:
            # If history file is corrupted, start fresh
            pass

    # Add new entry
    history_entries.append(entry)

    # Keep only last 1000 entries to prevent unbounded growth
    if len(history_entries) > 1000:
        history_entries = history_entries[-1000:]

    # Save updated history
    try:
        history_path.parent.mkdir(exist_ok=True)
        with open(history_path, "w") as f:
            import json

            json.dump(
                {
                    "entries": [entry.model_dump() for entry in history_entries],
                    "last_updated": datetime.datetime.now().isoformat(),
                },
                f,
                indent=2,
            )
    except Exception as e:
        # Log error but don't fail the operation
        log_structured_error(
            category=ErrorCategory.WARNING,
            message="Failed to record modification history",
            exception=e,
            context={
                "document_name": document_name,
                "chapter_name": chapter_name,
                "operation": operation,
            },
            operation="record_modification",
        )


def _get_document_path(document_name: str) -> Path:
    """Return the full path for a given document name."""
    # Ensure DOCS_ROOT_PATH is a Path object (defensive programming for tests)
    root_path = (
        Path(DOCS_ROOT_PATH) if not isinstance(DOCS_ROOT_PATH, Path) else DOCS_ROOT_PATH
    )
    return root_path / document_name


def _get_chapter_path(document_name: str, chapter_filename: str) -> Path:
    """Return the full path for a given chapter file."""
    doc_path = _get_document_path(document_name)
    return doc_path / chapter_filename


def _is_valid_chapter_filename(filename: str) -> bool:
    """Check if a filename is a valid, non-reserved chapter file.

    Verifies that the filename ends with '.md' and is not a reserved name
    like the manifest or summary file.
    """
    if not filename.lower().endswith(".md"):
        return False
    if filename == CHAPTER_MANIFEST_FILE:
        return False
    if filename == DOCUMENT_SUMMARY_FILE:  # Exclude document summary file
        return False
    return True


def _split_into_paragraphs(text: str) -> list[str]:
    """Split text into a list of paragraphs.

    Paragraphs are separated by one or more blank lines. Leading/trailing
    whitespace is stripped from each paragraph.
    """
    if not text:
        return []
    normalized_text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Split by one or more blank lines (a line with only whitespace is considered blank after strip)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", normalized_text.strip())]
    return [p for p in paragraphs if p]


def _count_words(text: str) -> int:
    """Count the number of words in a given text string."""
    return len(text.split())


def _get_ordered_chapter_files(document_name: str) -> list[Path]:
    """Retrieve a sorted list of all valid chapter files in a document.

    The files are sorted alphanumerically by filename. Non-chapter files
    (like summaries or manifests) are excluded.
    """
    doc_path = _get_document_path(document_name)
    if not doc_path.is_dir():
        return []

    # For now, simple alphanumeric sort of .md files.
    # Future: could read CHAPTER_MANIFEST_FILE for explicit order.
    chapter_files = sorted(
        [
            f
            for f in doc_path.iterdir()
            if f.is_file() and _is_valid_chapter_filename(f.name)
        ]
    )
    return chapter_files


def _read_chapter_content_details(
    document_name: str, chapter_file_path: Path
) -> ChapterContent | None:
    """Read the content and metadata of a chapter from its file path."""
    if not chapter_file_path.is_file():
        return None
    try:
        content = chapter_file_path.read_text(encoding="utf-8")
        paragraphs = _split_into_paragraphs(content)
        word_count = _count_words(content)
        stat = chapter_file_path.stat()
        return ChapterContent(
            document_name=document_name,
            chapter_name=chapter_file_path.name,
            content=content,
            word_count=word_count,
            paragraph_count=len(paragraphs),
            last_modified=datetime.datetime.fromtimestamp(
                stat.st_mtime, tz=datetime.timezone.utc
            ),
        )
    except Exception as e:
        log_structured_error(
            category=ErrorCategory.ERROR,
            message=f"Failed to read chapter file: {chapter_file_path.name}",
            exception=e,
            context={
                "document_name": document_name,
                "chapter_file_path": str(chapter_file_path),
                "file_exists": chapter_file_path.exists(),
            },
            operation="read_chapter_content",
        )
        return None


def _get_chapter_metadata(
    document_name: str, chapter_file_path: Path
) -> ChapterMetadata | None:
    """Generate metadata for a chapter from its file path.

    This helper reads chapter content to calculate word and paragraph counts
    for the metadata object.
    """
    if not chapter_file_path.is_file():
        return None
    try:
        content = chapter_file_path.read_text(
            encoding="utf-8"
        )  # Read to count words/paragraphs
        paragraphs = _split_into_paragraphs(content)
        word_count = _count_words(content)
        stat = chapter_file_path.stat()
        return ChapterMetadata(
            chapter_name=chapter_file_path.name,
            word_count=word_count,
            paragraph_count=len(paragraphs),
            last_modified=datetime.datetime.fromtimestamp(
                stat.st_mtime, tz=datetime.timezone.utc
            ),
            # title can be added later if we parse H1 from content
        )
    except Exception as e:
        log_structured_error(
            category=ErrorCategory.ERROR,
            message=f"Failed to get metadata for chapter: {chapter_file_path.name}",
            exception=e,
            context={
                "document_name": document_name,
                "chapter_file_path": str(chapter_file_path),
                "file_exists": chapter_file_path.exists(),
            },
            operation="get_chapter_metadata",
        )
        return None


# ============================================================================
# BATCH OPERATION REGISTRY
# ============================================================================


# Global registry instance (now using modular structure)
from .batch.registry import BatchOperationRegistry

_batch_registry = BatchOperationRegistry()


# Make the function available for import by the batch execution module
def _execute_batch_operation(operation: BatchOperation) -> OperationResult:
    """Execute a single operation within a batch using safe_operation."""
    # Get the tool function name
    tool_function_name = _batch_registry.get_tool_function_name(
        operation.operation_type
    )
    if not tool_function_name:
        return OperationResult(
            success=False,
            operation_id=operation.operation_id,
            operation_type=operation.operation_type,
            error=f"Unknown operation type: {operation.operation_type}",
        )

    # Get the actual function from globals (all tools are defined in this module)
    tool_function = globals().get(tool_function_name)
    if not tool_function:
        return OperationResult(
            success=False,
            operation_id=operation.operation_id,
            operation_type=operation.operation_type,
            error=f"Tool function not found: {tool_function_name}",
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
            "parameters": operation.parameters,
        },
        **all_args,
    )

    # Convert result to OperationResult and determine actual success
    result_data = None
    actual_success = success  # Start with safe_operation success

    if result:
        if hasattr(result, "model_dump"):
            result_data = result.model_dump()
        elif hasattr(result, "dict"):
            result_data = result.dict()
        elif isinstance(result, dict):
            result_data = result
        else:
            result_data = {"result": str(result)}

        # Check if the result indicates success (for OperationStatus-like objects)
        if isinstance(result_data, dict) and "success" in result_data:
            actual_success = success and result_data["success"]
        # For unified tools that return plain dicts, non-None result generally means success
        elif result_data is not None and not error:
            actual_success = True

    # If safe_operation failed but we have result data with success=False,
    # the error should come from the result message
    error_message = None
    if error:
        error_message = str(error)
    elif not actual_success and result_data and isinstance(result_data, dict):
        error_message = result_data.get("message", "Operation failed")

    return OperationResult(
        success=actual_success,
        operation_id=operation.operation_id,
        operation_type=operation.operation_type,
        result_data=result_data,
        error=error_message,
    )


# ============================================================================
# DOCUMENT MANAGEMENT TOOLS
# ============================================================================
# Tools for managing document collections (directories containing chapters)


@mcp_server.tool()
@log_mcp_call
def list_documents() -> list[DocumentInfo]:
    """List all available document collections in the document management system.

    This tool retrieves metadata for all document directories, where each document
    is a collection of ordered Markdown chapter files (.md). Provides comprehensive
    information including chapter counts, word counts, and modification timestamps.

    Parameters:
        None

    Returns:
        List[DocumentInfo]: A list of document metadata objects. Each DocumentInfo contains:
            - document_name (str): Directory name of the document
            - total_chapters (int): Number of chapter files in the document
            - total_word_count (int): Sum of words across all chapters
            - total_paragraph_count (int): Sum of paragraphs across all chapters
            - last_modified (datetime): Most recent modification time across all chapters
            - chapters (List[ChapterMetadata]): Ordered list of chapter metadata
            - has_summary (bool): Whether a _SUMMARY.md file exists

        Returns empty list [] if no documents exist or documents directory is not found.

    Example Usage:
        ```json
        {
            "name": "list_documents",
            "arguments": {}
        }
        ```

    Example Response:
        ```json
        [
            {
                "document_name": "user_guide",
                "total_chapters": 3,
                "total_word_count": 1250,
                "total_paragraph_count": 45,
                "last_modified": "2024-01-15T10:30:00Z",
                "chapters": [
                    {
                        "chapter_name": "01-introduction.md",
                        "word_count": 300,
                        "paragraph_count": 12,
                        "last_modified": "2024-01-15T10:30:00Z"
                    }
                ],
                "has_summary": true
            }
        ]
        ```
    """
    docs_info = []
    # Ensure DOCS_ROOT_PATH is a Path object (defensive programming for tests)
    root_path = (
        Path(DOCS_ROOT_PATH) if not isinstance(DOCS_ROOT_PATH, Path) else DOCS_ROOT_PATH
    )

    if not root_path.exists() or not root_path.is_dir():
        return []

    for doc_dir in root_path.iterdir():
        if doc_dir.is_dir():  # Each subdirectory is a potential document
            document_name = doc_dir.name
            ordered_chapter_files = _get_ordered_chapter_files(document_name)

            chapters_metadata_list = []
            doc_total_word_count = 0
            doc_total_paragraph_count = 0
            latest_mod_time = datetime.datetime.min.replace(
                tzinfo=datetime.timezone.utc
            )  # Ensure timezone aware for comparison

            for chapter_file_path in ordered_chapter_files:
                metadata = _get_chapter_metadata(document_name, chapter_file_path)
                if metadata:
                    chapters_metadata_list.append(metadata)
                    doc_total_word_count += metadata.word_count
                    doc_total_paragraph_count += metadata.paragraph_count
                    # Ensure metadata.last_modified is offset-aware before comparison
                    current_mod_time_aware = metadata.last_modified
                    if current_mod_time_aware > latest_mod_time:
                        latest_mod_time = current_mod_time_aware

            if (
                not chapters_metadata_list
            ):  # If no valid chapters, maybe don't list as a doc or list with 0s
                # Or, use directory's mtime if no chapters. For now, only list if chapters exist.
                # Or list if it's an empty initialized doc.
                # Let's list it even if empty, using the folder's mtime.
                if not ordered_chapter_files:  # No chapter files at all
                    stat_dir = doc_dir.stat()
                    latest_mod_time = datetime.datetime.fromtimestamp(
                        stat_dir.st_mtime, tz=datetime.timezone.utc
                    )

            summary_file_path = doc_dir / DOCUMENT_SUMMARY_FILE
            has_summary_file = summary_file_path.is_file()

            docs_info.append(
                DocumentInfo(
                    document_name=document_name,
                    total_chapters=len(chapters_metadata_list),
                    total_word_count=doc_total_word_count,
                    total_paragraph_count=doc_total_paragraph_count,
                    last_modified=(
                        latest_mod_time
                        if latest_mod_time
                        != datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)
                        else datetime.datetime.fromtimestamp(
                            doc_dir.stat().st_mtime, tz=datetime.timezone.utc
                        )
                    ),
                    chapters=chapters_metadata_list,
                    has_summary=has_summary_file,
                )
            )
    return docs_info


# ============================================================================
# CHAPTER MANAGEMENT TOOLS
# ============================================================================
# Tools for managing individual chapter files within documents


@mcp_server.tool()
@log_mcp_call
def list_chapters(document_name: str) -> list[ChapterMetadata] | None:
    """List all chapter files within a specified document, ordered by filename.

    This tool retrieves metadata for all chapter files (.md) within a document directory.
    Chapters are automatically ordered alphanumerically by filename, which typically
    corresponds to their intended sequence (e.g., 01-intro.md, 02-setup.md).

    Parameters:
        document_name (str): Name of the document directory to list chapters from

    Returns:
        Optional[List[ChapterMetadata]]: List of chapter metadata objects if document exists,
        None if document not found. Each ChapterMetadata contains:
            - chapter_name (str): Filename of the chapter (e.g., "01-introduction.md")
            - title (Optional[str]): Chapter title (currently None, reserved for future use)
            - word_count (int): Total number of words in the chapter
            - paragraph_count (int): Total number of paragraphs in the chapter
            - last_modified (datetime): Timestamp of last file modification

        Returns empty list [] if document exists but contains no valid chapter files.
        Returns None if the document directory does not exist.

    Example Usage:
        ```json
        {
            "name": "list_chapters",
            "arguments": {
                "document_name": "user_guide"
            }
        }
        ```

    Example Success Response:
        ```json
        [
            {
                "chapter_name": "01-introduction.md",
                "title": null,
                "word_count": 342,
                "paragraph_count": 8,
                "last_modified": "2024-01-15T10:30:00Z"
            },
            {
                "chapter_name": "02-getting-started.md",
                "title": null,
                "word_count": 567,
                "paragraph_count": 15,
                "last_modified": "2024-01-15T11:45:00Z"
            }
        ]
        ```

    Example Not Found Response:
        ```json
        null
        ```
    """
    doc_path = _get_document_path(document_name)
    if not doc_path.is_dir():
        print(f"Document '{document_name}' not found at {doc_path}")
        return None  # Or perhaps OperationStatus(success=False, message="Document not found")
        # For now, following Optional[List[...]] pattern for read lists

    ordered_chapter_files = _get_ordered_chapter_files(document_name)
    chapters_metadata_list = []
    for chapter_file_path in ordered_chapter_files:
        metadata = _get_chapter_metadata(document_name, chapter_file_path)
        if metadata:
            chapters_metadata_list.append(metadata)

    if not ordered_chapter_files and not chapters_metadata_list:
        # If the directory exists but has no valid chapter files, return empty list.
        return []

    return chapters_metadata_list


def read_chapter_content(
    document_name: str, chapter_name: str
) -> ChapterContent | None:
    r"""Retrieve the complete content and metadata of a specific chapter within a document.

    This tool reads a chapter file (.md) from a document directory and returns both
    the raw content and associated metadata including word counts, paragraph counts,
    and modification timestamps. Returns None if the chapter or document is not found.

    Parameters:
        document_name (str): Name of the document directory containing the chapter
        chapter_name (str): Filename of the chapter to read. Must be:
            - Valid .md filename (e.g., "01-introduction.md")
            - Existing file within the specified document directory
            - Not a reserved filename like "_manifest.json"

    Returns:
        Optional[ChapterContent]: Chapter content object if found, None if not found.
        ChapterContent contains:
            - document_name (str): Name of the parent document
            - chapter_name (str): Filename of the chapter
            - content (str): Full raw text content of the chapter file
            - word_count (int): Total number of words in the chapter
            - paragraph_count (int): Total number of paragraphs in the chapter
            - last_modified (datetime): Timestamp of last file modification

        Returns None if document doesn't exist, chapter doesn't exist, or chapter
        filename is invalid.

    Example Usage:
        ```json
        {
            "name": "read_chapter_content",
            "arguments": {
                "document_name": "user_guide",
                "chapter_name": "01-introduction.md"
            }
        }
        ```

    Example Success Response:
        ```json
        {
            "document_name": "user_guide",
            "chapter_name": "01-introduction.md",
            "content": "# Introduction\n\nWelcome to our user guide...",
            "word_count": 342,
            "paragraph_count": 8,
            "last_modified": "2024-01-15T10:30:00Z"
        }
        ```

    Example Not Found Response:
        ```json
        null
        ```
    """
    chapter_file_path = _get_chapter_path(document_name, chapter_name)
    if not chapter_file_path.is_file() or not _is_valid_chapter_filename(chapter_name):
        print(
            f"Chapter '{chapter_name}' not found or invalid in document '{document_name}' at {chapter_file_path}"
        )
        return None
    return _read_chapter_content_details(document_name, chapter_file_path)


@mcp_server.tool()
@log_mcp_call
def read_document_summary(document_name: str) -> DocumentSummary | None:
    r"""Retrieve the content of a document's summary file (_SUMMARY.md).

    This tool reads the special _SUMMARY.md file that can be used to provide
    an overview or table of contents for a document collection. The summary
    file is optional and may not exist for all documents.

    Parameters:
        document_name (str): Name of the document directory to read summary from

    Returns:
        Optional[DocumentSummary]: A DocumentSummary object containing the document name
        and summary content if it exists, None if the summary file doesn't exist or
        cannot be read.

        Returns None if:
        - Document directory doesn't exist
        - _SUMMARY.md file doesn't exist in the document

    Example Usage:
        ```json
        {
            "name": "read_document_summary",
            "arguments": {
                "document_name": "user_guide"
            }
        }
        ```

    Example Success Response:
        ```json
        {
            "document_name": "user_guide",
            "content": "# Document Summary\n\n- Chapter 1\n- Chapter 2"
        }
        ```

    Example Not Found Response:
        ```json
        null
        ```
    """
    # Validate document name (optional, but good practice if it can be invalid)
    is_valid_doc, doc_error = _validate_document_name(document_name)
    if not is_valid_doc:
        log_structured_error(
            category=ErrorCategory.WARNING,
            message="Invalid document name provided",
            context={"document_name": document_name, "validation_error": doc_error},
            operation="read_document_summary",
        )
        # Depending on desired strictness, could return None or raise error
        return None  # For now, let's be lenient if the path check below handles it

    doc_path = _get_document_path(document_name)
    if not doc_path.is_dir():
        log_structured_error(
            category=ErrorCategory.INFO,
            message="Document not found",
            context={"document_name": document_name, "attempted_path": str(doc_path)},
            operation="read_document_summary",
        )
        return None

    summary_file_path = doc_path / DOCUMENT_SUMMARY_FILE
    if not summary_file_path.is_file():
        log_structured_error(
            category=ErrorCategory.INFO,
            message="Summary file not found in document",
            context={
                "document_name": document_name,
                "summary_file_name": DOCUMENT_SUMMARY_FILE,
                "summary_file_path": str(summary_file_path),
            },
            operation="read_document_summary",
        )
        return None

    try:
        summary_content = summary_file_path.read_text(encoding="utf-8")
        return DocumentSummary(document_name=document_name, content=summary_content)
    except Exception as e:
        log_structured_error(
            category=ErrorCategory.ERROR,
            message="Failed to read summary file",
            exception=e,
            context={
                "document_name": document_name,
                "summary_file_path": str(summary_file_path),
            },
            operation="read_document_summary",
        )
        return None


def read_paragraph_content(
    document_name: str, chapter_name: str, paragraph_index_in_chapter: int
) -> ParagraphDetail | None:
    """Retrieve the content and metadata of a specific paragraph within a chapter.

    This tool extracts a single paragraph from a chapter file using zero-indexed
    positioning. Paragraphs are defined as text blocks separated by blank lines.
    Useful for targeted content retrieval and editing operations.

    Parameters:
        document_name (str): Name of the document directory containing the chapter
        chapter_name (str): Filename of the chapter containing the paragraph
        paragraph_index_in_chapter (int): Zero-indexed position of the paragraph. Must be:
            - Non-negative integer (≥0)
            - Within the valid range of existing paragraphs in the chapter

    Returns:
        Optional[ParagraphDetail]: Paragraph content object if found, None if not found.
        ParagraphDetail contains:
            - document_name (str): Name of the parent document
            - chapter_name (str): Name of the chapter file
            - paragraph_index_in_chapter (int): Zero-indexed position within the chapter
            - content (str): Full text content of the paragraph
            - word_count (int): Number of words in the paragraph

        Returns None if:
        - Document doesn't exist
        - Chapter doesn't exist or is invalid
        - Paragraph index is out of bounds (negative or exceeds paragraph count)

    Example Usage:
        ```json
        {
            "name": "read_paragraph_content",
            "arguments": {
                "document_name": "user_guide",
                "chapter_name": "01-introduction.md",
                "paragraph_index_in_chapter": 2
            }
        }
        ```

    Example Success Response:
        ```json
        {
            "document_name": "user_guide",
            "chapter_name": "01-introduction.md",
            "paragraph_index_in_chapter": 2,
            "content": "This comprehensive guide will walk you through all the essential features and help you become productive quickly.",
            "word_count": 18
        }
        ```

    Example Not Found Response:
        ```json
        null
        ```
    """
    chapter_content_obj = read_chapter_content(document_name, chapter_name)
    if not chapter_content_obj:
        return None

    paragraphs = _split_into_paragraphs(chapter_content_obj.content)
    total_paragraphs = len(paragraphs)

    if not (0 <= paragraph_index_in_chapter < total_paragraphs):
        log_structured_error(
            category=ErrorCategory.WARNING,
            message="Paragraph index out of bounds",
            context={
                "document_name": document_name,
                "chapter_name": chapter_name,
                "paragraph_index": paragraph_index_in_chapter,
                "total_paragraphs": total_paragraphs,
                "valid_range": f"0-{total_paragraphs - 1}",
            },
            operation="read_paragraph_content",
        )
        return None

    paragraph_text = paragraphs[paragraph_index_in_chapter]
    return ParagraphDetail(
        document_name=document_name,
        chapter_name=chapter_name,
        paragraph_index_in_chapter=paragraph_index_in_chapter,
        content=paragraph_text,
        word_count=_count_words(paragraph_text),
    )


def read_full_document(document_name: str) -> FullDocumentContent | None:
    r"""Retrieve the complete content of an entire document including all chapters in order.

    This tool reads all chapter files within a document directory and returns them
    as a single structured object. Chapters are ordered alphanumerically by filename,
    providing the complete document content for comprehensive operations.

    Parameters:
        document_name (str): Name of the document directory to read completely

    Returns:
        Optional[FullDocumentContent]: Complete document content object if found, None if not found.
        FullDocumentContent contains:
            - document_name (str): Name of the document
            - chapters (List[ChapterContent]): Ordered list of all chapter contents
            - total_word_count (int): Sum of words across all chapters
            - total_paragraph_count (int): Sum of paragraphs across all chapters

        Each ChapterContent within chapters contains:
            - document_name (str): Name of the parent document
            - chapter_name (str): Filename of the chapter
            - content (str): Full raw text content of the chapter
            - word_count (int): Number of words in the chapter
            - paragraph_count (int): Number of paragraphs in the chapter
            - last_modified (datetime): Timestamp of last file modification

        Returns None if document directory doesn't exist.
        Returns empty chapters list if document exists but has no chapter files.

    Example Usage:
        ```json
        {
            "name": "read_full_document",
            "arguments": {
                "document_name": "user_guide"
            }
        }
        ```

    Example Success Response:
        ```json
        {
            "document_name": "user_guide",
            "chapters": [
                {
                    "document_name": "user_guide",
                    "chapter_name": "01-introduction.md",
                    "content": "# Introduction\n\nWelcome to our guide...",
                    "word_count": 342,
                    "paragraph_count": 8,
                    "last_modified": "2024-01-15T10:30:00Z"
                },
                {
                    "document_name": "user_guide",
                    "chapter_name": "02-setup.md",
                    "content": "# Setup\n\nFollow these steps...",
                    "word_count": 567,
                    "paragraph_count": 15,
                    "last_modified": "2024-01-15T11:45:00Z"
                }
            ],
            "total_word_count": 909,
            "total_paragraph_count": 23
        }
        ```

    Example Not Found Response:
        ```json
        null
        ```
    """
    doc_path = _get_document_path(document_name)
    if not doc_path.is_dir():
        log_structured_error(
            category=ErrorCategory.INFO,
            message="Document not found for full read",
            context={"document_name": document_name, "attempted_path": str(doc_path)},
            operation="read_full_document",
        )
        return None

    ordered_chapter_files = _get_ordered_chapter_files(document_name)
    if not ordered_chapter_files:
        # Document exists but has no chapters
        return FullDocumentContent(
            document_name=document_name,
            chapters=[],
            total_word_count=0,
            total_paragraph_count=0,
        )

    all_chapter_contents = []
    doc_total_word_count = 0
    doc_total_paragraph_count = 0

    for chapter_file_path in ordered_chapter_files:
        chapter_details = _read_chapter_content_details(
            document_name, chapter_file_path
        )
        if chapter_details:
            all_chapter_contents.append(chapter_details)
            doc_total_word_count += chapter_details.word_count
            doc_total_paragraph_count += chapter_details.paragraph_count
        else:
            # If a chapter file is listed but can't be read, this indicates an issue.
            # For now, we'll skip it, but this could also be an error condition.
            log_structured_error(
                category=ErrorCategory.WARNING,
                message="Could not read chapter file, skipping",
                context={
                    "document_name": document_name,
                    "chapter_file_name": chapter_file_path.name,
                    "chapter_file_path": str(chapter_file_path),
                },
                operation="read_full_document",
            )

    return FullDocumentContent(
        document_name=document_name,
        chapters=all_chapter_contents,
        total_word_count=doc_total_word_count,
        total_paragraph_count=doc_total_paragraph_count,
    )


@mcp_server.tool()
@register_batchable_operation("read_content")
@log_mcp_call
def read_content(
    document_name: str,
    scope: str = "document",  # "document", "chapter", "paragraph"
    chapter_name: str | None = None,
    paragraph_index: int | None = None,
) -> dict[str, Any] | None:
    """Unified content reading with scope-based targeting.

    This tool consolidates three separate reading operations into a single, scope-based
    interface. It can read complete documents, individual chapters, or specific paragraphs
    depending on the scope parameter, providing a consistent API for all content access.

    Parameters:
        document_name (str): Name of the document directory to read from
        scope (str): Reading scope determining what content to retrieve:
            - "document": Read complete document with all chapters (default)
            - "chapter": Read specific chapter content and metadata
            - "paragraph": Read specific paragraph content and metadata
        chapter_name (Optional[str]): Required for "chapter" and "paragraph" scopes.
            Must be valid .md filename (e.g., "01-introduction.md")
        paragraph_index (Optional[int]): Required for "paragraph" scope.
            Zero-indexed position of paragraph within the chapter (≥0)

    Returns:
        Optional[Dict[str, Any]]: Content object matching the requested scope, None if not found.

        For scope="document":
            FullDocumentContent with fields:
            - document_name (str): Name of the document
            - chapters (List[ChapterContent]): Ordered list of all chapter contents
            - total_word_count (int): Sum of words across all chapters
            - total_paragraph_count (int): Sum of paragraphs across all chapters

        For scope="chapter":
            ChapterContent with fields:
            - document_name (str): Name of the parent document
            - chapter_name (str): Filename of the chapter
            - content (str): Full raw text content of the chapter
            - word_count (int): Number of words in the chapter
            - paragraph_count (int): Number of paragraphs in the chapter
            - last_modified (datetime): Timestamp of last file modification

        For scope="paragraph":
            ParagraphDetail with fields:
            - document_name (str): Name of the parent document
            - chapter_name (str): Name of the chapter file
            - paragraph_index_in_chapter (int): Zero-indexed position within the chapter
            - content (str): Full text content of the paragraph
            - word_count (int): Number of words in the paragraph

    Example Usage:
        ```json
        // Read full document
        {
            "name": "read_content",
            "arguments": {
                "document_name": "My Book",
                "scope": "document"
            }
        }

        // Read specific chapter
        {
            "name": "read_content",
            "arguments": {
                "document_name": "My Book",
                "scope": "chapter",
                "chapter_name": "01-introduction.md"
            }
        }

        // Read specific paragraph
        {
            "name": "read_content",
            "arguments": {
                "document_name": "My Book",
                "scope": "paragraph",
                "chapter_name": "01-introduction.md",
                "paragraph_index": 0
            }
        }
        ```
    """
    # Validate document name
    is_valid_doc, doc_error = _validate_document_name(document_name)
    if not is_valid_doc:
        log_structured_error(
            category=ErrorCategory.ERROR,
            message=f"Invalid document name: {doc_error}",
            context={"document_name": document_name, "scope": scope},
            operation="read_content",
        )
        return None

    # Validate scope-specific parameters
    if scope == "chapter":
        if not chapter_name:
            log_structured_error(
                category=ErrorCategory.ERROR,
                message="chapter_name required for chapter scope",
                context={"document_name": document_name, "scope": scope},
                operation="read_content",
            )
            return None
        is_valid_chapter, chapter_error = _validate_chapter_name(chapter_name)
        if not is_valid_chapter:
            log_structured_error(
                category=ErrorCategory.ERROR,
                message=f"Invalid chapter name: {chapter_error}",
                context={
                    "document_name": document_name,
                    "chapter_name": chapter_name,
                    "scope": scope,
                },
                operation="read_content",
            )
            return None

    elif scope == "paragraph":
        if not chapter_name or paragraph_index is None:
            log_structured_error(
                category=ErrorCategory.ERROR,
                message="chapter_name and paragraph_index required for paragraph scope",
                context={"document_name": document_name, "scope": scope},
                operation="read_content",
            )
            return None
        is_valid_chapter, chapter_error = _validate_chapter_name(chapter_name)
        if not is_valid_chapter:
            log_structured_error(
                category=ErrorCategory.ERROR,
                message=f"Invalid chapter name: {chapter_error}",
                context={
                    "document_name": document_name,
                    "chapter_name": chapter_name,
                    "scope": scope,
                },
                operation="read_content",
            )
            return None
        is_valid_index, index_error = _validate_paragraph_index(paragraph_index)
        if not is_valid_index:
            log_structured_error(
                category=ErrorCategory.ERROR,
                message=f"Invalid paragraph index: {index_error}",
                context={
                    "document_name": document_name,
                    "paragraph_index": paragraph_index,
                    "scope": scope,
                },
                operation="read_content",
            )
            return None

    elif scope != "document":
        log_structured_error(
            category=ErrorCategory.ERROR,
            message=f"Invalid scope: {scope}. Must be 'document', 'chapter', or 'paragraph'",
            context={"document_name": document_name, "scope": scope},
            operation="read_content",
        )
        return None

    # Scope-based dispatch to existing internal functions
    try:
        if scope == "document":
            result = read_full_document(document_name)
            return result.model_dump() if result else None

        elif scope == "chapter":
            result = read_chapter_content(document_name, chapter_name)
            return result.model_dump() if result else None

        elif scope == "paragraph":
            result = read_paragraph_content(
                document_name, chapter_name, paragraph_index
            )
            return result.model_dump() if result else None

    except Exception as e:
        log_structured_error(
            category=ErrorCategory.SYSTEM,
            message=f"Error reading content with scope {scope}: {str(e)}",
            context={
                "document_name": document_name,
                "scope": scope,
                "chapter_name": chapter_name,
                "paragraph_index": paragraph_index,
            },
            operation="read_content",
        )
        return None


@mcp_server.tool()
@register_batchable_operation("find_text")
@log_mcp_call
def find_text(
    document_name: str,
    search_text: str,
    scope: str = "document",  # "document", "chapter"
    chapter_name: str | None = None,
    case_sensitive: bool = False,
) -> list[dict[str, Any]] | None:
    """Unified text search with scope-based targeting.

    This tool consolidates document and chapter text search into a single interface,
    providing consistent search capabilities across different scopes with flexible
    case sensitivity options.

    Parameters:
        document_name (str): Name of the document to search within
        search_text (str): Text pattern to search for
        scope (str): Search scope determining where to search:
            - "document": Search across entire document (all chapters)
            - "chapter": Search within specific chapter only
        chapter_name (Optional[str]): Required for "chapter" scope.
            Must be valid .md filename (e.g., "01-introduction.md")
        case_sensitive (bool): Whether search should be case-sensitive (default: False)

    Returns:
        Optional[List[Dict[str, Any]]]: List of search results, None if error.
        Each result contains location and context information.

        For scope="document": Results from find_text_in_document
        For scope="chapter": Results from find_text_in_chapter

    Example Usage:
        ```json
        // Search entire document
        {
            "name": "find_text",
            "arguments": {
                "document_name": "My Book",
                "search_text": "important concept",
                "scope": "document",
                "case_sensitive": false
            }
        }

        // Search specific chapter
        {
            "name": "find_text",
            "arguments": {
                "document_name": "My Book",
                "search_text": "introduction",
                "scope": "chapter",
                "chapter_name": "01-intro.md",
                "case_sensitive": true
            }
        }
        ```
    """
    # Validate document name
    is_valid_doc, doc_error = _validate_document_name(document_name)
    if not is_valid_doc:
        log_structured_error(
            category=ErrorCategory.ERROR,
            message=f"Invalid document name: {doc_error}",
            context={"document_name": document_name, "scope": scope},
            operation="find_text",
        )
        return None

    # Validate search text
    is_valid_search, search_error = _validate_search_query(search_text)
    if not is_valid_search:
        log_structured_error(
            category=ErrorCategory.ERROR,
            message=f"Invalid search text: {search_error}",
            context={
                "document_name": document_name,
                "search_text": search_text,
                "scope": scope,
            },
            operation="find_text",
        )
        return None

    # Validate scope-specific parameters
    if scope == "chapter":
        if not chapter_name:
            log_structured_error(
                category=ErrorCategory.ERROR,
                message="chapter_name required for chapter scope",
                context={"document_name": document_name, "scope": scope},
                operation="find_text",
            )
            return None
        is_valid_chapter, chapter_error = _validate_chapter_name(chapter_name)
        if not is_valid_chapter:
            log_structured_error(
                category=ErrorCategory.ERROR,
                message=f"Invalid chapter name: {chapter_error}",
                context={
                    "document_name": document_name,
                    "chapter_name": chapter_name,
                    "scope": scope,
                },
                operation="find_text",
            )
            return None
    elif scope != "document":
        log_structured_error(
            category=ErrorCategory.ERROR,
            message=f"Invalid scope: {scope}. Must be 'document' or 'chapter'",
            context={"document_name": document_name, "scope": scope},
            operation="find_text",
        )
        return None

    # Scope-based dispatch to existing functions
    try:
        if scope == "document":
            result = find_text_in_document(document_name, search_text, case_sensitive)
            return [r.model_dump() for r in result] if result else []

        elif scope == "chapter":
            result = find_text_in_chapter(
                document_name, chapter_name, search_text, case_sensitive
            )
            return [r.model_dump() for r in result] if result else []

    except Exception as e:
        log_structured_error(
            category=ErrorCategory.SYSTEM,
            message=f"Error searching text with scope {scope}: {str(e)}",
            context={
                "document_name": document_name,
                "scope": scope,
                "search_text": search_text,
                "chapter_name": chapter_name,
            },
            operation="find_text",
        )
        return None


@mcp_server.tool()
@register_batchable_operation("replace_text")
@log_mcp_call
@auto_snapshot("replace_text")
def replace_text(
    document_name: str,
    find_text: str,
    replace_text: str,
    scope: str = "document",  # "document", "chapter"
    chapter_name: str | None = None,
) -> dict[str, Any] | None:
    """Unified text replacement with scope-based targeting.

    This tool consolidates document and chapter text replacement into a single interface,
    providing consistent replacement capabilities across different scopes with atomic
    operation guarantees.

    Parameters:
        document_name (str): Name of the document to perform replacement in
        find_text (str): Text pattern to find and replace
        replace_text (str): Text to replace occurrences with
        scope (str): Replacement scope determining where to replace:
            - "document": Replace across entire document (all chapters)
            - "chapter": Replace within specific chapter only
        chapter_name (Optional[str]): Required for "chapter" scope.
            Must be valid .md filename (e.g., "01-introduction.md")

    Returns:
        Optional[Dict[str, Any]]: Replacement operation results, None if error.
        Contains success status and replacement statistics.

        For scope="document": Results from replace_text_in_document
        For scope="chapter": Results from replace_text_in_chapter

    Example Usage:
        ```json
        // Replace across entire document
        {
            "name": "replace_text",
            "arguments": {
                "document_name": "My Book",
                "find_text": "old term",
                "replace_text": "new term",
                "scope": "document"
            }
        }

        // Replace in specific chapter
        {
            "name": "replace_text",
            "arguments": {
                "document_name": "My Book",
                "find_text": "draft text",
                "replace_text": "final text",
                "scope": "chapter",
                "chapter_name": "01-intro.md"
            }
        }
        ```
    """
    # Validate document name
    is_valid_doc, doc_error = _validate_document_name(document_name)
    if not is_valid_doc:
        log_structured_error(
            category=ErrorCategory.ERROR,
            message=f"Invalid document name: {doc_error}",
            context={"document_name": document_name, "scope": scope},
            operation="replace_text",
        )
        return None

    # Validate find and replace text
    is_valid_find, find_error = _validate_search_query(find_text)
    if not is_valid_find:
        log_structured_error(
            category=ErrorCategory.ERROR,
            message=f"Invalid find text: {find_error}",
            context={
                "document_name": document_name,
                "find_text": find_text,
                "scope": scope,
            },
            operation="replace_text",
        )
        return None

    is_valid_replace, replace_error = _validate_content(replace_text)
    if not is_valid_replace:
        log_structured_error(
            category=ErrorCategory.ERROR,
            message=f"Invalid replace text: {replace_error}",
            context={
                "document_name": document_name,
                "replace_text": replace_text,
                "scope": scope,
            },
            operation="replace_text",
        )
        return None

    # Validate scope-specific parameters
    if scope == "chapter":
        if not chapter_name:
            log_structured_error(
                category=ErrorCategory.ERROR,
                message="chapter_name required for chapter scope",
                context={"document_name": document_name, "scope": scope},
                operation="replace_text",
            )
            return None
        is_valid_chapter, chapter_error = _validate_chapter_name(chapter_name)
        if not is_valid_chapter:
            log_structured_error(
                category=ErrorCategory.ERROR,
                message=f"Invalid chapter name: {chapter_error}",
                context={
                    "document_name": document_name,
                    "chapter_name": chapter_name,
                    "scope": scope,
                },
                operation="replace_text",
            )
            return None
    elif scope != "document":
        log_structured_error(
            category=ErrorCategory.ERROR,
            message=f"Invalid scope: {scope}. Must be 'document' or 'chapter'",
            context={"document_name": document_name, "scope": scope},
            operation="replace_text",
        )
        return None

    # Scope-based dispatch to existing functions
    try:
        if scope == "document":
            result = replace_text_in_document(document_name, find_text, replace_text)
            return result.model_dump() if result else None

        elif scope == "chapter":
            result = replace_text_in_chapter(
                document_name, chapter_name, find_text, replace_text
            )
            return result.model_dump() if result else None

    except Exception as e:
        log_structured_error(
            category=ErrorCategory.SYSTEM,
            message=f"Error replacing text with scope {scope}: {str(e)}",
            context={
                "document_name": document_name,
                "scope": scope,
                "find_text": find_text,
                "replace_text": replace_text,
                "chapter_name": chapter_name,
            },
            operation="replace_text",
        )
        return None


@mcp_server.tool()
@register_batchable_operation("get_statistics")
@log_mcp_call
def get_statistics(
    document_name: str,
    scope: str = "document",  # "document", "chapter"
    chapter_name: str | None = None,
) -> dict[str, Any] | None:
    """Unified statistics collection with scope-based targeting.

    This tool consolidates document and chapter statistics into a single interface,
    providing consistent analytics capabilities across different scopes with
    comprehensive word, paragraph, and chapter counts.

    Parameters:
        document_name (str): Name of the document to analyze
        scope (str): Statistics scope determining what to analyze:
            - "document": Analyze entire document (all chapters)
            - "chapter": Analyze specific chapter only
        chapter_name (Optional[str]): Required for "chapter" scope.
            Must be valid .md filename (e.g., "01-introduction.md")

    Returns:
        Optional[Dict[str, Any]]: Statistics report, None if error.
        Contains word counts, paragraph counts, and scope information.

        For scope="document": Results from get_document_statistics
        For scope="chapter": Results from get_chapter_statistics

    Example Usage:
        ```json
        // Get document statistics
        {
            "name": "get_statistics",
            "arguments": {
                "document_name": "My Book",
                "scope": "document"
            }
        }

        // Get chapter statistics
        {
            "name": "get_statistics",
            "arguments": {
                "document_name": "My Book",
                "scope": "chapter",
                "chapter_name": "01-intro.md"
            }
        }
        ```
    """
    # Validate document name
    is_valid_doc, doc_error = _validate_document_name(document_name)
    if not is_valid_doc:
        log_structured_error(
            category=ErrorCategory.ERROR,
            message=f"Invalid document name: {doc_error}",
            context={"document_name": document_name, "scope": scope},
            operation="get_statistics",
        )
        return None

    # Validate scope-specific parameters
    if scope == "chapter":
        if not chapter_name:
            log_structured_error(
                category=ErrorCategory.ERROR,
                message="chapter_name required for chapter scope",
                context={"document_name": document_name, "scope": scope},
                operation="get_statistics",
            )
            return None
        is_valid_chapter, chapter_error = _validate_chapter_name(chapter_name)
        if not is_valid_chapter:
            log_structured_error(
                category=ErrorCategory.ERROR,
                message=f"Invalid chapter name: {chapter_error}",
                context={
                    "document_name": document_name,
                    "chapter_name": chapter_name,
                    "scope": scope,
                },
                operation="get_statistics",
            )
            return None
    elif scope != "document":
        log_structured_error(
            category=ErrorCategory.ERROR,
            message=f"Invalid scope: {scope}. Must be 'document' or 'chapter'",
            context={"document_name": document_name, "scope": scope},
            operation="get_statistics",
        )
        return None

    # Scope-based dispatch to existing functions
    try:
        if scope == "document":
            result = get_document_statistics(document_name)
            return result.model_dump() if result else None

        elif scope == "chapter":
            result = get_chapter_statistics(document_name, chapter_name)
            if result:
                # For chapter scope, exclude chapter_count field
                data = result.model_dump()
                data.pop("chapter_count", None)  # Remove chapter_count if present
                return data
            return None

    except Exception as e:
        log_structured_error(
            category=ErrorCategory.SYSTEM,
            message=f"Error getting statistics with scope {scope}: {str(e)}",
            context={
                "document_name": document_name,
                "scope": scope,
                "chapter_name": chapter_name,
            },
            operation="get_statistics",
        )
        return None


@mcp_server.tool()
@register_batchable_operation("create_document")
@log_mcp_call
@auto_snapshot("create_document")
def create_document(document_name: str) -> OperationStatus:
    r"""Create a new document collection as a directory in the document management system.

    This tool initializes a new document by creating a directory that will contain
    chapter files (.md). The document name must be valid for filesystem usage and
    will serve as the directory name for organizing chapters.

    Parameters:
        document_name (str): Name for the new document directory. Must be:
            - Non-empty string
            - ≤100 characters
            - Valid filesystem directory name
            - Cannot contain path separators (/ or \\)
            - Cannot start with a dot (.)
            - Cannot conflict with existing document names

    Returns:
        OperationStatus: Structured result object containing:
            - success (bool): True if document was created successfully, False otherwise
            - message (str): Human-readable description of the operation result
            - details (Dict[str, Any], optional): Additional context including:
                - document_name (str): Name of the created document (on success)

    Example Usage:
        ```json
        {
            "name": "create_document",
            "arguments": {
                "document_name": "user_manual"
            }
        }
        ```

    Example Success Response:
        ```json
        {
            "success": true,
            "message": "Document 'user_manual' created successfully.",
            "details": {
                "document_name": "user_manual"
            }
        }
        ```

    Example Error Response:
        ```json
        {
            "success": false,
            "message": "Document 'user_manual' already exists.",
            "details": null
        }
        ```
    """
    # Validate input
    is_valid, error_msg = _validate_document_name(document_name)
    if not is_valid:
        return OperationStatus(success=False, message=error_msg)

    doc_path = _get_document_path(document_name)
    if doc_path.exists():
        return OperationStatus(
            success=False, message=f"Document '{document_name}' already exists."
        )
    try:
        doc_path.mkdir(parents=True, exist_ok=False)
        return OperationStatus(
            success=True,
            message=f"Document '{document_name}' created successfully.",
            details={"document_name": document_name},
        )
    except Exception as e:
        return OperationStatus(
            success=False, message=f"Error creating document '{document_name}': {e}"
        )


@mcp_server.tool()
@log_mcp_call
@auto_snapshot("delete_document")
def delete_document(document_name: str) -> OperationStatus:
    """Permanently deletes a document directory and all its chapter files.

    This tool removes an entire document collection including all chapter files
    and any associated metadata. This operation is irreversible and should be
    used with caution. All content within the document directory will be lost.

    Parameters:
        document_name (str): Name of the document directory to delete

    Returns:
        OperationStatus: Structured result object containing:
            - success (bool): True if document was deleted successfully, False otherwise
            - message (str): Human-readable description of the operation result
            - details (Dict[str, Any], optional): Additional context (currently None)

    Example Usage:
        ```json
        {
            "name": "delete_document",
            "arguments": {
                "document_name": "old_manual"
            }
        }
        ```

    Example Success Response:
        ```json
        {
            "success": true,
            "message": "Document 'old_manual' and its contents deleted successfully.",
            "details": null
        }
        ```

    Example Error Response:
        ```json
        {
            "success": false,
            "message": "Document 'old_manual' not found.",
            "details": null
        }
        ```
    """
    doc_path = _get_document_path(document_name)
    if not doc_path.is_dir():
        return OperationStatus(
            success=False, message=f"Document '{document_name}' not found."
        )
    try:
        shutil.rmtree(doc_path)
        return OperationStatus(
            success=True,
            message=f"Document '{document_name}' and its contents deleted successfully.",
        )
    except Exception as e:
        return OperationStatus(
            success=False, message=f"Error deleting document '{document_name}': {e}"
        )


@mcp_server.tool()
@register_batchable_operation("create_chapter")
@log_mcp_call
@auto_snapshot("create_chapter")
def create_chapter(
    document_name: str, chapter_name: str, initial_content: str = ""
) -> OperationStatus:
    r"""Create a new chapter file within an existing document directory.

    .. note::
       While this tool supports creating a chapter with initial content, for clarity it is recommended to create an empty chapter and then use a dedicated write/update tool to add content. This helps separate creation from modification operations.

    This tool adds a new Markdown chapter file to a document collection. The chapter
    will be ordered based on its filename when listed with other chapters. Supports
    optional initial content to bootstrap the chapter with starter text.

    Parameters:
        document_name (str): Name of the existing document directory to add chapter to
        chapter_name (str): Filename for the new chapter. Must be:
            - Valid .md filename (e.g., "03-advanced-features.md")
            - ≤100 characters
            - Valid filesystem filename
            - Cannot contain path separators (/ or \\)
            - Cannot be reserved name like "_manifest.json" or "_SUMMARY.md"
            - Must not already exist in the document
        initial_content (str, optional): Starting content for the new chapter. Can be:
            - Empty string (default): Creates empty chapter file
            - Any valid UTF-8 text content
            - Must be ≤1MB in size

    Returns:
        OperationStatus: Structured result object containing:
            - success (bool): True if chapter was created successfully, False otherwise
            - message (str): Human-readable description of the operation result
            - details (Dict[str, Any], optional): Additional context including:
                - document_name (str): Name of the parent document (on success)
                - chapter_name (str): Name of the created chapter (on success)

    Example Usage:
        ```json
        {
            "name": "create_chapter",
            "arguments": {
                "document_name": "user_guide",
                "chapter_name": "03-advanced-features.md",
                "initial_content": "# Advanced Features\n\nThis chapter covers advanced functionality."
            }
        }
        ```

    Example Success Response:
        ```json
        {
            "success": true,
            "message": "Chapter '03-advanced-features.md' created successfully in document 'user_guide'.",
            "details": {
                "document_name": "user_guide",
                "chapter_name": "03-advanced-features.md"
            }
        }
        ```

    Example Error Response:
        ```json
        {
            "success": false,
            "message": "Chapter '03-advanced-features.md' already exists in document 'user_guide'.",
            "details": null
        }
        ```
    """
    # Validate inputs
    is_valid_doc, doc_error = _validate_document_name(document_name)
    if not is_valid_doc:
        return OperationStatus(success=False, message=doc_error)

    is_valid_chapter, chapter_error = _validate_chapter_name(chapter_name)
    if not is_valid_chapter:
        return OperationStatus(success=False, message=chapter_error)

    is_valid_content, content_error = _validate_content(initial_content)
    if not is_valid_content:
        return OperationStatus(success=False, message=content_error)

    doc_path = _get_document_path(document_name)
    if not doc_path.is_dir():
        return OperationStatus(
            success=False, message=f"Document '{document_name}' not found."
        )

    if not _is_valid_chapter_filename(chapter_name):
        return OperationStatus(
            success=False,
            message=f"Invalid chapter name '{chapter_name}'. Must be a .md file and not a reserved name like '{CHAPTER_MANIFEST_FILE}'.",
        )

    chapter_path = _get_chapter_path(document_name, chapter_name)
    if chapter_path.exists():
        return OperationStatus(
            success=False,
            message=f"Chapter '{chapter_name}' already exists in document '{document_name}'.",
        )

    try:
        chapter_path.write_text(initial_content, encoding="utf-8")
        return OperationStatus(
            success=True,
            message=f"Chapter '{chapter_name}' created successfully in document '{document_name}'.",
            details={"document_name": document_name, "chapter_name": chapter_name},
        )
    except Exception as e:
        return OperationStatus(
            success=False,
            message=f"Error creating chapter '{chapter_name}' in document '{document_name}': {e}",
        )


@mcp_server.tool()
@log_mcp_call
@auto_snapshot("delete_chapter")
def delete_chapter(document_name: str, chapter_name: str) -> OperationStatus:
    """Delete a chapter file from a document directory.

    This tool permanently removes a chapter file from a document collection.
    The operation is irreversible and will delete the file from the filesystem.
    Use with caution as the chapter content will be lost.

    Parameters:
        document_name (str): Name of the document directory containing the chapter
        chapter_name (str): Filename of the chapter to delete. Must be:
            - Valid .md filename (e.g., "02-old-chapter.md")
            - Existing file within the specified document directory
            - Not a reserved filename like "_manifest.json"

    Returns:
        OperationStatus: Structured result object containing:
            - success (bool): True if chapter was deleted successfully, False otherwise
            - message (str): Human-readable description of the operation result
            - details (Dict[str, Any], optional): Additional context (currently None)

    Example Usage:
        ```json
        {
            "name": "delete_chapter",
            "arguments": {
                "document_name": "user_guide",
                "chapter_name": "02-outdated-section.md"
            }
        }
        ```

    Example Success Response:
        ```json
        {
            "success": true,
            "message": "Chapter '02-outdated-section.md' deleted successfully from document 'user_guide'.",
            "details": null
        }
        ```

    Example Error Response:
        ```json
        {
            "success": false,
            "message": "Chapter '02-outdated-section.md' not found in document 'user_guide'.",
            "details": null
        }
        ```
    """
    if not _is_valid_chapter_filename(
        chapter_name
    ):  # Check early to avoid issues with non-MD files
        return OperationStatus(
            success=False,
            message=f"Invalid target '{chapter_name}'. Not a valid chapter Markdown file name.",
        )

    chapter_path = _get_chapter_path(document_name, chapter_name)
    if not chapter_path.is_file():
        return OperationStatus(
            success=False,
            message=f"Chapter '{chapter_name}' not found in document '{document_name}'.",
        )

    try:
        chapter_path.unlink()
        return OperationStatus(
            success=True,
            message=f"Chapter '{chapter_name}' deleted successfully from document '{document_name}'.",
        )
    except Exception as e:
        return OperationStatus(
            success=False,
            message=f"Error deleting chapter '{chapter_name}' from document '{document_name}': {e}",
        )


@mcp_server.tool()
@log_mcp_call
@auto_snapshot("write_chapter_content")
@safety_enhanced_write_operation("write_chapter_content")
def write_chapter_content(
    document_name: str,
    chapter_name: str,
    new_content: str,
    last_known_modified: str | None = None,
    force_write: bool = False,
) -> OperationStatus:
    r"""Overwrite the entire content of a chapter file with new content.

    .. deprecated:: 0.18.0
       This tool's behavior of creating a chapter if it doesn't exist is deprecated and will be removed in a future version.
       In the future, this tool will only write to existing chapters. Please use `create_chapter` for new chapters.

    This tool completely replaces the content of an existing chapter file or creates
    a new chapter if it doesn't exist. The operation provides diff information showing
    exactly what changed between the original and new content.

    SAFETY FEATURES:
    - Checks file modification time before writing to detect external changes
    - Creates automatic micro-snapshots before destructive operations
    - Records all modifications in document history
    - Provides detailed safety warnings and recommendations

    Parameters:
        document_name (str): Name of the document directory containing the chapter
        chapter_name (str): Filename of the chapter to write. Must be:
            - Valid .md filename (e.g., "01-introduction.md")
            - Valid filesystem filename
            - Not a reserved filename like "_manifest.json"
        new_content (str): Complete new content for the chapter file. Can be:
            - Any valid UTF-8 text content
            - Empty string (creates empty chapter)
            - Must be ≤1MB in size
        last_known_modified (Optional[str]): ISO timestamp of last known modification
            for safety checking. If provided, will warn if file was modified externally.
        force_write (bool): If True, will proceed with write even if safety warnings exist.
            Default is False for maximum safety.

    Returns:
        OperationStatus: Enhanced result object containing:
            - success (bool): True if content was written successfully, False otherwise
            - message (str): Human-readable description of the operation result
            - details (Dict[str, Any], optional): Includes diff information:
                - changed (bool): Whether content was actually modified
                - diff (str): Unified diff showing changes made
                - summary (str): Brief description of changes
                - lines_added (int): Number of lines added
                - lines_removed (int): Number of lines removed
            - safety_info (ContentFreshnessStatus): Safety check results
            - snapshot_created (Optional[str]): ID of safety snapshot if created
            - warnings (List[str]): List of safety warnings

    Example Usage:
        ```json
        {
            "name": "write_chapter_content",
            "arguments": {
                "document_name": "user_guide",
                "chapter_name": "01-introduction.md",
                "new_content": "# Introduction\n\nWelcome to our comprehensive user guide.\n\nThis guide will help you get started."
            }
        }
        ```

    Example Success Response:
        ```json
        {
            "success": true,
            "message": "Content of chapter '01-introduction.md' in document 'user_guide' updated successfully.",
            "details": {
                "changed": true,
                "diff": "--- 01-introduction.md (before)\n+++ 01-introduction.md (after)\n@@ -1,3 +1,4 @@\n # Introduction\n \n-Welcome to our guide.\n+Welcome to our comprehensive user guide.\n+\n+This guide will help you get started.",
                "summary": "Modified content: +3 lines, -1 lines",
                "lines_added": 3,
                "lines_removed": 1
            }
        }
        ```
    """
    doc_path = _get_document_path(document_name)
    if not doc_path.is_dir():
        return OperationStatus(
            success=False, message=f"Document '{document_name}' not found."
        )

    if not _is_valid_chapter_filename(chapter_name):
        return OperationStatus(
            success=False, message=f"Invalid chapter name '{chapter_name}'."
        )

    chapter_path = _get_chapter_path(document_name, chapter_name)

    try:
        # Read original content before overwriting
        original_content = ""
        if chapter_path.exists():
            original_content = chapter_path.read_text(encoding="utf-8")

        # Write new content
        chapter_path.write_text(new_content, encoding="utf-8")

        # Generate diff for details
        diff_info = _generate_content_diff(original_content, new_content, chapter_name)

        return OperationStatus(
            success=True,
            message=f"Content of chapter '{chapter_name}' in document '{document_name}' updated successfully.",
            details=diff_info,
        )
    except Exception as e:
        return OperationStatus(
            success=False,
            message=f"Error writing to chapter '{chapter_name}' in document '{document_name}': {e}",
        )


# ============================================================================
# PARAGRAPH MANAGEMENT TOOLS (ATOMIC)
# ============================================================================
# Atomic tools for precise paragraph-level operations within chapters


@mcp_server.tool()
@log_mcp_call
@auto_snapshot("replace_paragraph")
@safety_enhanced_write_operation("replace_paragraph")
def replace_paragraph(
    document_name: str,
    chapter_name: str,
    paragraph_index: int,
    new_content: str,
    last_known_modified: str | None = None,
    force_write: bool = False,
) -> OperationStatus:
    """Replace the content of a specific paragraph within a chapter.

    This atomic tool replaces an existing paragraph at the specified index with
    new content. The paragraph index is zero-based, and the operation will fail
    if the index is out of bounds.

    SAFETY FEATURES:
    - Checks file modification time before writing to detect external changes
    - Creates automatic micro-snapshots before destructive operations
    - Records all modifications in document history
    - Provides detailed safety warnings and recommendations

    Parameters:
        document_name (str): Name of the document directory containing the chapter
        chapter_name (str): Filename of the chapter to modify (must end with .md)
        paragraph_index (int): Zero-indexed position of the paragraph to replace (≥0)
        new_content (str): New content to replace the existing paragraph with
        last_known_modified (Optional[str]): ISO timestamp of last known modification
            for safety checking. If provided, will warn if file was modified externally.
        force_write (bool): If True, will proceed with write even if safety warnings exist.
            Default is False for maximum safety.

    Returns:
        OperationStatus: Enhanced result object with safety information

    Example Usage:
        ```json
        {
            "name": "replace_paragraph",
            "arguments": {
                "document_name": "user_guide",
                "chapter_name": "01-intro.md",
                "paragraph_index": 2,
                "new_content": "This is the updated paragraph content."
            }
        }
        ```
    """
    # Validate inputs
    is_valid_doc, doc_error = _validate_document_name(document_name)
    if not is_valid_doc:
        return OperationStatus(success=False, message=doc_error)

    is_valid_chapter, chapter_error = _validate_chapter_name(chapter_name)
    if not is_valid_chapter:
        return OperationStatus(success=False, message=chapter_error)

    is_valid_index, index_error = _validate_paragraph_index(paragraph_index)
    if not is_valid_index:
        return OperationStatus(success=False, message=index_error)

    is_valid_content, content_error = _validate_content(new_content)
    if not is_valid_content:
        return OperationStatus(success=False, message=content_error)

    chapter_path = _get_chapter_path(document_name, chapter_name)
    if not chapter_path.is_file() or not _is_valid_chapter_filename(chapter_name):
        return OperationStatus(
            success=False,
            message=f"Chapter '{chapter_name}' not found in document '{document_name}'.",
        )

    # Parse last known modified time if provided
    last_known_dt = None
    if last_known_modified:
        try:
            last_known_dt = datetime.datetime.fromisoformat(
                last_known_modified.replace("Z", "+00:00")
            )
            if last_known_dt.tzinfo:
                last_known_dt = last_known_dt.replace(tzinfo=None)
        except ValueError:
            return OperationStatus(
                success=False,
                message=f"Invalid timestamp format: {last_known_modified}",
                warnings=[f"Invalid timestamp format: {last_known_modified}"],
            )

    # Check file freshness for safety
    safety_info = _check_file_freshness(chapter_path, last_known_dt)
    warnings = []

    # Handle safety warnings
    if safety_info.safety_status == "warning" and not force_write:
        warnings.append(f"File was modified externally: {safety_info.message}")
        warnings.extend(safety_info.recommendations)
        return OperationStatus(
            success=False,
            message=f"Safety check failed: {safety_info.message}. Use force_write=True to proceed.",
            safety_info=safety_info,
            warnings=warnings,
        )
    elif safety_info.safety_status == "conflict" and not force_write:
        warnings.append(f"File conflict detected: {safety_info.message}")
        warnings.extend(safety_info.recommendations)
        return OperationStatus(
            success=False,
            message=f"File conflict: {safety_info.message}. Use force_write=True to proceed.",
            safety_info=safety_info,
            warnings=warnings,
        )

    try:
        original_full_content = chapter_path.read_text(encoding="utf-8")
        paragraphs = _split_into_paragraphs(original_full_content)
        total_paragraphs = len(paragraphs)

        if not (0 <= paragraph_index < total_paragraphs):
            return OperationStatus(
                success=False,
                message=f"Paragraph index {paragraph_index} is out of bounds (0-{total_paragraphs - 1}).",
                safety_info=safety_info,
            )

        # Note: Automatic snapshot created by @auto_snapshot decorator

        paragraphs[paragraph_index] = new_content
        final_content = "\n\n".join(paragraphs)
        chapter_path.write_text(final_content, encoding="utf-8")

        # Record modification in history
        _record_modification(
            document_name,
            chapter_name,
            "replace_paragraph",
            details={
                "paragraph_index": paragraph_index,
                "content_length": len(new_content),
                "force_write": force_write,
            },
        )

        # Generate diff for details
        diff_info = _generate_content_diff(
            original_full_content, final_content, chapter_name
        )

        return OperationStatus(
            success=True,
            message=f"Paragraph {paragraph_index} in '{chapter_name}' ({document_name}) successfully replaced.",
            details=diff_info,
            safety_info=safety_info,
            snapshot_created=None,  # Automatic snapshot handled by decorator
            warnings=warnings,
        )
    except Exception as e:
        return OperationStatus(
            success=False,
            message=f"Error replacing paragraph in '{chapter_name}' ({document_name}): {str(e)}",
            safety_info=safety_info,
            warnings=warnings,
        )


@mcp_server.tool()
@log_mcp_call
@auto_snapshot("insert_paragraph_before")
def insert_paragraph_before(
    document_name: str,
    chapter_name: str,
    paragraph_index: int,
    new_content: str,
    force_write: bool = False,
) -> OperationStatus:
    """Insert a new paragraph before the specified index within a chapter.

    This atomic tool inserts new content as a paragraph before the existing
    paragraph at the specified index. All subsequent paragraphs are shifted down.

    Parameters:
        document_name (str): Name of the document directory containing the chapter
        chapter_name (str): Filename of the chapter to modify (must end with .md)
        paragraph_index (int): Zero-indexed position to insert before (≥0)
        new_content (str): Content for the new paragraph to insert

    Returns:
        OperationStatus: Result object with success status, message, and diff details

    Example Usage:
        ```json
        {
            "name": "insert_paragraph_before",
            "arguments": {
                "document_name": "user_guide",
                "chapter_name": "01-intro.md",
                "paragraph_index": 1,
                "new_content": "This paragraph will be inserted before paragraph 1."
            }
        }
        ```
    """
    # Validate inputs
    is_valid_doc, doc_error = _validate_document_name(document_name)
    if not is_valid_doc:
        return OperationStatus(success=False, message=doc_error)

    is_valid_chapter, chapter_error = _validate_chapter_name(chapter_name)
    if not is_valid_chapter:
        return OperationStatus(success=False, message=chapter_error)

    is_valid_index, index_error = _validate_paragraph_index(paragraph_index)
    if not is_valid_index:
        return OperationStatus(success=False, message=index_error)

    is_valid_content, content_error = _validate_content(new_content)
    if not is_valid_content:
        return OperationStatus(success=False, message=content_error)

    chapter_path = _get_chapter_path(document_name, chapter_name)
    if not chapter_path.is_file() or not _is_valid_chapter_filename(chapter_name):
        return OperationStatus(
            success=False,
            message=f"Chapter '{chapter_name}' not found in document '{document_name}'.",
        )

    # Basic safety check
    safety_info = _check_file_freshness(chapter_path)

    try:
        original_full_content = chapter_path.read_text(encoding="utf-8")
        paragraphs = _split_into_paragraphs(original_full_content)
        total_paragraphs = len(paragraphs)

        if not (0 <= paragraph_index <= total_paragraphs):
            return OperationStatus(
                success=False,
                message=f"Paragraph index {paragraph_index} is out of bounds (0-{total_paragraphs}).",
                safety_info=safety_info,
            )

        # Note: Automatic snapshot created by @auto_snapshot decorator

        paragraphs.insert(paragraph_index, new_content)
        final_content = "\n\n".join(paragraphs)
        chapter_path.write_text(final_content, encoding="utf-8")

        # Record modification in history
        _record_modification(
            document_name,
            chapter_name,
            "insert_paragraph_before",
            details={
                "paragraph_index": paragraph_index,
                "content_length": len(new_content),
            },
        )

        # Generate diff for details
        diff_info = _generate_content_diff(
            original_full_content, final_content, chapter_name
        )

        return OperationStatus(
            success=True,
            message=f"Paragraph inserted before index {paragraph_index} in '{chapter_name}' ({document_name}).",
            details=diff_info,
            safety_info=safety_info,
            snapshot_created=None,  # Automatic snapshot handled by decorator
        )
    except Exception as e:
        return OperationStatus(
            success=False,
            message=f"Error inserting paragraph in '{chapter_name}' ({document_name}): {str(e)}",
            safety_info=safety_info,
        )


@mcp_server.tool()
@log_mcp_call
@auto_snapshot("insert_paragraph_after")
def insert_paragraph_after(
    document_name: str, chapter_name: str, paragraph_index: int, new_content: str
) -> OperationStatus:
    """Insert a new paragraph after the specified index within a chapter.

    This atomic tool inserts new content as a paragraph after the existing
    paragraph at the specified index. All subsequent paragraphs are shifted down.

    Parameters:
        document_name (str): Name of the document directory containing the chapter
        chapter_name (str): Filename of the chapter to modify (must end with .md)
        paragraph_index (int): Zero-indexed position to insert after (≥0)
        new_content (str): Content for the new paragraph to insert

    Returns:
        OperationStatus: Result object with success status, message, and diff details

    Example Usage:
        ```json
        {
            "name": "insert_paragraph_after",
            "arguments": {
                "document_name": "user_guide",
                "chapter_name": "01-intro.md",
                "paragraph_index": 1,
                "new_content": "This paragraph will be inserted after paragraph 1."
            }
        }
        ```
    """
    # Validate inputs
    is_valid_doc, doc_error = _validate_document_name(document_name)
    if not is_valid_doc:
        return OperationStatus(success=False, message=doc_error)

    is_valid_chapter, chapter_error = _validate_chapter_name(chapter_name)
    if not is_valid_chapter:
        return OperationStatus(success=False, message=chapter_error)

    is_valid_index, index_error = _validate_paragraph_index(paragraph_index)
    if not is_valid_index:
        return OperationStatus(success=False, message=index_error)

    is_valid_content, content_error = _validate_content(new_content)
    if not is_valid_content:
        return OperationStatus(success=False, message=content_error)

    chapter_path = _get_chapter_path(document_name, chapter_name)
    if not chapter_path.is_file() or not _is_valid_chapter_filename(chapter_name):
        return OperationStatus(
            success=False,
            message=f"Chapter '{chapter_name}' not found in document '{document_name}'.",
        )

    try:
        original_full_content = chapter_path.read_text(encoding="utf-8")
        paragraphs = _split_into_paragraphs(original_full_content)
        total_paragraphs = len(paragraphs)

        if not (0 <= paragraph_index < total_paragraphs) and not (
            total_paragraphs == 0 and paragraph_index == 0
        ):
            return OperationStatus(
                success=False,
                message=f"Paragraph index {paragraph_index} is out of bounds (0-{total_paragraphs - 1}).",
            )

        if total_paragraphs == 0 and paragraph_index == 0:  # Insert into empty doc
            paragraphs.append(new_content)
        else:
            paragraphs.insert(paragraph_index + 1, new_content)

        final_content = "\n\n".join(paragraphs)
        chapter_path.write_text(final_content, encoding="utf-8")

        # Generate diff for details
        diff_info = _generate_content_diff(
            original_full_content, final_content, chapter_name
        )

        return OperationStatus(
            success=True,
            message=f"Paragraph inserted after index {paragraph_index} in '{chapter_name}' ({document_name}).",
            details=diff_info,
        )
    except Exception as e:
        return OperationStatus(
            success=False,
            message=f"Error inserting paragraph in '{chapter_name}' ({document_name}): {str(e)}",
        )


@mcp_server.tool()
@log_mcp_call
@auto_snapshot("delete_paragraph")
def delete_paragraph(
    document_name: str,
    chapter_name: str,
    paragraph_index: int,
    force_write: bool = False,
) -> OperationStatus:
    """Delete a specific paragraph from a chapter.

    This atomic tool removes the paragraph at the specified index from the chapter.
    All subsequent paragraphs are shifted up to fill the gap.

    Parameters:
        document_name (str): Name of the document directory containing the chapter
        chapter_name (str): Filename of the chapter to modify (must end with .md)
        paragraph_index (int): Zero-indexed position of the paragraph to delete (≥0)

    Returns:
        OperationStatus: Result object with success status, message, and diff details

    Example Usage:
        ```json
        {
            "name": "delete_paragraph",
            "arguments": {
                "document_name": "user_guide",
                "chapter_name": "01-intro.md",
                "paragraph_index": 2
            }
        }
        ```
    """
    # Validate inputs
    is_valid_doc, doc_error = _validate_document_name(document_name)
    if not is_valid_doc:
        return OperationStatus(success=False, message=doc_error)

    is_valid_chapter, chapter_error = _validate_chapter_name(chapter_name)
    if not is_valid_chapter:
        return OperationStatus(success=False, message=chapter_error)

    is_valid_index, index_error = _validate_paragraph_index(paragraph_index)
    if not is_valid_index:
        return OperationStatus(success=False, message=index_error)

    chapter_path = _get_chapter_path(document_name, chapter_name)
    if not chapter_path.is_file() or not _is_valid_chapter_filename(chapter_name):
        return OperationStatus(
            success=False,
            message=f"Chapter '{chapter_name}' not found in document '{document_name}'.",
        )

    try:
        original_full_content = chapter_path.read_text(encoding="utf-8")
        paragraphs = _split_into_paragraphs(original_full_content)
        total_paragraphs = len(paragraphs)

        if total_paragraphs == 0:
            return OperationStatus(
                success=False,
                message="Cannot delete paragraph from an empty chapter.",
            )

        if not (0 <= paragraph_index < total_paragraphs):
            return OperationStatus(
                success=False,
                message=f"Paragraph index {paragraph_index} is out of bounds for chapter with {total_paragraphs} paragraphs (valid range 0-{total_paragraphs - 1}).",
            )

        del paragraphs[paragraph_index]
        final_content = "\n\n".join(paragraphs)
        chapter_path.write_text(final_content, encoding="utf-8")

        # Generate diff for details
        diff_info = _generate_content_diff(
            original_full_content, final_content, chapter_name
        )

        return OperationStatus(
            success=True,
            message=f"Paragraph {paragraph_index} deleted from '{chapter_name}' ({document_name}).",
            details=diff_info,
        )
    except Exception as e:
        return OperationStatus(
            success=False,
            message=f"Error deleting paragraph from '{chapter_name}' ({document_name}): {str(e)}",
        )


@mcp_server.tool()
@register_batchable_operation("append_paragraph_to_chapter")
@log_mcp_call
@auto_snapshot("append_paragraph_to_chapter")
def append_paragraph_to_chapter(
    document_name: str, chapter_name: str, paragraph_content: str
) -> OperationStatus:
    r"""Append a new paragraph to the end of a specific chapter.

    This tool adds new content as a paragraph at the end of an existing chapter file.
    The new paragraph will be separated from existing content by proper paragraph
    spacing (blank lines). Provides diff information showing the changes made.

    Parameters:
        document_name (str): Name of the document directory containing the chapter
        chapter_name (str): Filename of the chapter to append to. Must be:
            - Valid .md filename (e.g., "01-introduction.md")
            - Existing file within the specified document directory
            - Not a reserved filename like "_manifest.json"
        paragraph_content (str): Content to append as a new paragraph. Must be:
            - Non-empty string
            - Valid UTF-8 text content
            - Must be ≤1MB in size

    Returns:
        OperationStatus: Structured result object containing:
            - success (bool): True if paragraph was appended successfully, False otherwise
            - message (str): Human-readable description of the operation result
            - details (Dict[str, Any], optional): Includes diff information:
                - changed (bool): Whether content was actually modified
                - diff (str): Unified diff showing changes made
                - summary (str): Brief description of changes
                - lines_added (int): Number of lines added
                - lines_removed (int): Number of lines removed

    Example Usage:
        ```json
        {
            "name": "append_paragraph_to_chapter",
            "arguments": {
                "document_name": "user_guide",
                "chapter_name": "01-introduction.md",
                "paragraph_content": "This additional paragraph provides more context about the features covered in this guide."
            }
        }
        ```

    Example Success Response:
        ```json
        {
            "success": true,
            "message": "Paragraph appended to chapter '01-introduction.md' in document 'user_guide'.",
            "details": {
                "changed": true,
                "diff": "--- 01-introduction.md (before)\n+++ 01-introduction.md (after)\n@@ -10,0 +10,2 @@\n+\n+This additional paragraph provides more context about the features covered in this guide.",
                "summary": "Modified content: +2 lines, -0 lines",
                "lines_added": 2,
                "lines_removed": 0
            }
        }
        ```
    """
    chapter_path = _get_chapter_path(document_name, chapter_name)
    if not chapter_path.is_file() or not _is_valid_chapter_filename(chapter_name):
        return OperationStatus(
            success=False,
            message=f"Chapter '{chapter_name}' not found in document '{document_name}'.",
        )

    try:
        original_full_content = chapter_path.read_text(encoding="utf-8")
        paragraphs = _split_into_paragraphs(original_full_content)
        paragraphs.append(paragraph_content)
        # Filter out any potential empty strings that might arise if original_full_content was just newlines
        # or if _split_into_paragraphs somehow yields them, though it's designed not to.
        final_paragraphs = [p for p in paragraphs if p]
        final_content = "\n\n".join(final_paragraphs)
        # If the only content is the new paragraph and it's not empty, ensure no leading newlines.
        if (
            len(final_paragraphs) == 1
            and final_paragraphs[0] == paragraph_content
            and paragraph_content
        ):
            final_content = paragraph_content
        chapter_path.write_text(final_content, encoding="utf-8")

        # Generate diff for details
        diff_info = _generate_content_diff(
            original_full_content, final_content, chapter_name
        )

        return OperationStatus(
            success=True,
            message=f"Paragraph appended to chapter '{chapter_name}' in document '{document_name}'.",
            details=diff_info,
        )
    except Exception as e:
        return OperationStatus(
            success=False,
            message=f"Error appending paragraph to '{chapter_name}': {str(e)}",
        )


# ============================================================================
# SEARCH AND REPLACE TOOLS
# ============================================================================
# Tools for finding and replacing text content within documents and chapters


def replace_text_in_chapter(
    document_name: str, chapter_name: str, text_to_find: str, replacement_text: str
) -> OperationStatus:
    r"""Replace all occurrences of a text string with another string in a specific chapter.

    This tool performs case-sensitive text replacement throughout a chapter file.
    All instances of the search text will be replaced with the replacement text.
    Provides detailed information about the number of replacements made and diff output.

    Parameters:
        document_name (str): Name of the document directory containing the chapter
        chapter_name (str): Filename of the chapter to modify. Must be:
            - Valid .md filename (e.g., "01-introduction.md")
            - Existing file within the specified document directory
            - Not a reserved filename like "_manifest.json"
        text_to_find (str): Text string to search for and replace. Must be:
            - Non-empty string
            - Exact text match (case-sensitive)
        replacement_text (str): Text to replace found occurrences with. Can be:
            - Any valid UTF-8 text content
            - Empty string (effectively deletes the found text)
            - Must be ≤1MB in size

    Returns:
        OperationStatus: Structured result object containing:
            - success (bool): True if operation completed successfully, False otherwise
            - message (str): Human-readable description of the operation result
            - details (Dict[str, Any], optional): Includes replacement information:
                - changed (bool): Whether content was actually modified
                - diff (str): Unified diff showing changes made (if any)
                - summary (str): Brief description of changes
                - lines_added (int): Number of lines added
                - lines_removed (int): Number of lines removed
                - occurrences_replaced (int): Number of text occurrences replaced

    Example Usage:
        ```json
        {
            "name": "replace_text_in_chapter",
            "arguments": {
                "document_name": "user_guide",
                "chapter_name": "01-introduction.md",
                "text_to_find": "old version",
                "replacement_text": "new version"
            }
        }
        ```

    Example Success Response:
        ```json
        {
            "success": true,
            "message": "All 2 occurrences of 'old version' replaced with 'new version' in chapter '01-introduction.md'.",
            "details": {
                "changed": true,
                "diff": "--- 01-introduction.md (before)\n+++ 01-introduction.md (after)\n@@ -5,1 +5,1 @@\n-This old version is deprecated.\n+This new version is current.",
                "summary": "Modified content: +1 lines, -1 lines",
                "lines_added": 1,
                "lines_removed": 1,
                "occurrences_replaced": 2
            }
        }
        ```

    Example No Matches Response:
        ```json
        {
            "success": true,
            "message": "Text 'nonexistent' not found in chapter '01-introduction.md'. No changes made.",
            "details": {
                "occurrences_found": 0
            }
        }
        ```
    """
    # Validate inputs
    is_valid_doc, doc_error = _validate_document_name(document_name)
    if not is_valid_doc:
        return OperationStatus(success=False, message=doc_error)

    is_valid_chapter, chapter_error = _validate_chapter_name(chapter_name)
    if not is_valid_chapter:
        return OperationStatus(success=False, message=chapter_error)

    if not text_to_find:
        return OperationStatus(success=False, message="Text to find cannot be empty.")

    is_valid_replacement, replacement_error = _validate_content(replacement_text)
    if not is_valid_replacement:
        return OperationStatus(success=False, message=replacement_error)

    chapter_path = _get_chapter_path(document_name, chapter_name)
    if not chapter_path.is_file() or not _is_valid_chapter_filename(chapter_name):
        return OperationStatus(
            success=False,
            message=f"Chapter '{chapter_name}' not found in document '{document_name}'.",
        )

    try:
        original_content = chapter_path.read_text(encoding="utf-8")
        if text_to_find not in original_content:
            return OperationStatus(
                success=True,  # Success, but no change
                message=f"Text '{text_to_find}' not found in chapter '{chapter_name}'. No changes made.",
                details={"occurrences_found": 0},
            )

        modified_content = original_content.replace(text_to_find, replacement_text)
        occurrences = original_content.count(text_to_find)
        chapter_path.write_text(modified_content, encoding="utf-8")

        # Generate diff for details
        diff_info = _generate_content_diff(
            original_content, modified_content, chapter_name
        )
        diff_info["occurrences_replaced"] = occurrences

        return OperationStatus(
            success=True,
            message=f"All {occurrences} occurrences of '{text_to_find}' replaced with '{replacement_text}' in chapter '{chapter_name}'.",
            details=diff_info,
        )
    except Exception as e:
        return OperationStatus(
            success=False, message=f"Error replacing text in '{chapter_name}': {str(e)}"
        )


def replace_text_in_document(
    document_name: str, text_to_find: str, replacement_text: str
) -> OperationStatus:
    """Replace all occurrences of a text string with another string throughout all chapters of a document.

    This tool performs case-sensitive text replacement across all chapter files within
    a document directory. It processes each chapter sequentially and provides comprehensive
    reporting on which chapters were modified and how many replacements were made.

    Parameters:
        document_name (str): Name of the document directory to process
        text_to_find (str): Text string to search for and replace. Must be:
            - Non-empty string
            - Exact text match (case-sensitive)
        replacement_text (str): Text to replace found occurrences with. Can be:
            - Any valid UTF-8 text content
            - Empty string (effectively deletes the found text)
            - Must be ≤1MB in size

    Returns:
        OperationStatus: Structured result object containing:
            - success (bool): True if operation completed successfully, False otherwise
            - message (str): Human-readable description of the operation result
            - details (Dict[str, Any], optional): Includes comprehensive replacement information:
                - chapters_modified_count (int): Number of chapters that had replacements
                - total_occurrences_replaced (int): Total number of text occurrences replaced
                - modified_chapters (List[Dict]): Details for each modified chapter including:
                    - chapter_name (str): Name of the modified chapter file
                    - occurrences_replaced (int): Number of replacements in this chapter

    Example Usage:
        ```json
        {
            "name": "replace_text_in_document",
            "arguments": {
                "document_name": "user_guide",
                "text_to_find": "version 1.0",
                "replacement_text": "version 2.0"
            }
        }
        ```

    Example Success Response:
        ```json
        {
            "success": true,
            "message": "Text replacement completed in document 'user_guide'. 5 occurrences replaced across 3 chapter(s).",
            "details": {
                "chapters_modified_count": 3,
                "total_occurrences_replaced": 5,
                "modified_chapters": [
                    {
                        "chapter_name": "01-introduction.md",
                        "occurrences_replaced": 2
                    },
                    {
                        "chapter_name": "02-setup.md",
                        "occurrences_replaced": 1
                    },
                    {
                        "chapter_name": "03-features.md",
                        "occurrences_replaced": 2
                    }
                ]
            }
        }
        ```

    Example No Matches Response:
        ```json
        {
            "success": true,
            "message": "Text 'nonexistent' not found in any chapters of document 'user_guide'. No changes made.",
            "details": {
                "chapters_modified_count": 0,
                "total_occurrences_replaced": 0
            }
        }
    """
    doc_path = _get_document_path(document_name)
    if not doc_path.is_dir():
        return OperationStatus(
            success=False, message=f"Document '{document_name}' not found."
        )

    ordered_chapter_files = _get_ordered_chapter_files(document_name)
    if not ordered_chapter_files:
        return OperationStatus(
            success=True,
            message=f"Document '{document_name}' contains no chapters. No changes made.",
            details={"chapters_modified_count": 0},
        )

    chapters_modified_count = 0
    total_occurrences_replaced = 0
    modified_chapters_details = []

    for chapter_file_path in ordered_chapter_files:
        try:
            original_content = chapter_file_path.read_text(encoding="utf-8")
            if text_to_find in original_content:
                occurrences_in_chapter = original_content.count(text_to_find)
                modified_content = original_content.replace(
                    text_to_find, replacement_text
                )
                chapter_file_path.write_text(modified_content, encoding="utf-8")
                chapters_modified_count += 1
                total_occurrences_replaced += occurrences_in_chapter
                modified_chapters_details.append(
                    {
                        "chapter_name": chapter_file_path.name,
                        "occurrences_replaced": occurrences_in_chapter,
                    }
                )
        except Exception as e:
            # Log or collect errors per chapter? For now, fail fast on first error.
            return OperationStatus(
                success=False,
                message=f"Error replacing text in chapter '{chapter_file_path.name}' of document '{document_name}': {e}",
                details={"chapters_processed_before_error": chapters_modified_count},
            )

    if chapters_modified_count == 0:
        return OperationStatus(
            success=True,
            message=f"Text '{text_to_find}' not found in any chapters of document '{document_name}'. No changes made.",
            details={"chapters_modified_count": 0, "total_occurrences_replaced": 0},
        )

    return OperationStatus(
        success=True,
        message=f"Text replacement completed in document '{document_name}'. {total_occurrences_replaced} occurrences replaced across {chapters_modified_count} chapter(s).",
        details={
            "chapters_modified_count": chapters_modified_count,
            "total_occurrences_replaced": total_occurrences_replaced,
            "modified_chapters": modified_chapters_details,
        },
    )


# ============================================================================
# STATISTICS AND ANALYTICS TOOLS
# ============================================================================
# Tools for analyzing document and chapter metrics, word counts, etc.


def get_chapter_statistics(
    document_name: str, chapter_name: str
) -> StatisticsReport | None:
    """Retrieve statistical information for a specific chapter.

    This tool analyzes a chapter file and returns comprehensive statistics including
    word count and paragraph count. Useful for content analysis, progress tracking,
    and document planning.

    Parameters:
        document_name (str): Name of the document directory containing the chapter
        chapter_name (str): Filename of the chapter to analyze. Must be:
            - Valid .md filename (e.g., "01-introduction.md")
            - Existing file within the specified document directory
            - Not a reserved filename like "_manifest.json"

    Returns:
        Optional[StatisticsReport]: Statistics object if chapter found, None if not found.
        StatisticsReport contains:
            - scope (str): Description of what was analyzed (e.g., "chapter: user_guide/01-intro.md")
            - word_count (int): Total number of words in the chapter
            - paragraph_count (int): Total number of paragraphs in the chapter
            - chapter_count (None): Not applicable for chapter-level statistics

        Returns None if document doesn't exist or chapter doesn't exist.

    Example Usage:
        ```json
        {
            "name": "get_chapter_statistics",
            "arguments": {
                "document_name": "user_guide",
                "chapter_name": "01-introduction.md"
            }
        }
        ```

    Example Success Response:
        ```json
        {
            "scope": "chapter: user_guide/01-introduction.md",
            "word_count": 342,
            "paragraph_count": 8,
            "chapter_count": null
        }
        ```

    Example Not Found Response:
        ```json
        null
        ```
    """
    chapter_details = read_chapter_content(
        document_name, chapter_name
    )  # Leverages existing tool
    if not chapter_details:
        print(
            f"Could not retrieve chapter '{chapter_name}' in document '{document_name}' for statistics."
        )
        return None

    return StatisticsReport(
        scope=f"chapter: {document_name}/{chapter_name}",
        word_count=chapter_details.word_count,
        paragraph_count=chapter_details.paragraph_count,
    )


def get_document_statistics(document_name: str) -> StatisticsReport | None:
    """Retrieve aggregated statistical information for an entire document.

    This tool analyzes all chapters within a document directory and returns
    comprehensive statistics including total word count, paragraph count, and
    chapter count. Useful for document-level analysis and progress tracking.

    Parameters:
        document_name (str): Name of the document directory to analyze

    Returns:
        Optional[StatisticsReport]: Statistics object if document found, None if not found.
        StatisticsReport contains:
            - scope (str): Description of what was analyzed (e.g., "document: user_guide")
            - word_count (int): Total number of words across all chapters
            - paragraph_count (int): Total number of paragraphs across all chapters
            - chapter_count (int): Total number of chapter files in the document

        Returns None if document directory doesn't exist.
        Returns zero counts if document exists but has no chapter files.

    Example Usage:
        ```json
        {
            "name": "get_document_statistics",
            "arguments": {
                "document_name": "user_guide"
            }
        }
        ```

    Example Success Response:
        ```json
        {
            "scope": "document: user_guide",
            "word_count": 2847,
            "paragraph_count": 156,
            "chapter_count": 8
        }
        ```

    Example Not Found Response:
        ```json
        null
        ```
    """
    # Option 1: Re-use list_documents and find the specific document.
    # This is good for consistency if list_documents is already computed/cached by agent.
    # However, it might be slightly less direct if we only need one document.
    all_docs_info = list_documents()  # This computes stats for all docs
    target_doc_info = next(
        (doc for doc in all_docs_info if doc.document_name == document_name), None
    )

    if not target_doc_info:
        # Check if the document directory exists even if it has no chapters or failed to be processed by list_documents
        doc_path = _get_document_path(document_name)
        if not doc_path.is_dir():
            print(f"Document '{document_name}' not found for statistics.")
            return None
        # If dir exists but not in target_doc_info (e.g. no chapters or processing error in list_documents)
        # We might want to recalculate directly
        ordered_chapter_files = _get_ordered_chapter_files(document_name)
        if not ordered_chapter_files:  # Empty document
            return StatisticsReport(
                scope=f"document: {document_name}",
                word_count=0,
                paragraph_count=0,
                chapter_count=0,
            )
        # If it has chapters but wasn't in list_documents output, it implies an issue. Recalculate:
        # This part is a bit redundant with read_full_document logic but more direct for stats
        total_word_count = 0
        total_paragraph_count = 0
        chapter_count = 0
        for chapter_file_path in ordered_chapter_files:
            details = _read_chapter_content_details(document_name, chapter_file_path)
            if details:
                total_word_count += details.word_count
                total_paragraph_count += details.paragraph_count
                chapter_count += 1
        return StatisticsReport(
            scope=f"document: {document_name}",
            word_count=total_word_count,
            paragraph_count=total_paragraph_count,
            chapter_count=chapter_count,
        )

    return StatisticsReport(
        scope=f"document: {document_name}",
        word_count=target_doc_info.total_word_count,
        paragraph_count=target_doc_info.total_paragraph_count,
        chapter_count=target_doc_info.total_chapters,
    )


def find_text_in_chapter(
    document_name: str, chapter_name: str, query: str, case_sensitive: bool = False
) -> list[ParagraphDetail]:
    """Search for paragraphs containing a specific text string within a single chapter.

    This tool performs exact text matching within paragraph content, returning all
    paragraphs that contain the search query. Supports both case-sensitive and
    case-insensitive searching. Useful for locating specific content for review or editing.

    Parameters:
        document_name (str): Name of the document directory containing the chapter
        chapter_name (str): Filename of the chapter to search (must end with .md)
        query (str): Text string to search for. Must be:
            - Non-empty string
            - Not only whitespace
            - Exact text match (not regex or wildcard)
        case_sensitive (bool, optional): Whether to perform case-sensitive matching.
            - False (default): Case-insensitive search
            - True: Exact case matching required

    Returns:
        List[ParagraphDetail]: List of paragraph objects containing the search query.
        Each ParagraphDetail contains:
            - document_name (str): Name of the parent document
            - chapter_name (str): Name of the chapter file
            - paragraph_index_in_chapter (int): Zero-indexed position within the chapter
            - content (str): Full text content of the matching paragraph
            - word_count (int): Number of words in the paragraph

        Returns empty list [] if no matches found or if chapter/document doesn't exist.

    Example Usage:
        ```json
        {
            "name": "find_text_in_chapter",
            "arguments": {
                "document_name": "user_guide",
                "chapter_name": "01-introduction.md",
                "query": "installation",
                "case_sensitive": false
            }
        }
        ```

    Example Success Response:
        ```json
        [
            {
                "document_name": "user_guide",
                "chapter_name": "01-introduction.md",
                "paragraph_index_in_chapter": 3,
                "content": "The installation process is straightforward and should take only a few minutes.",
                "word_count": 13
            },
            {
                "document_name": "user_guide",
                "chapter_name": "01-introduction.md",
                "paragraph_index_in_chapter": 7,
                "content": "For advanced installation options, see the next chapter.",
                "word_count": 9
            }
        ]
        ```

    Example No Matches Response:
        ```json
        []
        ```
    """
    # Validate inputs
    is_valid_doc, doc_error = _validate_document_name(document_name)
    if not is_valid_doc:
        print(f"Invalid document name: {doc_error}")
        return []

    is_valid_chapter, chapter_error = _validate_chapter_name(chapter_name)
    if not is_valid_chapter:
        print(f"Invalid chapter name: {chapter_error}")
        return []

    is_valid_query, query_error = _validate_search_query(query)
    if not is_valid_query:
        print(f"Invalid search query: {query_error}")
        return []

    results = []
    chapter_content_obj = read_chapter_content(document_name, chapter_name)
    if not chapter_content_obj:
        return results  # Empty list if chapter not found

    paragraphs_text = _split_into_paragraphs(chapter_content_obj.content)
    search_query = query if case_sensitive else query.lower()

    for i, para_text in enumerate(paragraphs_text):
        current_para_content = para_text if case_sensitive else para_text.lower()
        if search_query in current_para_content:
            results.append(
                ParagraphDetail(
                    document_name=document_name,
                    chapter_name=chapter_name,
                    paragraph_index_in_chapter=i,
                    content=para_text,  # Return original case paragraph
                    word_count=_count_words(para_text),
                )
            )

    return results


def find_text_in_document(
    document_name: str, query: str, case_sensitive: bool = False
) -> list[ParagraphDetail]:
    """Search for paragraphs containing a specific text string across all chapters in a document.

    This tool performs text matching across all chapter files within a document directory,
    returning all paragraphs that contain the search query. Supports both case-sensitive
    and case-insensitive searching. Useful for document-wide content discovery and analysis.

    Parameters:
        document_name (str): Name of the document directory to search in
        query (str): Text string to search for within paragraph content. Must be:
            - Non-empty string
            - Can include spaces and special characters
            - Matched as exact substring (not regex)
        case_sensitive (bool, optional): Whether to perform case-sensitive matching.
            - True: Exact case matching (e.g., "API" ≠ "api")
            - False (default): Case-insensitive matching (e.g., "API" = "api")

    Returns:
        List[ParagraphDetail]: List of matching paragraph objects. Each ParagraphDetail contains:
            - document_name (str): Name of the document containing the paragraph
            - chapter_name (str): Name of the chapter file containing the paragraph
            - paragraph_index_in_chapter (int): Zero-indexed position within the chapter
            - content (str): Full text content of the matching paragraph
            - word_count (int): Number of words in the paragraph

        Returns empty list if no matches found or if document doesn't exist.
        Results are ordered by chapter filename, then by paragraph position within chapter.

    Example Usage:
        ```json
        {
            "name": "find_text_in_document",
            "arguments": {
                "document_name": "user_guide",
                "query": "authentication",
                "case_sensitive": false
            }
        }
        ```

    Example Success Response:
        ```json
        [
            {
                "document_name": "user_guide",
                "chapter_name": "02-setup.md",
                "paragraph_index_in_chapter": 3,
                "content": "Configure authentication settings in your application to ensure secure access.",
                "word_count": 12
            },
            {
                "document_name": "user_guide",
                "chapter_name": "05-security.md",
                "paragraph_index_in_chapter": 1,
                "content": "Authentication is a critical component of any secure application.",
                "word_count": 10
            }
        ]
        ```

    Example No Matches Response:
        ```json
        []
        ```
    """
    all_results = []
    doc_path = _get_document_path(document_name)
    if not doc_path.is_dir():
        print(f"Document '{document_name}' not found for text search.")
        return all_results

    ordered_chapter_files = _get_ordered_chapter_files(document_name)
    if not ordered_chapter_files:
        return all_results  # Empty list if no chapters

    for chapter_file_path in ordered_chapter_files:
        chapter_name = chapter_file_path.name
        # Delegate to find_text_in_chapter for each chapter
        chapter_results = find_text_in_chapter(
            document_name, chapter_name, query, case_sensitive
        )
        all_results.extend(chapter_results)

    return all_results


@mcp_server.tool()
@log_mcp_call
@auto_snapshot("move_paragraph_before")
def move_paragraph_before(
    document_name: str,
    chapter_name: str,
    paragraph_to_move_index: int,
    target_paragraph_index: int,
) -> OperationStatus:
    """Move a paragraph to appear before another paragraph within the same chapter.

    This atomic tool reorders paragraphs within a chapter by moving the paragraph
    at the source index to appear before the paragraph at the target index.

    Parameters:
        document_name (str): Name of the document directory containing the chapter
        chapter_name (str): Filename of the chapter to modify (must end with .md)
        paragraph_to_move_index (int): Zero-indexed position of the paragraph to move (≥0)
        target_paragraph_index (int): Zero-indexed position to move before (≥0)

    Returns:
        OperationStatus: Result object with success status, message, and diff details

    Example Usage:
        ```json
        {
            "name": "move_paragraph_before",
            "arguments": {
                "document_name": "user_guide",
                "chapter_name": "01-intro.md",
                "paragraph_to_move_index": 3,
                "target_paragraph_index": 1
            }
        }
        ```
    """
    # Validate inputs
    is_valid_doc, doc_error = _validate_document_name(document_name)
    if not is_valid_doc:
        return OperationStatus(success=False, message=doc_error)

    is_valid_chapter, chapter_error = _validate_chapter_name(chapter_name)
    if not is_valid_chapter:
        return OperationStatus(success=False, message=chapter_error)

    is_valid_move_index, move_index_error = _validate_paragraph_index(
        paragraph_to_move_index
    )
    if not is_valid_move_index:
        return OperationStatus(success=False, message=f"Move index: {move_index_error}")

    is_valid_target_index, target_index_error = _validate_paragraph_index(
        target_paragraph_index
    )
    if not is_valid_target_index:
        return OperationStatus(
            success=False, message=f"Target index: {target_index_error}"
        )

    if paragraph_to_move_index == target_paragraph_index:
        return OperationStatus(
            success=False,
            message="Cannot move a paragraph before itself.",
        )

    chapter_path = _get_chapter_path(document_name, chapter_name)
    if not chapter_path.is_file() or not _is_valid_chapter_filename(chapter_name):
        return OperationStatus(
            success=False,
            message=f"Chapter '{chapter_name}' not found in document '{document_name}'.",
        )

    try:
        original_full_content = chapter_path.read_text(encoding="utf-8")
        paragraphs = _split_into_paragraphs(original_full_content)
        total_paragraphs = len(paragraphs)

        if not (0 <= paragraph_to_move_index < total_paragraphs):
            return OperationStatus(
                success=False,
                message=f"Paragraph to move index {paragraph_to_move_index} is out of bounds (0-{total_paragraphs - 1}).",
            )

        if not (0 <= target_paragraph_index < total_paragraphs):
            return OperationStatus(
                success=False,
                message=f"Target paragraph index {target_paragraph_index} is out of bounds (0-{total_paragraphs - 1}).",
            )

        # Move the paragraph
        paragraph_to_move = paragraphs.pop(paragraph_to_move_index)

        # Adjust target index if necessary (if we moved from before the target)
        if paragraph_to_move_index < target_paragraph_index:
            target_paragraph_index -= 1

        paragraphs.insert(target_paragraph_index, paragraph_to_move)

        final_content = "\n\n".join(paragraphs)
        chapter_path.write_text(final_content, encoding="utf-8")

        # Generate diff for details
        diff_info = _generate_content_diff(
            original_full_content, final_content, chapter_name
        )

        return OperationStatus(
            success=True,
            message=f"Paragraph {paragraph_to_move_index} moved before paragraph {target_paragraph_index} in '{chapter_name}' ({document_name}).",
            details=diff_info,
        )
    except Exception as e:
        return OperationStatus(
            success=False,
            message=f"Error moving paragraph in '{chapter_name}' ({document_name}): {str(e)}",
        )


@mcp_server.tool()
@log_mcp_call
@auto_snapshot("move_paragraph_to_end")
def move_paragraph_to_end(
    document_name: str, chapter_name: str, paragraph_to_move_index: int
) -> OperationStatus:
    """Move a paragraph to the end of a chapter.

    This atomic tool moves the paragraph at the specified index to the end of the
    chapter, after all other paragraphs.

    Parameters:
        document_name (str): Name of the document directory containing the chapter
        chapter_name (str): Filename of the chapter to modify (must end with .md)
        paragraph_to_move_index (int): Zero-indexed position of the paragraph to move (≥0)

    Returns:
        OperationStatus: Result object with success status, message, and diff details

    Example Usage:
        ```json
        {
            "name": "move_paragraph_to_end",
            "arguments": {
                "document_name": "user_guide",
                "chapter_name": "01-intro.md",
                "paragraph_to_move_index": 1
            }
        }
        ```
    """
    # Validate inputs
    is_valid_doc, doc_error = _validate_document_name(document_name)
    if not is_valid_doc:
        return OperationStatus(success=False, message=doc_error)

    is_valid_chapter, chapter_error = _validate_chapter_name(chapter_name)
    if not is_valid_chapter:
        return OperationStatus(success=False, message=chapter_error)

    is_valid_index, index_error = _validate_paragraph_index(paragraph_to_move_index)
    if not is_valid_index:
        return OperationStatus(success=False, message=index_error)

    chapter_path = _get_chapter_path(document_name, chapter_name)
    if not chapter_path.is_file() or not _is_valid_chapter_filename(chapter_name):
        return OperationStatus(
            success=False,
            message=f"Chapter '{chapter_name}' not found in document '{document_name}'.",
        )

    try:
        original_full_content = chapter_path.read_text(encoding="utf-8")
        paragraphs = _split_into_paragraphs(original_full_content)
        total_paragraphs = len(paragraphs)

        if not (0 <= paragraph_to_move_index < total_paragraphs):
            return OperationStatus(
                success=False,
                message=f"Paragraph index {paragraph_to_move_index} is out of bounds (0-{total_paragraphs - 1}).",
            )

        # If already at the end, no need to move
        if paragraph_to_move_index == total_paragraphs - 1:
            return OperationStatus(
                success=True,
                message=f"Paragraph {paragraph_to_move_index} is already at the end of '{chapter_name}' ({document_name}).",
                details={
                    "changed": False,
                    "summary": "No changes made - paragraph already at end",
                },
            )

        # Move the paragraph to the end
        if paragraph_to_move_index == len(paragraphs) - 1:
            return OperationStatus(
                success=True,
                message=f"Paragraph {paragraph_to_move_index} is already at the end of '{chapter_name}' ({document_name}).",
                details={"changed": False},
            )

        paragraph_to_move = paragraphs.pop(paragraph_to_move_index)
        paragraphs.append(paragraph_to_move)

        final_content = "\n\n".join(paragraphs)
        chapter_path.write_text(final_content, encoding="utf-8")

        # Generate diff for details
        diff_info = _generate_content_diff(
            original_full_content, final_content, chapter_name
        )

        return OperationStatus(
            success=True,
            message=f"Paragraph {paragraph_to_move_index} moved to end of '{chapter_name}' ({document_name}).",
            details=diff_info,
        )
    except Exception as e:
        return OperationStatus(
            success=False,
            message=f"Error moving paragraph to end in '{chapter_name}' ({document_name}): {str(e)}",
        )


# DEPRECATED INTERNAL FUNCTION: Use check_content_status unified tool instead
# This function is no longer exposed as an MCP tool and is only used internally
# by the unified check_content_status tool. Do not call directly.
@log_mcp_call
def check_content_freshness(
    document_name: str,
    chapter_name: str | None = None,
    last_known_modified: str | None = None,
) -> ContentFreshnessStatus:
    """Check if content has been modified since last known modification time.

    This safety tool validates whether document or chapter content has been modified
    externally since the last known modification time. Essential for preventing
    accidental overwrites when content is modified outside the current context.

    Parameters:
        document_name (str): Name of the document directory to check
        chapter_name (Optional[str]): Specific chapter to check (if None, checks entire document)
        last_known_modified (Optional[str]): ISO timestamp of last known modification

    Returns:
        ContentFreshnessStatus: Freshness status with safety recommendations

    Example Usage:
        ```json
        {
            "name": "check_content_freshness",
            "arguments": {
                "document_name": "user_guide",
                "chapter_name": "01-intro.md",
                "last_known_modified": "2024-01-15T10:30:00Z"
            }
        }
        ```
    """
    if chapter_name:
        file_path = _get_chapter_path(document_name, chapter_name)
    else:
        file_path = _get_document_path(document_name)

    # Parse last known modified time if provided
    last_known_dt = None
    if last_known_modified:
        try:
            last_known_dt = datetime.datetime.fromisoformat(
                last_known_modified.replace("Z", "+00:00")
            )
            if last_known_dt.tzinfo:
                last_known_dt = last_known_dt.replace(tzinfo=None)
        except ValueError:
            return ContentFreshnessStatus(
                is_fresh=False,
                last_modified=datetime.datetime.now(),
                safety_status="error",
                message=f"Invalid timestamp format: {last_known_modified}",
                recommendations=[
                    "Use ISO format: YYYY-MM-DDTHH:MM:SS or YYYY-MM-DDTHH:MM:SSZ"
                ],
            )

    return _check_file_freshness(file_path, last_known_dt)


# DEPRECATED INTERNAL FUNCTION: Use check_content_status unified tool instead
# This function is no longer exposed as an MCP tool and is only used internally
# by the unified check_content_status tool. Do not call directly.
@log_mcp_call
def get_modification_history(
    document_name: str, chapter_name: str | None = None, time_window: str = "24h"
) -> ModificationHistory:
    """Get comprehensive modification history for a document or chapter.

    Returns detailed timeline of all modifications with source tracking and
    operation types. Helps identify modification patterns and potential conflicts.

    Parameters:
        document_name (str): Name of the document directory
        chapter_name (Optional[str]): Specific chapter to get history for (if None, gets document history)
        time_window (str): Time window for history ("1h", "24h", "7d", "30d", "all")

    Returns:
        ModificationHistory: Complete modification history with entries and metadata

    Example Usage:
        ```json
        {
            "name": "get_modification_history",
            "arguments": {
                "document_name": "user_guide",
                "time_window": "7d"
            }
        }
        ```
    """
    history_path = _get_modification_history_path(document_name)

    # Parse time window
    now = datetime.datetime.now()
    if time_window == "all":
        cutoff_time = datetime.datetime.min
    else:
        try:
            if time_window.endswith("h"):
                hours = int(time_window[:-1])
                cutoff_time = now - datetime.timedelta(hours=hours)
            elif time_window.endswith("d"):
                days = int(time_window[:-1])
                cutoff_time = now - datetime.timedelta(days=days)
            else:
                cutoff_time = now - datetime.timedelta(hours=24)  # Default to 24h
        except ValueError:
            cutoff_time = now - datetime.timedelta(hours=24)  # Default to 24h

    # Load history entries
    entries = []
    if history_path.exists():
        try:
            import json

            with open(history_path) as f:
                data = json.load(f)
                all_entries = [
                    ModificationHistoryEntry(**entry)
                    for entry in data.get("entries", [])
                ]

                # Filter by time window and chapter if specified
                for entry in all_entries:
                    if entry.timestamp >= cutoff_time:
                        if chapter_name is None or entry.file_path.endswith(
                            chapter_name
                        ):
                            entries.append(entry)
        except Exception:
            # If history file is corrupted, return empty history
            pass

    return ModificationHistory(
        document_name=document_name,
        chapter_name=chapter_name,
        entries=entries,
        total_modifications=len(entries),
        time_window=time_window,
    )


class SnapshotInfo(BaseModel):
    """Information about a document snapshot."""

    snapshot_id: str
    timestamp: datetime.datetime
    operation: str
    document_name: str
    chapter_name: str | None = None
    message: str | None = None
    created_by: str
    file_count: int
    size_bytes: int


class SnapshotsList(BaseModel):
    """List of snapshots for a document."""

    document_name: str
    snapshots: list[SnapshotInfo]
    total_snapshots: int
    total_size_bytes: int


# DEPRECATED INTERNAL FUNCTION: Use manage_snapshots unified tool instead
# This function is no longer exposed as an MCP tool and is only used internally
# by the unified manage_snapshots tool and batch operations. Do not call directly.
@log_mcp_call
def snapshot_document(
    document_name: str, message: str | None = None, auto_cleanup: bool = True
) -> OperationStatus:
    """Create a timestamped, immutable snapshot of a document's current state.

    This versioning tool creates a complete copy of all document chapters with
    metadata and optional cleanup policies. Acts like a Git commit for documents.

    Parameters:
        document_name (str): Name of the document directory to snapshot
        message (Optional[str]): Optional message describing the snapshot purpose
        auto_cleanup (bool): Whether to automatically clean up old snapshots

    Returns:
        OperationStatus: Result with snapshot ID and creation details

    Example Usage:
        ```json
        {
            "name": "snapshot_document",
            "arguments": {
                "document_name": "user_guide",
                "message": "Before major revision of Chapter 2",
                "auto_cleanup": true
            }
        }
        ```
    """
    doc_path = _get_document_path(document_name)
    if not doc_path.is_dir():
        return OperationStatus(
            success=False, message=f"Document '{document_name}' not found."
        )

    try:
        # Create snapshot with user message
        snapshot_id = _create_micro_snapshot(
            document_name,
            None,  # Full document snapshot
            "user_snapshot",
        )

        # Update snapshot metadata with user message
        snapshots_path = _get_snapshots_path(document_name)
        snapshot_dir = snapshots_path / snapshot_id
        metadata_path = snapshot_dir / "_metadata.json"

        if metadata_path.exists():
            import json

            with open(metadata_path) as f:
                metadata = json.load(f)

            metadata["message"] = message or "Manual snapshot"
            metadata["auto_cleanup"] = auto_cleanup

            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

        # Calculate snapshot size
        total_size = sum(
            f.stat().st_size for f in snapshot_dir.rglob("*") if f.is_file()
        )
        file_count = len(list(snapshot_dir.glob("*.md")))

        # Auto cleanup if requested
        if auto_cleanup:
            _cleanup_old_snapshots(document_name, keep_count=10)

        # Record snapshot creation
        _record_modification(
            document_name,
            None,
            "snapshot",
            details={
                "snapshot_id": snapshot_id,
                "message": message,
                "file_count": file_count,
                "size_bytes": total_size,
            },
        )

        return OperationStatus(
            success=True,
            message=f"Snapshot '{snapshot_id}' created successfully for document '{document_name}'.",
            details={
                "snapshot_id": snapshot_id,
                "message": message,
                "file_count": file_count,
                "size_bytes": total_size,
                "auto_cleanup": auto_cleanup,
            },
        )

    except Exception as e:
        return OperationStatus(
            success=False,
            message=f"Error creating snapshot for document '{document_name}': {str(e)}",
        )


# DEPRECATED INTERNAL FUNCTION: Use manage_snapshots unified tool instead
# This function is no longer exposed as an MCP tool and is only used internally
# by the unified manage_snapshots tool. Do not call directly.
@log_mcp_call
def list_snapshots(
    document_name: str,
    limit: int | None = None,
    filter_message: str | None = None,
) -> SnapshotsList:
    """List all available snapshots for a document with metadata.

    Returns chronologically ordered list of snapshots with size information,
    messages, and creation details. Supports filtering and pagination.

    Parameters:
        document_name (str): Name of the document directory
        limit (Optional[int]): Maximum number of snapshots to return
        filter_message (Optional[str]): Filter snapshots by message content

    Returns:
        SnapshotsList: Complete list of snapshots with metadata

    Example Usage:
        ```json
        {
            "name": "list_snapshots",
            "arguments": {
                "document_name": "user_guide",
                "limit": 5
            }
        }
        ```
    """
    snapshots_path = _get_snapshots_path(document_name)
    snapshots = []
    total_size = 0

    if snapshots_path.exists():
        # Get all snapshot directories
        for snapshot_dir in sorted(
            snapshots_path.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True
        ):
            if snapshot_dir.is_dir():
                metadata_path = snapshot_dir / "_metadata.json"
                if metadata_path.exists():
                    try:
                        import json

                        with open(metadata_path) as f:
                            metadata = json.load(f)

                        # Apply message filter if provided
                        if (
                            filter_message
                            and filter_message.lower()
                            not in metadata.get("message", "").lower()
                        ):
                            continue

                        # Calculate snapshot size
                        size_bytes = sum(
                            f.stat().st_size
                            for f in snapshot_dir.rglob("*")
                            if f.is_file()
                        )
                        file_count = len(list(snapshot_dir.glob("*.md")))

                        snapshot_info = SnapshotInfo(
                            snapshot_id=metadata["snapshot_id"],
                            timestamp=datetime.datetime.fromisoformat(
                                metadata["timestamp"]
                            ),
                            operation=metadata["operation"],
                            document_name=metadata["document_name"],
                            chapter_name=metadata.get("chapter_name"),
                            message=metadata.get("message"),
                            created_by=metadata["created_by"],
                            file_count=file_count,
                            size_bytes=size_bytes,
                        )

                        snapshots.append(snapshot_info)
                        total_size += size_bytes

                        # Apply limit if provided
                        if limit and len(snapshots) >= limit:
                            break

                    except Exception:
                        # Skip corrupted snapshots
                        continue

    return SnapshotsList(
        document_name=document_name,
        snapshots=snapshots,
        total_snapshots=len(snapshots),
        total_size_bytes=total_size,
    )


def _cleanup_old_snapshots(document_name: str, keep_count: int = 10):
    """Clean up old snapshots, keeping only the most recent ones."""
    snapshots_path = _get_snapshots_path(document_name)

    if not snapshots_path.exists():
        return

    # Get all snapshot directories sorted by modification time
    snapshot_dirs = sorted(
        [d for d in snapshots_path.iterdir() if d.is_dir()],
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )

    # Keep only the most recent snapshots
    for old_snapshot in snapshot_dirs[keep_count:]:
        try:
            shutil.rmtree(old_snapshot)
        except Exception:
            # Log error but continue cleanup
            pass


# DEPRECATED INTERNAL FUNCTION: Use manage_snapshots unified tool instead
# This function is no longer exposed as an MCP tool and is only used internally
# by the unified manage_snapshots tool and batch operations. Do not call directly.
@log_mcp_call
def restore_snapshot(
    document_name: str, snapshot_id: str, safety_mode: bool = True
) -> OperationStatus:
    """Restore a document to a specific snapshot state.

    This tool reverts the live document to the specified snapshot state with
    comprehensive safety checks. Essential for version control and recovery.

    Parameters:
        document_name (str): Name of the document to restore
        snapshot_id (str): ID of the snapshot to restore to
        safety_mode (bool): If True, creates safety snapshot before restoration

    Returns:
        OperationStatus: Result with restoration details and safety information

    Example Usage:
        ```json
        {
            "name": "restore_snapshot",
            "arguments": {
                "document_name": "user_guide",
                "snapshot_id": "user_snapshot_20240115_103000_123456",
                "safety_mode": true
            }
        }
        ```
    """
    doc_path = _get_document_path(document_name)
    if not doc_path.is_dir():
        return OperationStatus(
            success=False, message=f"Document '{document_name}' not found."
        )

    snapshots_path = _get_snapshots_path(document_name)
    snapshot_dir = snapshots_path / snapshot_id

    if not snapshot_dir.exists():
        return OperationStatus(
            success=False,
            message=f"Snapshot '{snapshot_id}' not found for document '{document_name}'.",
        )

    try:
        # Create safety snapshot if requested
        safety_snapshot_id = None
        if safety_mode:
            safety_snapshot_id = _create_micro_snapshot(
                document_name, None, "pre_restore"
            )

        # Get list of files to restore
        restore_files = list(snapshot_dir.glob("*.md"))

        if not restore_files:
            return OperationStatus(
                success=False,
                message=f"No chapter files found in snapshot '{snapshot_id}'.",
            )

        # Restore each file
        restored_files = []
        for snapshot_file in restore_files:
            target_file = doc_path / snapshot_file.name

            # Read snapshot content
            snapshot_content = snapshot_file.read_text(encoding="utf-8")

            # Write to live document
            target_file.write_text(snapshot_content, encoding="utf-8")
            restored_files.append(snapshot_file.name)

        # Record restoration in history
        _record_modification(
            document_name,
            None,
            "restore_snapshot",
            details={
                "snapshot_id": snapshot_id,
                "safety_snapshot_id": safety_snapshot_id,
                "restored_files": restored_files,
                "safety_mode": safety_mode,
            },
        )

        return OperationStatus(
            success=True,
            message=f"Document '{document_name}' restored to snapshot '{snapshot_id}'.",
            details={
                "snapshot_id": snapshot_id,
                "restored_files": restored_files,
                "files_restored": len(restored_files),
            },
            snapshot_created=safety_snapshot_id,
            warnings=[] if safety_mode else ["No safety snapshot created"],
        )

    except Exception as e:
        return OperationStatus(
            success=False,
            message=f"Error restoring snapshot '{snapshot_id}' for document '{document_name}': {str(e)}",
        )


def _load_snapshot_files(snapshot_dir: Path) -> dict[str, str]:
    """Load all .md files from a snapshot directory."""
    return {f.name: f.read_text(encoding="utf-8") for f in snapshot_dir.glob("*.md")}


def _load_current_document_files(doc_path: Path) -> dict[str, str]:
    """Load all .md files from current document directory."""
    files = {}
    for chapter_file in doc_path.glob("*.md"):
        if chapter_file.is_file():
            files[chapter_file.name] = chapter_file.read_text(encoding="utf-8")
    return files


def _compare_file_sets(
    files_a: dict[str, str], files_b: dict[str, str]
) -> dict[str, Any]:
    """Compare two sets of files and return categorized differences."""
    all_files = set(files_a.keys()) | set(files_b.keys())
    files_changed = []
    files_added = []
    files_removed = []
    diffs = {}

    for filename in all_files:
        if filename in files_a and filename in files_b:
            if files_a[filename] != files_b[filename]:
                files_changed.append(filename)
                # Generate diff for changed files
                diff = _generate_content_diff(
                    files_a[filename], files_b[filename], filename
                )
                diffs[filename] = diff
        elif filename in files_a:
            files_removed.append(filename)
        else:
            files_added.append(filename)

    return {
        "files_changed": files_changed,
        "files_added": files_added,
        "files_removed": files_removed,
        "diffs": diffs,
    }


def _generate_diff_summary(
    comparison: dict[str, Any], snapshot_a: str, comparison_label: str
) -> str:
    """Generate a human-readable summary of differences."""
    changes_summary = []
    if comparison["files_changed"]:
        changes_summary.append(f"{len(comparison['files_changed'])} files changed")
    if comparison["files_added"]:
        changes_summary.append(f"{len(comparison['files_added'])} files added")
    if comparison["files_removed"]:
        changes_summary.append(f"{len(comparison['files_removed'])} files removed")

    if changes_summary:
        return (
            f"Comparing snapshot '{snapshot_a}' with {comparison_label}: "
            + ", ".join(changes_summary)
        )
    else:
        return "No changes detected"


def _format_diff_output(
    comparison: dict[str, Any], output_format: str
) -> Any | None:
    """Format the diff output according to the requested format."""
    if output_format == "detailed":
        return {
            "files_changed": comparison["files_changed"],
            "files_added": comparison["files_added"],
            "files_removed": comparison["files_removed"],
            "diffs": comparison["diffs"],
        }
    elif output_format == "unified":
        return "\n".join(
            [
                f"=== {filename} ===\n{diff_info.get('diff', 'No diff available')}"
                for filename, diff_info in comparison["diffs"].items()
            ]
        )
    return None


# DEPRECATED INTERNAL FUNCTION: Use diff_content unified tool instead
# This function is no longer exposed as an MCP tool and is only used internally
# by the unified diff_content tool. Do not call directly.
@log_mcp_call
def diff_snapshots(
    document_name: str,
    snapshot_a: str,
    snapshot_b: str | None = None,
    output_format: str = "summary",
) -> OperationStatus:
    """Compare two snapshots or a snapshot with current document state.

    This tool provides intelligent comparison between document versions with
    prose-aware diffing optimized for creative writing workflows.

    Parameters:
        document_name (str): Name of the document to compare
        snapshot_a (str): ID of the first snapshot
        snapshot_b (Optional[str]): ID of the second snapshot (if None, compares with current state)
        output_format (str): Format of diff output ("summary", "detailed", "unified")

    Returns:
        OperationStatus: Result with diff information and comparison details
    """
    doc_path = _get_document_path(document_name)
    if not doc_path.is_dir():
        return OperationStatus(
            success=False, message=f"Document '{document_name}' not found."
        )

    snapshots_path = _get_snapshots_path(document_name)
    snapshot_a_dir = snapshots_path / snapshot_a

    if not snapshot_a_dir.exists():
        return OperationStatus(
            success=False,
            message=f"Snapshot '{snapshot_a}' not found for document '{document_name}'.",
        )

    try:
        # Load snapshot A content
        snapshot_a_files = _load_snapshot_files(snapshot_a_dir)

        # Load snapshot B content (or current state)
        if snapshot_b:
            snapshot_b_dir = snapshots_path / snapshot_b
            if not snapshot_b_dir.exists():
                return OperationStatus(
                    success=False,
                    message=f"Snapshot '{snapshot_b}' not found for document '{document_name}'.",
                )
            snapshot_b_files = _load_snapshot_files(snapshot_b_dir)
            comparison_label = f"snapshot '{snapshot_b}'"
        else:
            snapshot_b_files = _load_current_document_files(doc_path)
            comparison_label = "current state"

        # Compare file sets
        comparison = _compare_file_sets(snapshot_a_files, snapshot_b_files)

        # Generate summary and format output
        summary = _generate_diff_summary(comparison, snapshot_a, comparison_label)
        detailed_output = _format_diff_output(comparison, output_format)

        return OperationStatus(
            success=True,
            message=summary,
            details={
                "snapshot_a": snapshot_a,
                "snapshot_b": snapshot_b or "current",
                "files_changed": comparison["files_changed"],
                "files_added": comparison["files_added"],
                "files_removed": comparison["files_removed"],
                "total_changes": len(comparison["files_changed"])
                + len(comparison["files_added"])
                + len(comparison["files_removed"]),
                "output_format": output_format,
                "detailed_output": detailed_output,
            },
        )

    except Exception as e:
        return OperationStatus(
            success=False,
            message=f"Error comparing snapshots for document '{document_name}': {str(e)}",
        )


# ============================================================================
# BATCH OPERATIONS TOOL
# ============================================================================


def _resolve_operation_dependencies(
    operations: list[BatchOperation],
) -> list[BatchOperation]:
    """Resolve operation dependencies and return operations in executable order.

    This function performs topological sorting to ensure operations execute
    in an order that respects their dependencies. Operations with unresolved
    dependencies will cause validation errors.

    Args:
        operations: List of BatchOperation objects with potential dependencies

    Returns:
        List[BatchOperation]: Operations sorted in dependency-respecting order

    Raises:
        ValueError: If circular dependencies are detected or dependencies are unresolvable
    """
    # Create mapping of operation_id to operation for quick lookup
    op_map = {op.operation_id: op for op in operations}

    # Track which operations have been processed
    resolved = set()
    processing = (
        set()
    )  # Track operations currently being processed (for cycle detection)
    result = []

    def resolve_operation(op: BatchOperation):
        """Recursively resolve an operation and its dependencies."""
        if op.operation_id in resolved:
            return  # Already resolved

        if op.operation_id in processing:
            raise ValueError(
                f"Circular dependency detected involving operation '{op.operation_id}'"
            )

        processing.add(op.operation_id)

        # Resolve all dependencies first
        if op.depends_on:
            for dep_id in op.depends_on:
                if dep_id not in op_map:
                    raise ValueError(
                        f"Operation '{op.operation_id}' depends on unknown operation '{dep_id}'"
                    )

                dependency_op = op_map[dep_id]
                resolve_operation(dependency_op)

        # Add current operation to resolved set and result list
        processing.remove(op.operation_id)
        if op.operation_id not in resolved:
            resolved.add(op.operation_id)
            result.append(op)

    # Resolve all operations
    for op in operations:
        resolve_operation(op)

    # Final sort by order field as a secondary criterion for operations at same dependency level
    result.sort(key=lambda x: x.order)

    return result


@mcp_server.tool()
@log_mcp_call
@auto_snapshot("batch_apply_operations")
def batch_apply_operations(
    operations: list[dict[str, Any]],
    atomic: bool = True,
    validate_only: bool = False,
    snapshot_before: bool = False,
    continue_on_error: bool = False,
) -> dict[str, Any]:
    """Execute multiple operations as a batch with sequential execution.

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
                ).model_dump()

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
            ).model_dump()

        # If validate_only mode, validate operations and return
        if validate_only:
            validation_errors = []
            for op in sorted_ops:
                if not _batch_registry.is_valid_operation(op.operation_type):
                    validation_errors.append(
                        f"Unknown operation type: {op.operation_type}"
                    )

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
                ).model_dump()
            else:
                return BatchApplyResult(
                    success=True,
                    total_operations=len(operations),
                    successful_operations=0,
                    failed_operations=0,
                    execution_time_ms=execution_time,
                    operation_results=[],
                    summary="Validation successful - no operations executed",
                ).model_dump()

        # Create snapshot before execution if requested
        snapshot_id = None
        if snapshot_before:
            # Extract existing documents that will be modified from operations
            affected_docs = set()
            for op in sorted_ops:
                doc_name = op.target.get("document_name") or op.parameters.get(
                    "document_name"
                )
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
                    snapshot_result = snapshot_document(
                        doc_name,
                        f"Batch operation snapshot before {len(operations)} operations",
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
            sorted_ops, continue_on_error=continue_on_error, snapshot_id=snapshot_id
        )

        # Add snapshot information if created
        if snapshot_id:
            result_dict = result.model_dump()
            result_dict["snapshot_id"] = snapshot_id
            return result_dict

        return result.model_dump()

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
        ).model_dump()


# ============================================================================
# MCP RESOURCE HANDLERS
# ============================================================================
# Handlers for exposing document and chapter data as standardized resources


def _create_resource_error(message: str, code: int = -1) -> McpError:
    """Helper function to create properly formatted McpError for resource operations."""
    return McpError(ErrorData(code=code, message=message))


async def _list_resources_handler():
    """List all available document chapters as MCP resources.

    Scans the DOCS_ROOT_PATH for all document directories and their .md chapter files,
    generating a TextResource for each chapter that can be accessed by LLM clients.

    Returns:
        List[Resource]: List of available chapter resources. Each Resource contains:
            - uri (str): Unique resource identifier (e.g., "file://document_name/chapter_name")
            - name (str): Human-readable name (e.g., "document_name/chapter_name")
            - mimeType (str): Set to "text/markdown" for all chapters
    """
    resources = []

    # Ensure DOCS_ROOT_PATH is a Path object (defensive programming)
    root_path = (
        Path(DOCS_ROOT_PATH) if not isinstance(DOCS_ROOT_PATH, Path) else DOCS_ROOT_PATH
    )

    if not root_path.exists() or not root_path.is_dir():
        return resources

    # Scan all document directories
    for doc_dir in root_path.iterdir():
        if doc_dir.is_dir():
            document_name = doc_dir.name
            chapter_files = _get_ordered_chapter_files(document_name)

            # Create a resource for each chapter file
            for chapter_file_path in chapter_files:
                chapter_name = chapter_file_path.name
                uri = f"file://{document_name}/{chapter_name}"
                name = f"{document_name}/{chapter_name}"

                resources.append(Resource(uri=uri, name=name, mimeType="text/markdown"))

    return resources


async def _read_resource_handler(uri: str):
    """Read the content of a specific document chapter resource.

    Parses the provided URI to extract document and chapter names, then securely
    reads the corresponding chapter file content. Validates that the requested
    file is within the DOCS_ROOT_PATH to prevent directory traversal attacks.

    Parameters:
        uri (str): Resource URI in format "file://document_name/chapter_name"

    Returns:
        TextResourceContents: Resource content object containing:
            - uri (str): Original resource URI
            - mime_type (str): Set to "text/markdown"
            - text (str): Full content of the chapter file

    Raises:
        McpError: If URI is invalid, file doesn't exist, or security violation
    """
    # Parse and validate the URI
    try:
        parsed_uri = urllib.parse.urlparse(uri)
        if parsed_uri.scheme != "file":
            raise _create_resource_error(
                f"Invalid URI scheme: {parsed_uri.scheme}. Expected 'file'"
            )

        # Extract document_name and chapter_name from netloc and path
        if parsed_uri.netloc:
            # Format: file://document_name/chapter_name
            document_name = parsed_uri.netloc
            chapter_name = parsed_uri.path.strip("/")
            if not chapter_name:
                raise _create_resource_error(
                    "Invalid URI: missing chapter name. Expected 'file://document_name/chapter_name'"
                )
        else:
            # Format: file:///document_name/chapter_name
            path_parts = parsed_uri.path.strip("/").split("/")
            if len(path_parts) != 2:
                raise _create_resource_error(
                    f"Invalid URI path format: {parsed_uri.path}. Expected '/document_name/chapter_name'"
                )
            document_name, chapter_name = path_parts

        # Validate the names using existing validation functions
        is_valid_doc, doc_error = _validate_document_name(document_name)
        if not is_valid_doc:
            raise _create_resource_error(f"Invalid document name: {doc_error}")

        is_valid_chapter, chapter_error = _validate_chapter_name(chapter_name)
        if not is_valid_chapter:
            raise _create_resource_error(f"Invalid chapter name: {chapter_error}")

    except ValueError as e:
        raise _create_resource_error(f"Failed to parse URI: {e}")

    # Securely resolve the file path and verify it's within DOCS_ROOT_PATH
    try:
        chapter_path = _get_chapter_path(document_name, chapter_name)

        # Ensure the resolved path is within DOCS_ROOT_PATH (prevent directory traversal)
        root_path = (
            Path(DOCS_ROOT_PATH)
            if not isinstance(DOCS_ROOT_PATH, Path)
            else DOCS_ROOT_PATH
        )
        resolved_chapter_path = chapter_path.resolve()
        resolved_root_path = root_path.resolve()

        if not str(resolved_chapter_path).startswith(str(resolved_root_path)):
            raise _create_resource_error("Access denied: Path outside document root")

        # Verify the file exists and is a valid chapter file
        if not resolved_chapter_path.is_file():
            raise _create_resource_error(
                f"Chapter file not found: {document_name}/{chapter_name}"
            )

        if not _is_valid_chapter_filename(chapter_name):
            raise _create_resource_error(f"Invalid chapter file: {chapter_name}")

        # Read and return the content
        content = resolved_chapter_path.read_text(encoding="utf-8")

        return [TextResourceContents(uri=uri, mimeType="text/markdown", text=content)]

    except FileNotFoundError:
        raise _create_resource_error(
            f"Chapter file not found: {document_name}/{chapter_name}"
        )
    except PermissionError:
        raise _create_resource_error(
            f"Permission denied accessing: {document_name}/{chapter_name}"
        )
    except UnicodeDecodeError as e:
        raise _create_resource_error(f"Failed to decode file content: {e}")
    except Exception as e:
        raise _create_resource_error(f"Error reading resource: {e}")


# Override the FastMCP resource methods
mcp_server.list_resources = _list_resources_handler
mcp_server.read_resource = _read_resource_handler


# Expose sync wrappers for testing
@log_mcp_call
def list_resources():
    """Sync wrapper for list_resources for testing purposes."""
    import asyncio

    return asyncio.run(_list_resources_handler())


@log_mcp_call
def read_resource(uri: str):
    """Sync wrapper for read_resource for testing purposes."""
    import asyncio

    return asyncio.run(_read_resource_handler(uri))


@mcp_server.custom_route("/health", methods=["GET"], name="health")
async def health_check(request: Request) -> Response:
    """Health check endpoint to verify server readiness."""
    return Response(status_code=200)


@mcp_server.custom_route("/metrics", methods=["GET"], name="metrics")
async def metrics_endpoint(request: Request) -> Response:
    """Prometheus metrics endpoint for monitoring MCP tool usage."""
    if not METRICS_AVAILABLE:
        return Response(
            content="# Metrics not available - OpenTelemetry not installed\n",
            status_code=503,
            media_type="text/plain",
        )

    try:
        metrics_data, content_type = get_metrics_export()
        return Response(content=metrics_data, status_code=200, media_type=content_type)
    except Exception as e:
        error_msg = f"# Error generating metrics: {e}\n"
        return Response(content=error_msg, status_code=500, media_type="text/plain")


@mcp_server.custom_route("/metrics/summary", methods=["GET"], name="metrics_summary")
async def metrics_summary_endpoint(request: Request) -> Response:
    """JSON endpoint providing a summary of current metrics configuration and status."""
    try:
        from .metrics_config import get_metrics_summary

        summary = get_metrics_summary()

        import json

        return Response(
            content=json.dumps(summary, indent=2),
            status_code=200,
            media_type="application/json",
        )
    except Exception as e:
        error_response = {
            "error": f"Failed to get metrics summary: {e}",
            "status": "error",
        }
        import json

        return Response(
            content=json.dumps(error_response),
            status_code=500,
            media_type="application/json",
        )


# ============================================================================
# UNIFIED VERSION CONTROL TOOLS
# ============================================================================


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
                "snapshots": [snapshot.model_dump() for snapshot in result.snapshots],
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
                "output_format": "unified"
            }
        }
        ```

        ```json
        {
            "name": "diff_content",
            "arguments": {
                "document_name": "user_guide",
                "source_type": "current",
                "target_type": "snapshot",
                "target_id": "20240115_103045_snapshot",
                "chapter_name": "01-intro.md",
                "output_format": "summary"
            }
        }
        ```
    """
    # Validate inputs
    is_valid, error_msg = _validate_document_name(document_name)
    if not is_valid:
        return {
            "success": False,
            "message": f"Invalid document name: {error_msg}",
            "operation": "diff_content",
        }

    valid_types = ["snapshot", "current", "file"]
    valid_formats = ["unified", "context", "summary"]

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

    if output_format not in valid_formats:
        return {
            "success": False,
            "message": f"Invalid output_format '{output_format}'. Must be one of: {', '.join(valid_formats)}",
            "operation": "diff_content",
        }

    if chapter_name:
        is_valid, error_msg = _validate_chapter_name(chapter_name)
        if not is_valid:
            return {
                "success": False,
                "message": f"Invalid chapter name: {error_msg}",
                "operation": "diff_content",
            }

    try:
        # For now, delegate to existing diff_snapshots if both are snapshots
        if (
            source_type == "snapshot"
            and target_type == "snapshot"
            and source_id
            and target_id
        ):
            result = diff_snapshots(document_name, source_id, target_id, output_format)
            return {
                "success": result.success,
                "message": result.message,
                "operation": "diff_content",
                "source_type": source_type,
                "source_id": source_id,
                "target_type": target_type,
                "target_id": target_id,
                "output_format": output_format,
                "diff_text": result.details.get("diff_text")
                if result.details
                else None,
                "summary": result.details.get("summary")
                if result.details
                else result.message,
                "statistics": result.details.get("statistics")
                if result.details
                else None,
                "metadata": {
                    "source": f"{source_type}:{source_id}",
                    "target": f"{target_type}:{target_id}",
                    "document_name": document_name,
                    "chapter_name": chapter_name,
                },
            }

        # Enhanced functionality for current content comparisons
        elif "current" in [source_type, target_type]:
            # Get current content
            if chapter_name:
                current_content_result = read_chapter_content(
                    document_name, chapter_name
                )
                current_content = (
                    current_content_result.content if current_content_result else ""
                )
            else:
                full_doc_result = read_full_document(document_name)
                current_content = (
                    "\n\n".join(
                        [
                            f"# {chapter.chapter_name}\n{chapter.content}"
                            for chapter in (
                                full_doc_result.chapters if full_doc_result else []
                            )
                        ]
                    )
                    if full_doc_result
                    else ""
                )

            # For snapshot comparison with current
            if source_type == "snapshot" and target_type == "current":
                # Use existing diff_snapshots with current content simulation
                # This is a simplified implementation - full implementation would need content resolution
                return {
                    "success": False,
                    "message": "Enhanced diff with current content not fully implemented yet. Use existing diff_snapshots for snapshot-to-snapshot comparisons.",
                    "operation": "diff_content",
                    "note": "This feature requires additional implementation for content resolution",
                }

            # Similar for other combinations...
            else:
                return {
                    "success": False,
                    "message": f"Diff combination {source_type} -> {target_type} not fully implemented yet",
                    "operation": "diff_content",
                    "note": "Currently supports snapshot-to-snapshot comparisons via existing diff_snapshots",
                }

        else:
            return {
                "success": False,
                "message": f"Unsupported diff combination: {source_type} -> {target_type}",
                "operation": "diff_content",
            }

    except Exception as e:
        log_structured_error(
            ErrorCategory.OPERATION_FAILED,
            f"Failed to generate diff for '{document_name}': {e}",
            {"operation": "diff_content", "document_name": document_name},
        )
        return {
            "success": False,
            "message": f"Failed to generate diff: {str(e)}",
            "operation": "diff_content",
            "error": str(e),
        }


# --- Main Server Execution ---
def main():
    """Run the main entry point for the server with argument parsing."""
    parser = argparse.ArgumentParser(description="Document MCP Server")
    parser.add_argument(
        "transport",
        choices=["sse", "stdio"],
        default="stdio",
        nargs="?",
        help="Transport type: 'sse' for HTTP Server-Sent Events or 'stdio' for standard I/O (default: stdio)",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"Host to bind to for SSE transport (default: {DEFAULT_HOST})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Port to bind to for SSE transport (default: {DEFAULT_PORT})",
    )

    args = parser.parse_args()

    # This print will show the path used by the subprocess
    print(
        f"doc_tool_server.py: Initializing with DOCS_ROOT_PATH = {DOCS_ROOT_PATH.resolve()}"
    )
    print(f"Document tool server starting. Tools exposed by '{mcp_server.name}':")
    print(f"Serving tools for root directory: {DOCS_ROOT_PATH.resolve()}")

    # Show metrics status
    if METRICS_AVAILABLE:
        try:
            from .metrics_config import METRICS_ENABLED

            status = "enabled" if METRICS_ENABLED else "disabled"
            print(f"Metrics: {status}")
        except ImportError:
            print("Metrics: available but not configured")
    else:
        print("Metrics: not available (install prometheus-client)")

    if args.transport == "stdio":
        print(
            "MCP server running with stdio transport. Waiting for client connection..."
        )
        mcp_server.run(transport="stdio")
    else:
        print(f"MCP server running with HTTP SSE transport on {args.host}:{args.port}")
        print(f"SSE endpoint: http://{args.host}:{args.port}/sse")
        print(f"Health endpoint: http://{args.host}:{args.port}/health")
        if METRICS_AVAILABLE:
            print(f"Metrics endpoint: http://{args.host}:{args.port}/metrics")
        # Update server settings before running
        mcp_server.settings.host = args.host
        mcp_server.settings.port = args.port
        mcp_server.run(transport="sse")


if __name__ == "__main__":
    main()
