"""Input validation utilities for the Document MCP system.

This module contains all validation functions used throughout the system
to ensure data integrity and security.
"""

import datetime
from pathlib import Path

from ..models import ContentFreshnessStatus

# === Configuration Constants ===

MAX_DOCUMENT_NAME_LENGTH = 100
MAX_CHAPTER_NAME_LENGTH = 100
MAX_CONTENT_LENGTH = 1_000_000  # 1MB max content
MIN_PARAGRAPH_INDEX = 0
CHAPTER_MANIFEST_FILE = "_manifest.json"


# === Validation Functions ===


def validate_document_name(document_name: str) -> tuple[bool, str]:
    """Validate document name input."""
    if not document_name or not isinstance(document_name, str) or not document_name.strip():
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


def validate_chapter_name(chapter_name: str) -> tuple[bool, str]:
    """Validate chapter name input."""
    if not chapter_name or not isinstance(chapter_name, str) or not chapter_name.strip():
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


def validate_content(content: str) -> tuple[bool, str]:
    """Validate content input."""
    if content is None:
        return False, "Content cannot be None"
    if not isinstance(content, str):
        return False, "Content must be a string"
    if len(content) > MAX_CONTENT_LENGTH:
        return False, f"Content too long (max {MAX_CONTENT_LENGTH} characters)"
    return True, ""


def validate_paragraph_index(index: int) -> tuple[bool, str]:
    """Validate paragraph index input."""
    if not isinstance(index, int):
        return False, "Paragraph index must be an integer"
    if index < MIN_PARAGRAPH_INDEX:
        return False, "Paragraph index cannot be negative"
    return True, ""


def validate_search_query(query: str) -> tuple[bool, str]:
    """Validate search query input."""
    if query is None:
        return False, "Search query cannot be None"
    if not isinstance(query, str):
        return False, "Search query must be a string"
    if not query.strip():
        return False, "Search query cannot be empty or whitespace only"
    if len(query) > 1000:  # Reasonable limit for search queries
        return False, "Search query too long (max 1000 characters)"
    return True, ""


def parse_timestamp(timestamp_str: str) -> datetime.datetime | None:
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


def check_file_freshness(
    file_path: Path, last_known_modified: datetime.datetime | None = None
) -> ContentFreshnessStatus:
    """Check if a file has been modified externally since last known modification.

    Args:
        file_path: Path to the file to check
        last_known_modified: Last known modification time from the client

    Returns:
        ContentFreshnessStatus with freshness information and recommendations
    """
    if not file_path.exists():
        return ContentFreshnessStatus(
            is_fresh=False,
            last_modified=datetime.datetime.now(),
            safety_status="conflict",
            message="File no longer exists",
            recommendations=[
                "Verify file was not accidentally deleted",
                "Consider restoring from snapshot",
            ],
        )

    actual_modified = datetime.datetime.fromtimestamp(file_path.stat().st_mtime)

    if last_known_modified is None:
        return ContentFreshnessStatus(
            is_fresh=True,
            last_modified=actual_modified,
            safety_status="safe",
            message="No previous modification time to compare",
            recommendations=[],
        )

    # Allow small time differences (1 second) for filesystem precision
    time_diff = abs((actual_modified - last_known_modified).total_seconds())

    if time_diff <= 1.0:
        return ContentFreshnessStatus(
            is_fresh=True,
            last_modified=actual_modified,
            last_known_modified=last_known_modified,
            safety_status="safe",
            message="File is up to date",
            recommendations=[],
        )
    elif actual_modified > last_known_modified:
        return ContentFreshnessStatus(
            is_fresh=False,
            last_modified=actual_modified,
            last_known_modified=last_known_modified,
            safety_status="warning",
            message="Content was modified externally",
            recommendations=[
                "Review changes before proceeding",
                "Consider creating a backup",
                "Use force_write=True to override",
            ],
        )
    else:
        return ContentFreshnessStatus(
            is_fresh=False,
            last_modified=actual_modified,
            last_known_modified=last_known_modified,
            safety_status="conflict",
            message="File appears to be older than expected",
            recommendations=[
                "Check system clock",
                "Verify file integrity",
                "Consider refreshing file reference",
            ],
        )
