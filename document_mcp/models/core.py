"""Core models for the Document MCP system.

This module contains base models and core operation status models
that are used throughout the system.
"""

import datetime
from typing import Any

from pydantic import BaseModel

__all__ = [
    "OperationStatus",
    "ParagraphDetail",
    "ContentFreshnessStatus",
    "ModificationHistoryEntry",
    "ModificationHistory",
]


class OperationStatus(BaseModel):
    """Generic status for operations with optional safety information."""

    success: bool
    message: str
    details: dict[str, Any] | None = None  # For extra info, e.g., created entity name
    safety_info: Any | None = None
    snapshot_created: str | None = None
    warnings: list[str] = []


class ParagraphDetail(BaseModel):
    """Detailed information about a paragraph."""

    document_name: str
    chapter_name: str
    paragraph_index_in_chapter: int  # 0-indexed within its chapter
    content: str
    word_count: int


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
