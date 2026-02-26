"""Context Management Models for OneContext-inspired system.

This module contains Pydantic models for managing document context,
including memory entries, session metadata, and import/export operations.
"""

from __future__ import annotations

import datetime
from typing import Any

from pydantic import BaseModel
from pydantic import Field

__all__ = [
    "MemoryEntry",
    "SessionMetadata",
    "ExportStatus",
    "ImportStatus",
]


class MemoryEntry(BaseModel):
    """A single memory entry in the context system.

    Represents a key-value pair with metadata about storage, retrieval,
    and lifecycle management.
    """

    key: str = Field(description="Unique identifier for this memory entry")
    value: Any = Field(description="The stored value (string, JSON, or other serializable data)")
    stored_at: datetime.datetime = Field(
        description="Timestamp when memory was first stored"
    )
    retrieved_at: datetime.datetime | None = Field(
        default=None, description="Timestamp of most recent retrieval"
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Optional tags for organizing and filtering memories",
    )
    expires: datetime.datetime | None = Field(
        default=None, description="Optional expiration timestamp"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about this memory entry",
    )

    class Config:
        """Pydantic config."""

        json_encoders = {
            datetime.datetime: lambda v: v.isoformat(),
        }


class SessionMetadata(BaseModel):
    """Metadata for a document context session.

    Tracks session state, goals, progress, and blockers for a document.
    """

    session_id: str = Field(description="Unique session identifier")
    document_name: str = Field(description="Associated document name")
    started_at: datetime.datetime = Field(description="Session start timestamp")
    last_activity: datetime.datetime = Field(description="Timestamp of last activity")
    goals: list[str] = Field(
        default_factory=list,
        description="Current session goals/objectives",
    )
    progress: dict[str, Any] = Field(
        default_factory=dict,
        description="Progress tracking for current work",
    )
    blockers: list[str] = Field(
        default_factory=list,
        description="Current blockers or issues",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional session metadata",
    )

    class Config:
        """Pydantic config."""

        json_encoders = {
            datetime.datetime: lambda v: v.isoformat(),
        }


class ExportStatus(BaseModel):
    """Status of a context export operation.

    Indicates success and provides details about exported context package.
    """

    success: bool = Field(
        description="Whether export completed successfully"
    )
    exported_file_path: str | None = Field(
        default=None,
        description="Path to exported context file",
    )
    file_size: int | None = Field(
        default=None,
        description="Size of exported file in bytes",
    )
    entry_count: int = Field(
        default=0,
        description="Number of memory entries exported",
    )
    format_used: str = Field(
        description="Format of export (json, yaml, or markdown)",
    )
    message: str = Field(
        description="Human-readable status message",
    )
    timestamp: datetime.datetime = Field(
        default_factory=datetime.datetime.utcnow,
        description="When export was performed",
    )

    class Config:
        """Pydantic config."""

        json_encoders = {
            datetime.datetime: lambda v: v.isoformat(),
        }


class ImportStatus(BaseModel):
    """Status of a context import operation.

    Indicates success, conflicts, and details about imported entries.
    """

    success: bool = Field(
        description="Whether import completed successfully"
    )
    entries_imported: int = Field(
        default=0,
        description="Number of memory entries imported",
    )
    conflicts_detected: int = Field(
        default=0,
        description="Number of key conflicts found",
    )
    conflict_details: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Details of detected conflicts",
    )
    message: str = Field(
        description="Human-readable status message",
    )
    merge_mode_used: bool = Field(
        default=False,
        description="Whether merge mode was used (vs replace)",
    )
    timestamp: datetime.datetime = Field(
        default_factory=datetime.datetime.utcnow,
        description="When import was performed",
    )

    class Config:
        """Pydantic config."""

        json_encoders = {
            datetime.datetime: lambda v: v.isoformat(),
        }
