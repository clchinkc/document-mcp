"""Version control and Git-based models for the Document MCP system.

This module contains models for Git-backed version history, commits, and diffs.
"""

from __future__ import annotations

import datetime
from typing import Any

from pydantic import BaseModel
from pydantic import Field

__all__ = [
    "CommitInfo",
    "VersionHistory",
    "VersionDiff",
    "VersionComparisonResult",
]


class CommitInfo(BaseModel):
    """Information about a Git commit."""

    hash: str = Field(..., description="Commit SHA hash (40 hex characters)")
    author: str = Field(..., description="Author name and email")
    timestamp: datetime.datetime = Field(..., description="Commit timestamp")
    message: str = Field(..., description="Full commit message")
    summary: str = Field(..., description="First line of commit message")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "hash": self.hash,
            "author": self.author,
            "timestamp": self.timestamp.isoformat(),
            "message": self.message,
            "summary": self.summary,
        }


class VersionHistory(BaseModel):
    """Git commit history for a document."""

    document_name: str = Field(..., description="Name of the document")
    total_commits: int = Field(..., description="Total number of commits")
    commits: list[CommitInfo] = Field(..., description="List of commits (newest first)")
    time_window: str = Field(default="all", description="Time window for history (e.g., '24h', 'all')")

    class Config:
        """Pydantic configuration."""

        json_encoders = {datetime.datetime: lambda v: v.isoformat()}


class VersionDiff(BaseModel):
    """Diff between two Git commits."""

    source_hash: str = Field(
        ...,
        description="Source commit hash (or 'working' for working tree)",
    )
    target_hash: str = Field(
        ...,
        description="Target commit hash or 'HEAD'",
    )
    diff_text: str = Field(..., description="Unified diff format text")
    files_changed: int = Field(..., description="Number of files changed")
    insertions: int = Field(..., description="Number of lines added")
    deletions: int = Field(..., description="Number of lines removed")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "source_hash": self.source_hash,
            "target_hash": self.target_hash,
            "diff_text": self.diff_text,
            "files_changed": self.files_changed,
            "insertions": self.insertions,
            "deletions": self.deletions,
        }


class VersionComparisonResult(BaseModel):
    """Result of comparing two versions."""

    document_name: str = Field(..., description="Name of the document")
    version1: str = Field(..., description="First version hash or tag")
    version2: str = Field(..., description="Second version hash or tag")
    diff: VersionDiff = Field(..., description="Diff between versions")
    has_changes: bool = Field(..., description="Whether any changes exist between versions")
    summary: str = Field(..., description="Human-readable comparison summary")

    class Config:
        """Pydantic configuration."""

        json_encoders = {datetime.datetime: lambda v: v.isoformat()}
