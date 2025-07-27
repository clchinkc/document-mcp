"""Document-related models for the Document MCP system.

This module contains models for document metadata, structure,
and document-level operations.
"""

import datetime

from pydantic import BaseModel

__all__ = [
    "ChapterMetadata",
    "DocumentInfo",
    "DocumentSummary",
    "SnapshotInfo",
    "SnapshotsList",
]


class ChapterMetadata(BaseModel):
    """Metadata for a chapter within a document."""

    chapter_name: str  # File name of the chapter, e.g., "01-introduction.md"
    title: str | None = None  # Optional: Could be extracted from H1 or from manifest
    word_count: int
    paragraph_count: int
    last_modified: datetime.datetime


class DocumentInfo(BaseModel):
    """Represents metadata for a document."""

    document_name: str  # Directory name of the document
    total_chapters: int
    total_word_count: int
    total_paragraph_count: int
    last_modified: datetime.datetime
    chapters: list[ChapterMetadata]  # Ordered list of chapter metadata
    has_summary: bool = False


class DocumentSummary(BaseModel):
    """Content of a document's summary file."""

    document_name: str
    content: str
    scope: str = "document"  # "document", "chapter", "section"
    target_name: str | None = None  # chapter filename or section name when scope is not "document"


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
