"""Pydantic models for the Document MCP system.

This module contains all data models used throughout the Document MCP system,
including request/response models, metadata structures, and batch operation models.
"""

import datetime
from typing import Any

from pydantic import BaseModel

# === Core Operation Models ===


class OperationStatus(BaseModel):
    """Generic status for operations with optional safety information."""

    success: bool
    message: str
    details: dict[str, Any] | None = None  # For extra info, e.g., created entity name
    # Safety fields (optional for backward compatibility)
    safety_info: Any | None = None
    snapshot_created: str | None = None
    warnings: list[str] = []


# === Document Metadata Models ===


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
    last_modified: datetime.datetime  # Could be latest of any chapter or document folder itself
    chapters: list[ChapterMetadata]  # Ordered list of chapter metadata
    has_summary: bool = False


class ParagraphDetail(BaseModel):
    """Detailed information about a paragraph."""

    document_name: str
    chapter_name: str
    paragraph_index_in_chapter: int  # 0-indexed within its chapter
    content: str
    word_count: int


# === Content Models ===


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


# === Analysis and Statistics Models ===


class StatisticsReport(BaseModel):
    """Report for analytical queries."""

    scope: str  # e.g., "document: my_doc", "chapter: my_doc/ch1.md"
    word_count: int
    paragraph_count: int
    chapter_count: int | None = None  # Only for document-level stats


# === Safety and History Models ===


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


# === Batch Operation Models ===


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


# === Snapshot Models ===


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


# === Semantic Search Models ===


class SemanticSearchResult(BaseModel):
    """Semantic search result with similarity scoring."""

    document_name: str
    chapter_name: str
    paragraph_index: int  # Zero-indexed within chapter
    content: str
    similarity_score: float
    context_snippet: str | None = None  # Surrounding text for context


class SemanticSearchResponse(BaseModel):
    """Response wrapper for semantic search operations."""

    document_name: str
    scope: str  # "document" or "chapter"
    query_text: str
    results: list[SemanticSearchResult]
    total_results: int
    execution_time_ms: float


# === Embedding Cache Models ===


class EmbeddingCacheEntry(BaseModel):
    """Single paragraph embedding cache entry."""

    content_hash: str  # MD5 of paragraph content
    paragraph_index: int  # Index within chapter
    model_version: str  # "models/text-embedding-004"
    created_at: datetime.datetime  # When embedding was generated
    file_modified_time: datetime.datetime  # Source file modification time


class ChapterEmbeddingManifest(BaseModel):
    """Manifest for chapter embedding cache."""

    chapter_name: str
    total_paragraphs: int
    cache_entries: list[EmbeddingCacheEntry]
    last_updated: datetime.datetime
