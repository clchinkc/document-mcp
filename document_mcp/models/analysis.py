"""Analysis and statistics models for the Document MCP system.

This module contains models for statistics, semantic search,
and analytical operations.
"""

import datetime

from pydantic import BaseModel

__all__ = [
    "StatisticsReport",
    "SemanticSearchResult",
    "SemanticSearchResponse",
    "EmbeddingCacheEntry",
    "ChapterEmbeddingManifest",
]


class StatisticsReport(BaseModel):
    """Report for analytical queries."""

    scope: str  # e.g., "document: my_doc", "chapter: my_doc/ch1.md"
    word_count: int
    paragraph_count: int
    chapter_count: int | None = None  # Only for document-level stats


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
