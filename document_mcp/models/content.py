"""Content-related models for the Document MCP system.

This module contains models for chapter content, document content,
and content-based operations.
"""

import datetime

from pydantic import BaseModel

__all__ = [
    "ChapterContent",
    "FullDocumentContent",
    "PaginationInfo",
    "PaginatedContent",
]


class ChapterContent(BaseModel):
    """Content of a chapter file."""

    document_name: str
    chapter_name: str
    content: str
    word_count: int
    paragraph_count: int
    last_modified: datetime.datetime


class FullDocumentContent(BaseModel):
    """Content of an entire document, comprising all its chapters in order."""

    document_name: str
    chapters: list[ChapterContent]
    total_word_count: int
    total_paragraph_count: int


class PaginationInfo(BaseModel):
    """Pagination metadata for content responses."""

    page: int
    page_size: int
    total_characters: int
    total_pages: int
    has_more: bool
    has_previous: bool
    next_page: int | None = None
    previous_page: int | None = None


class PaginatedContent(BaseModel):
    """Paginated content response with metadata."""

    content: str
    document_name: str
    scope: str
    chapter_name: str | None = None
    paragraph_index: int | None = None
    pagination: PaginationInfo
