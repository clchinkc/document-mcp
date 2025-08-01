"""Content-related models for the Document MCP system.

This module contains models for chapter content, document content,
and content-based operations.
"""

import datetime

from pydantic import BaseModel

__all__ = [
    "ChapterContent",
    "FullDocumentContent",
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
    chapters: list[ChapterContent]  # Ordered list of chapter contents
    total_word_count: int
    total_paragraph_count: int
