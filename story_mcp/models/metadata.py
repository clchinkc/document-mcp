"""Metadata response models for the Document MCP system.

This module contains models for metadata operations:
- MetadataResponse: Unified response for read_metadata
- MetadataListResponse: Response for list_metadata
- Input models for Gemini-compatible tool parameters
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

__all__ = [
    "MetadataResponse",
    "MetadataListResponse",
    # Input models for Gemini compatibility (no additionalProperties)
    "ChapterMetadataInput",
    "EntityDataInput",
    "TimelineEventInput",
    "MetadataFilterInput",
]


class MetadataResponse(BaseModel):
    """Response model for read_metadata tool."""

    document_name: str
    scope: str  # "chapter", "entity", "timeline"
    target: str | None = None  # chapter name or entity name
    data: dict[str, Any]
    exists: bool = True


class MetadataListResponse(BaseModel):
    """Response model for list_metadata tool."""

    document_name: str
    scope: str  # "chapters", "entities", "timeline"
    items: list[dict[str, Any]]
    count: int
    filter_applied: dict[str, Any] | None = None


# --- Input Models for Gemini Compatibility ---
# These replace dict[str, Any] parameters to avoid additionalProperties issue


class ChapterMetadataInput(BaseModel):
    """Input model for chapter metadata (frontmatter).

    Uses explicit fields instead of dict to support Gemini.
    """

    status: str | None = None  # "draft", "revised", "complete"
    pov_character: str | None = None
    tags: list[str] | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict, excluding None values."""
        return {k: v for k, v in self.model_dump().items() if v is not None}


class EntityDataInput(BaseModel):
    """Input model for entity data.

    Uses explicit fields instead of dict to support Gemini.
    """

    type: str | None = None  # "character", "location", "item"
    aliases: list[str] | None = None
    description: str | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict, excluding None values."""
        return {k: v for k, v in self.model_dump().items() if v is not None}


class TimelineEventInput(BaseModel):
    """Input model for timeline events.

    Uses explicit fields instead of dict to support Gemini.
    """

    id: str  # Required
    date: str | None = None
    description: str | None = None
    chapters: list[str] | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict, excluding None values."""
        return {k: v for k, v in self.model_dump().items() if v is not None}


class MetadataFilterInput(BaseModel):
    """Input model for metadata filtering.

    Uses explicit fields instead of dict to support Gemini.
    """

    status: str | None = None  # Filter by status
    type: str | None = None  # Filter by entity type
    pov_character: str | None = None  # Filter by POV character

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict, excluding None values."""
        return {k: v for k, v in self.model_dump().items() if v is not None}
