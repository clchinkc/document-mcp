"""Pydantic models for metadata validation.

This module defines schemas for:
- Chapter frontmatter (status, POV, tags, etc.)
- Document-level entities (characters, locations, items)
- Timeline events
"""
from __future__ import annotations


from enum import Enum
from typing import Any

from pydantic import BaseModel
from pydantic import Field


class ChapterStatus(str, Enum):
    """Chapter writing status."""

    DRAFT = "draft"
    REVISED = "revised"
    COMPLETE = "complete"


class ChapterMetadata(BaseModel):
    """Schema for chapter frontmatter."""

    status: ChapterStatus | None = None
    pov_character: str | None = None
    tags: list[str] = Field(default_factory=list)
    timeline_position: str | None = None
    word_target: int | None = None
    notes: str | None = None

    class Config:
        extra = "allow"  # Allow additional custom fields


class Entity(BaseModel):
    """Schema for a story entity (character, location, item)."""

    name: str
    aliases: list[str] = Field(default_factory=list)
    type: str | None = None
    description: str | None = None
    first_appearance: str | None = None
    notes: str | None = None

    class Config:
        extra = "allow"


class EntitiesFile(BaseModel):
    """Schema for entities.yaml file."""

    characters: list[Entity] = Field(default_factory=list)
    locations: list[Entity] = Field(default_factory=list)
    items: list[Entity] = Field(default_factory=list)

    class Config:
        extra = "allow"


class TimelineEvent(BaseModel):
    """Schema for a timeline event."""

    id: str
    date: str
    description: str
    chapters: list[str] = Field(default_factory=list)
    notes: str | None = None

    class Config:
        extra = "allow"


class TimelineFile(BaseModel):
    """Schema for timeline.yaml file."""

    events: list[TimelineEvent] = Field(default_factory=list)

    class Config:
        extra = "allow"


def validate_chapter_metadata(data: dict[str, Any]) -> tuple[ChapterMetadata, list[str]]:
    """Validate chapter metadata against schema.

    Args:
        data: Raw metadata dict

    Returns:
        Tuple of (validated ChapterMetadata, list of warning messages)
    """
    warnings = []

    # Validate status enum if present
    if "status" in data and data["status"] not in [s.value for s in ChapterStatus]:
        valid_values = [s.value for s in ChapterStatus]
        warnings.append(f"Invalid status '{data['status']}', expected one of: {valid_values}")

    try:
        metadata = ChapterMetadata(**data)
        return metadata, warnings
    except Exception as e:
        warnings.append(f"Validation error: {e}")
        return ChapterMetadata(), warnings


def validate_entities_file(data: dict[str, Any]) -> tuple[EntitiesFile, list[str]]:
    """Validate entities.yaml data against schema.

    Args:
        data: Raw YAML data

    Returns:
        Tuple of (validated EntitiesFile, list of warning messages)
    """
    warnings = []

    try:
        entities = EntitiesFile(**data)
        return entities, warnings
    except Exception as e:
        warnings.append(f"Validation error: {e}")
        return EntitiesFile(), warnings


def validate_timeline_file(data: dict[str, Any]) -> tuple[TimelineFile, list[str]]:
    """Validate timeline.yaml data against schema.

    Args:
        data: Raw YAML data

    Returns:
        Tuple of (validated TimelineFile, list of warning messages)
    """
    warnings = []

    try:
        timeline = TimelineFile(**data)
        return timeline, warnings
    except Exception as e:
        warnings.append(f"Validation error: {e}")
        return TimelineFile(), warnings
