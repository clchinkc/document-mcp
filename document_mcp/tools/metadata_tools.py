"""Metadata Management Tools.

This module contains MCP tools for managing document metadata:
- read_metadata: Read chapter frontmatter or document-level metadata
- write_metadata: Write/update metadata for chapters or documents
- list_metadata: List and filter metadata entries
"""

from typing import Any

import yaml  # type: ignore[import-untyped]
from mcp.server import FastMCP

from ..helpers import _get_chapter_path
from ..helpers import _get_document_path
from ..helpers import _get_entities_path
from ..helpers import _get_metadata_path
from ..helpers import _get_ordered_chapter_files
from ..helpers import _get_timeline_path
from ..logger_config import log_mcp_call
from ..models import MetadataListResponse
from ..models import MetadataResponse
from ..models import OperationStatus
from ..utils.decorators import auto_snapshot
from ..utils.frontmatter import has_frontmatter
from ..utils.frontmatter import parse_frontmatter
from ..utils.frontmatter import update_frontmatter
from ..utils.validation import validate_document_name


def register_metadata_tools(mcp_server: FastMCP) -> None:
    """Register all metadata management tools with the MCP server."""

    @mcp_server.tool()
    @log_mcp_call
    def read_metadata(
        document_name: str,
        scope: str,
        target: str | None = None,
    ) -> MetadataResponse | None:
        """Read metadata from chapter frontmatter or document metadata files.

        WHAT: Read metadata from chapter frontmatter or document metadata files.
        WHEN: Use when user asks about chapter status, entity details, or timeline.
        RETURNS: Structured metadata object for the specified scope.

        Parameters:
            document_name: Name of the document directory
            scope: Type of metadata - "chapter", "entity", or "timeline"
            target: Required for scope="chapter" (chapter filename) or scope="entity" (entity name)

        Returns:
            MetadataResponse with the metadata data, or None if not found

        Example Usage:
            # Read chapter frontmatter
            read_metadata("my_novel", scope="chapter", target="01-intro.md")

            # Read all entities
            read_metadata("my_novel", scope="entity")

            # Read specific entity
            read_metadata("my_novel", scope="entity", target="Marcus Chen")

            # Read timeline
            read_metadata("my_novel", scope="timeline")
        """
        is_valid, error_msg = validate_document_name(document_name)
        if not is_valid:
            return None

        doc_path = _get_document_path(document_name)
        if not doc_path.is_dir():
            return None

        if scope == "chapter":
            return _read_chapter_metadata(document_name, target)
        elif scope == "entity":
            return _read_entity_metadata(document_name, target)
        elif scope == "timeline":
            return _read_timeline_metadata(document_name)
        else:
            return None

    @mcp_server.tool()
    @log_mcp_call
    @auto_snapshot("write_metadata")
    def write_metadata(
        document_name: str,
        scope: str,
        target: str | None = None,
        # Chapter metadata fields (for scope="chapter")
        status: str | None = None,
        pov_character: str | None = None,
        tags: list[str] | None = None,
        notes: str | None = None,
        # Entity fields (for scope="entity")
        entity_type: str | None = None,
        aliases: list[str] | None = None,
        description: str | None = None,
        # Timeline fields (for scope="timeline")
        event_id: str | None = None,
        date: str | None = None,
        chapters: list[str] | None = None,
    ) -> OperationStatus:
        """Write or update metadata for chapters or document-level data.

        WHAT: Write or update metadata for chapters or document-level data.
        WHEN: Use when user wants to set chapter status, add character, or update timeline.
        RETURNS: Confirmation with updated metadata summary.

        Parameters:
            document_name: Name of the document directory
            scope: Type of metadata - "chapter", "entity", or "timeline"
            target: Required for scope="chapter" (chapter filename) or scope="entity" (entity name)
            status: Chapter status (draft/revised/complete) - for scope="chapter"
            pov_character: Point of view character - for scope="chapter"
            tags: List of tags - for scope="chapter"
            notes: Notes field - for all scopes
            entity_type: Entity type (character/location/item) - for scope="entity"
            aliases: List of aliases - for scope="entity"
            description: Description text - for scope="entity" or "timeline"
            event_id: Event identifier (required for scope="timeline")
            date: Date string - for scope="timeline"
            chapters: Related chapters - for scope="timeline"

        Returns:
            OperationStatus indicating success or failure

        Example Usage:
            # Update chapter frontmatter
            write_metadata("my_novel", scope="chapter", target="01-intro.md",
                          status="revised", pov_character="Marcus")

            # Add/update an entity
            write_metadata("my_novel", scope="entity", target="Marcus Chen",
                          aliases=["Marc", "The Detective"], entity_type="protagonist")

            # Add timeline event
            write_metadata("my_novel", scope="timeline",
                          event_id="case_reopened", date="Day 1", description="Case reopened")
        """
        is_valid, error_msg = validate_document_name(document_name)
        if not is_valid:
            return OperationStatus(success=False, message=error_msg)

        doc_path = _get_document_path(document_name)
        if not doc_path.is_dir():
            return OperationStatus(success=False, message=f"Document '{document_name}' not found")

        # Build data dict from individual parameters based on scope
        data: dict[str, Any] = {}
        if scope == "chapter":
            if status:
                data["status"] = status
            if pov_character:
                data["pov_character"] = pov_character
            if tags:
                data["tags"] = tags
            if notes:
                data["notes"] = notes
            return _write_chapter_metadata(document_name, target, data)
        elif scope == "entity":
            if entity_type:
                data["type"] = entity_type
            if aliases:
                data["aliases"] = aliases
            if description:
                data["description"] = description
            if notes:
                data["notes"] = notes
            return _write_entity_metadata(document_name, target, data)
        elif scope == "timeline":
            if event_id:
                data["id"] = event_id
            if date:
                data["date"] = date
            if description:
                data["description"] = description
            if chapters:
                data["chapters"] = chapters
            if notes:
                data["notes"] = notes
            return _write_timeline_metadata(document_name, data)
        else:
            return OperationStatus(success=False, message=f"Invalid scope: {scope}")

    @mcp_server.tool()
    @log_mcp_call
    def list_metadata(
        document_name: str,
        scope: str,
        # Filter parameters (use None to not filter on that field)
        filter_status: str | None = None,
        filter_type: str | None = None,
        filter_pov_character: str | None = None,
    ) -> MetadataListResponse | None:
        """List all metadata entries with optional filtering.

        WHAT: List all metadata entries with optional filtering.
        WHEN: Use when user asks "what characters exist?" or "which chapters are drafts?"
        RETURNS: Filtered list of metadata entries.

        Parameters:
            document_name: Name of the document directory
            scope: Type of metadata - "chapters", "entities", or "timeline"
            filter_status: Filter by status field (e.g., "draft", "revised", "complete")
            filter_type: Filter by type field (e.g., "character", "location", "item")
            filter_pov_character: Filter by POV character name

        Returns:
            MetadataListResponse with matching items

        Example Usage:
            # List all chapter metadata
            list_metadata("my_novel", scope="chapters")

            # List draft chapters only
            list_metadata("my_novel", scope="chapters", filter_status="draft")

            # List all entities
            list_metadata("my_novel", scope="entities")

            # List only characters
            list_metadata("my_novel", scope="entities", filter_type="character")
        """
        is_valid, error_msg = validate_document_name(document_name)
        if not is_valid:
            return None

        doc_path = _get_document_path(document_name)
        if not doc_path.is_dir():
            return None

        # Build filter criteria from individual parameters
        filter_criteria: dict[str, Any] | None = None
        criteria = {}
        if filter_status:
            criteria["status"] = filter_status
        if filter_type:
            criteria["type"] = filter_type
        if filter_pov_character:
            criteria["pov_character"] = filter_pov_character
        if criteria:
            filter_criteria = criteria

        if scope == "chapters":
            return _list_chapter_metadata(document_name, filter_criteria)
        elif scope == "entities":
            return _list_entity_metadata(document_name, filter_criteria)
        elif scope == "timeline":
            return _list_timeline_metadata(document_name, filter_criteria)
        else:
            return None


# --- Private Helper Functions ---


def _read_chapter_metadata(document_name: str, chapter_name: str | None) -> MetadataResponse | None:
    """Read frontmatter from a chapter file."""
    if not chapter_name:
        return None

    chapter_path = _get_chapter_path(document_name, chapter_name)
    if not chapter_path.is_file():
        return MetadataResponse(
            document_name=document_name,
            scope="chapter",
            target=chapter_name,
            data={},
            exists=False,
        )

    content = chapter_path.read_text(encoding="utf-8")
    metadata, _ = parse_frontmatter(content)

    return MetadataResponse(
        document_name=document_name,
        scope="chapter",
        target=chapter_name,
        data=metadata,
        exists=True,
    )


def _read_entity_metadata(document_name: str, entity_name: str | None) -> MetadataResponse | None:
    """Read entity metadata from entities.yaml."""
    entities_path = _get_entities_path(document_name)

    if not entities_path.is_file():
        return MetadataResponse(
            document_name=document_name,
            scope="entity",
            target=entity_name,
            data={},
            exists=False,
        )

    try:
        content = entities_path.read_text(encoding="utf-8")
        data = yaml.safe_load(content) or {}
    except yaml.YAMLError:
        return None

    if entity_name:
        # Find specific entity
        entity = _find_entity_by_name(data, entity_name)
        if entity:
            return MetadataResponse(
                document_name=document_name,
                scope="entity",
                target=entity_name,
                data=entity,
                exists=True,
            )
        return MetadataResponse(
            document_name=document_name,
            scope="entity",
            target=entity_name,
            data={},
            exists=False,
        )

    return MetadataResponse(
        document_name=document_name,
        scope="entity",
        target=None,
        data=data,
        exists=True,
    )


def _read_timeline_metadata(document_name: str) -> MetadataResponse | None:
    """Read timeline metadata from timeline.yaml."""
    timeline_path = _get_timeline_path(document_name)

    if not timeline_path.is_file():
        return MetadataResponse(
            document_name=document_name,
            scope="timeline",
            target=None,
            data={},
            exists=False,
        )

    try:
        content = timeline_path.read_text(encoding="utf-8")
        data = yaml.safe_load(content) or {}
    except yaml.YAMLError:
        return None

    return MetadataResponse(
        document_name=document_name,
        scope="timeline",
        target=None,
        data=data,
        exists=True,
    )


def _write_chapter_metadata(
    document_name: str, chapter_name: str | None, data: dict[str, Any]
) -> OperationStatus:
    """Write/update frontmatter in a chapter file."""
    if not chapter_name:
        return OperationStatus(success=False, message="chapter_name (target) is required for chapter scope")

    chapter_path = _get_chapter_path(document_name, chapter_name)
    if not chapter_path.is_file():
        return OperationStatus(success=False, message=f"Chapter '{chapter_name}' not found")

    try:
        content = chapter_path.read_text(encoding="utf-8")
        updated_content = update_frontmatter(content, data)
        chapter_path.write_text(updated_content, encoding="utf-8")

        return OperationStatus(
            success=True,
            message=f"Updated metadata for chapter '{chapter_name}'",
            details={"chapter_name": chapter_name, "fields_updated": list(data.keys())},
        )
    except Exception as e:
        return OperationStatus(success=False, message=f"Error updating chapter metadata: {e}")


def _write_entity_metadata(
    document_name: str, entity_name: str | None, data: dict[str, Any]
) -> OperationStatus:
    """Write/update entity in entities.yaml."""
    if not entity_name:
        return OperationStatus(success=False, message="entity_name (target) is required for entity scope")

    metadata_path = _get_metadata_path(document_name)
    entities_path = _get_entities_path(document_name)

    # Ensure metadata directory exists
    metadata_path.mkdir(exist_ok=True)

    # Load existing entities
    existing_data: dict[str, Any] = {"characters": [], "locations": [], "items": []}
    if entities_path.is_file():
        try:
            content = entities_path.read_text(encoding="utf-8")
            existing_data = yaml.safe_load(content) or existing_data
        except yaml.YAMLError:
            # Use empty data structure if YAML parsing fails
            pass

    # Determine entity type and update
    entity_type = data.get("type", "characters")
    if entity_type in ("character", "protagonist", "antagonist", "supporting"):
        category = "characters"
    elif entity_type in ("location", "place"):
        category = "locations"
    elif entity_type in ("item", "object"):
        category = "items"
    else:
        category = "characters"  # Default

    # Ensure category exists
    if category not in existing_data:
        existing_data[category] = []

    # Find and update or add entity
    entity_list = existing_data[category]
    entity_found = False
    for i, entity in enumerate(entity_list):
        if entity.get("name") == entity_name:
            entity_list[i] = {**entity, **data, "name": entity_name}
            entity_found = True
            break

    if not entity_found:
        entity_list.append({"name": entity_name, **data})

    try:
        yaml_content = yaml.dump(existing_data, default_flow_style=False, allow_unicode=True, sort_keys=False)
        entities_path.write_text(yaml_content, encoding="utf-8")

        action = "Updated" if entity_found else "Added"
        return OperationStatus(
            success=True,
            message=f"{action} entity '{entity_name}' in {category}",
            details={"entity_name": entity_name, "category": category, "action": action.lower()},
        )
    except Exception as e:
        return OperationStatus(success=False, message=f"Error writing entity metadata: {e}")


def _write_timeline_metadata(document_name: str, data: dict[str, Any]) -> OperationStatus:
    """Write/update event in timeline.yaml."""
    if "id" not in data:
        return OperationStatus(success=False, message="Timeline event must have an 'id' field")

    metadata_path = _get_metadata_path(document_name)
    timeline_path = _get_timeline_path(document_name)

    # Ensure metadata directory exists
    metadata_path.mkdir(exist_ok=True)

    # Load existing timeline
    existing_data: dict[str, Any] = {"events": []}
    if timeline_path.is_file():
        try:
            content = timeline_path.read_text(encoding="utf-8")
            existing_data = yaml.safe_load(content) or existing_data
        except yaml.YAMLError:
            # Use empty events list if YAML parsing fails
            pass

    if "events" not in existing_data:
        existing_data["events"] = []

    # Find and update or add event
    events_list = existing_data["events"]
    event_id = data["id"]
    event_found = False
    for i, event in enumerate(events_list):
        if event.get("id") == event_id:
            events_list[i] = {**event, **data}
            event_found = True
            break

    if not event_found:
        events_list.append(data)

    try:
        yaml_content = yaml.dump(existing_data, default_flow_style=False, allow_unicode=True, sort_keys=False)
        timeline_path.write_text(yaml_content, encoding="utf-8")

        action = "Updated" if event_found else "Added"
        return OperationStatus(
            success=True,
            message=f"{action} timeline event '{event_id}'",
            details={"event_id": event_id, "action": action.lower()},
        )
    except Exception as e:
        return OperationStatus(success=False, message=f"Error writing timeline metadata: {e}")


def _list_chapter_metadata(
    document_name: str, filter_criteria: dict[str, Any] | None
) -> MetadataListResponse:
    """List metadata for all chapters with optional filtering."""
    chapter_files = _get_ordered_chapter_files(document_name)
    items: list[dict[str, Any]] = []

    for chapter_path in chapter_files:
        chapter_name = chapter_path.name
        content = chapter_path.read_text(encoding="utf-8")
        metadata, _ = parse_frontmatter(content)

        chapter_info = {"chapter_name": chapter_name, "has_frontmatter": has_frontmatter(content), **metadata}

        # Apply filter if provided
        if filter_criteria:
            matches = all(chapter_info.get(k) == v for k, v in filter_criteria.items())
            if matches:
                items.append(chapter_info)
        else:
            items.append(chapter_info)

    return MetadataListResponse(
        document_name=document_name,
        scope="chapters",
        items=items,
        count=len(items),
        filter_applied=filter_criteria,
    )


def _list_entity_metadata(document_name: str, filter_criteria: dict[str, Any] | None) -> MetadataListResponse:
    """List all entities with optional filtering."""
    entities_path = _get_entities_path(document_name)
    items: list[dict[str, Any]] = []

    if entities_path.is_file():
        try:
            content = entities_path.read_text(encoding="utf-8")
            data = yaml.safe_load(content) or {}

            # Collect entities from all categories
            for category in ["characters", "locations", "items"]:
                for entity in data.get(category, []):
                    entity_info = {"category": category, **entity}

                    # Apply filter if provided
                    if filter_criteria:
                        # Handle 'type' filter specially to match category
                        if "type" in filter_criteria:
                            filter_type = filter_criteria["type"]
                            type_matches = (
                                filter_type == category
                                or filter_type == category.rstrip("s")  # "character" matches "characters"
                                or entity.get("type") == filter_type
                            )
                            if not type_matches:
                                continue

                        # Check other filter criteria
                        other_criteria = {k: v for k, v in filter_criteria.items() if k != "type"}
                        matches = all(entity_info.get(k) == v for k, v in other_criteria.items())
                        if matches:
                            items.append(entity_info)
                    else:
                        items.append(entity_info)
        except yaml.YAMLError:
            # Continue with empty list if YAML parsing fails
            pass

    return MetadataListResponse(
        document_name=document_name,
        scope="entities",
        items=items,
        count=len(items),
        filter_applied=filter_criteria,
    )


def _list_timeline_metadata(
    document_name: str, filter_criteria: dict[str, Any] | None
) -> MetadataListResponse:
    """List all timeline events with optional filtering."""
    timeline_path = _get_timeline_path(document_name)
    items: list[dict[str, Any]] = []

    if timeline_path.is_file():
        try:
            content = timeline_path.read_text(encoding="utf-8")
            data = yaml.safe_load(content) or {}

            for event in data.get("events", []):
                # Apply filter if provided
                if filter_criteria:
                    matches = all(event.get(k) == v for k, v in filter_criteria.items())
                    if matches:
                        items.append(event)
                else:
                    items.append(event)
        except yaml.YAMLError:
            # Continue with empty list if YAML parsing fails
            pass

    return MetadataListResponse(
        document_name=document_name,
        scope="timeline",
        items=items,
        count=len(items),
        filter_applied=filter_criteria,
    )


def _find_entity_by_name(data: dict[str, Any], entity_name: str) -> dict[str, Any] | None:
    """Find an entity by name or alias across all categories."""
    name_lower = entity_name.lower()

    for category in ["characters", "locations", "items"]:
        for entity in data.get(category, []):
            # Check name
            if entity.get("name", "").lower() == name_lower:
                return {**entity, "category": category}

            # Check aliases
            aliases = entity.get("aliases", [])
            for alias in aliases:
                if alias.lower() == name_lower:
                    return {**entity, "category": category}

    return None
