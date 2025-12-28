"""Overview Tools for Document MCP system.

This module contains MCP tools for getting document-level overviews:
- get_document_outline: Get comprehensive document structure with metadata
"""
from __future__ import annotations


from typing import Any

import yaml
from mcp.server import FastMCP

from ..helpers import _count_words
from ..helpers import _get_document_path
from ..helpers import _get_entities_path
from ..helpers import _get_ordered_chapter_files
from ..helpers import _get_timeline_path
from ..helpers import _split_into_paragraphs
from ..logger_config import log_mcp_call
from ..utils.frontmatter import parse_frontmatter
from ..utils.validation import validate_document_name


def register_overview_tools(mcp_server: FastMCP) -> None:
    """Register all overview tools with the MCP server."""

    @mcp_server.tool()
    @log_mcp_call
    def get_document_outline(
        document_name: str,
        include_metadata: bool = True,
        include_entity_counts: bool = True,
    ) -> dict[str, Any] | None:
        """Get comprehensive document structure with optional metadata.

        WHAT: Get comprehensive document structure with chapter metadata.
        WHEN: Use when user asks "what's the document structure?" or needs an overview.
        RETURNS: Hierarchical outline with chapter status, word counts, and entity mentions.

        This tool provides a complete overview of the document including:
        - Chapter list with frontmatter metadata (status, POV, tags)
        - Word counts and paragraph counts per chapter
        - Entity mentions per chapter (if entities.yaml exists)
        - Timeline event associations (if timeline.yaml exists)

        Parameters:
            document_name (str): Name of the document to outline
            include_metadata (bool): Include chapter frontmatter metadata (default: True)
            include_entity_counts (bool): Count entity mentions per chapter (default: True)

        Returns:
            Optional[dict]: Document outline with structure:
                {
                    "document_name": "My Novel",
                    "total_chapters": 12,
                    "total_words": 45000,
                    "total_paragraphs": 432,
                    "entities": {"characters": 5, "locations": 3, "items": 2},
                    "timeline_events": 8,
                    "chapters": [
                        {
                            "name": "01-intro.md",
                            "word_count": 3500,
                            "paragraph_count": 28,
                            "metadata": {"status": "revised", "pov_character": "Marcus"},
                            "entity_mentions": {"Marcus": 12, "The City": 5}
                        },
                        ...
                    ]
                }

        Example Usage:
            ```json
            {
                "name": "get_document_outline",
                "arguments": {
                    "document_name": "My Novel",
                    "include_metadata": true,
                    "include_entity_counts": true
                }
            }
            ```
        """
        # Validate document name
        is_valid, error_msg = validate_document_name(document_name)
        if not is_valid:
            return None

        doc_path = _get_document_path(document_name)
        if not doc_path.is_dir():
            return None

        # Get chapter files
        chapter_files = _get_ordered_chapter_files(document_name)

        # Load entities for entity mention counting
        all_entities: list[dict[str, Any]] = []
        entity_summary = {"characters": 0, "locations": 0, "items": 0}
        entities_path = _get_entities_path(document_name)
        if entities_path.is_file():
            try:
                content = entities_path.read_text(encoding="utf-8")
                data = yaml.safe_load(content) or {}
                for category in ["characters", "locations", "items"]:
                    category_entities = data.get(category, [])
                    entity_summary[category] = len(category_entities)
                    for entity in category_entities:
                        all_entities.append(
                            {
                                "name": entity.get("name", ""),
                                "aliases": entity.get("aliases", []),
                                "category": category,
                            }
                        )
            except Exception:
                pass

        # Load timeline events count
        timeline_event_count = 0
        timeline_path = _get_timeline_path(document_name)
        if timeline_path.is_file():
            try:
                content = timeline_path.read_text(encoding="utf-8")
                data = yaml.safe_load(content) or {}
                timeline_event_count = len(data.get("events", []))
            except Exception:
                pass

        # Process each chapter
        chapters_info: list[dict[str, Any]] = []
        total_words = 0
        total_paragraphs = 0

        for chapter_file in chapter_files:
            try:
                chapter_content = chapter_file.read_text(encoding="utf-8")

                # Get frontmatter and body
                frontmatter_data, body_content = parse_frontmatter(chapter_content)

                # Calculate stats (on body content, not frontmatter)
                paragraphs = _split_into_paragraphs(body_content)
                word_count = _count_words(body_content)

                total_words += word_count
                total_paragraphs += len(paragraphs)

                chapter_info: dict[str, Any] = {
                    "name": chapter_file.name,
                    "word_count": word_count,
                    "paragraph_count": len(paragraphs),
                }

                if include_metadata:
                    chapter_info["metadata"] = frontmatter_data

                if include_entity_counts and all_entities:
                    # Count entity mentions in this chapter
                    entity_mentions = {}
                    content_lower = chapter_content.lower()

                    for entity in all_entities:
                        # Search for name and all aliases
                        search_terms = [entity["name"]] + entity["aliases"]
                        count = 0
                        for term in search_terms:
                            if term:
                                count += content_lower.count(term.lower())
                        if count > 0:
                            entity_mentions[entity["name"]] = count

                    if entity_mentions:
                        chapter_info["entity_mentions"] = entity_mentions

                chapters_info.append(chapter_info)

            except Exception:
                # Add minimal info for chapters that fail to read
                chapters_info.append(
                    {
                        "name": chapter_file.name,
                        "word_count": 0,
                        "paragraph_count": 0,
                        "error": "Failed to read chapter",
                    }
                )

        return {
            "document_name": document_name,
            "total_chapters": len(chapter_files),
            "total_words": total_words,
            "total_paragraphs": total_paragraphs,
            "entities": entity_summary,
            "timeline_events": timeline_event_count,
            "chapters": chapters_info,
        }
