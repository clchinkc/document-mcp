"""JSON Schema definitions for all 27 registered MCP tools in the Document MCP system.

This module provides comprehensive outputSchema definitions for all MCP-registered tools across
8 categories organized by domain. Schemas are generated using Pydantic models
and unified utilities for MCP 2025-06-18 compliance.

Tool Categories (27 tools total):
- Document Tools (6): list_documents, create_document, delete_document, read_summary, write_summary, list_summaries
- Chapter Tools (4): list_chapters, create_chapter, delete_chapter, write_chapter_content
- Paragraph Tools (4): add_paragraph, replace_paragraph, delete_paragraph, move_paragraph
- Content Tools (6): read_content, find_text, replace_text, get_statistics, find_similar_text, find_entity
- Metadata Tools (3): read_metadata, write_metadata, list_metadata
- Safety Tools (3): manage_snapshots, check_content_status, diff_content
- Overview Tools (1): get_document_outline
- Discovery Tools (1): search_tool

Schema Generation Strategy:
- Single models: Use pydantic_model_to_json_schema()
- List types: Use pydantic_list_to_json_schema()
- Union types: Use pydantic_union_to_json_schema()
- Flexible objects: Use create_flexible_object_schema()
- Complex unions: Use create_oneOf_schema() or combined approaches
"""

from __future__ import annotations

from typing import Any

from mcp.types import TextContent

from ..models import ChapterContent
from ..models import ChapterMetadata
from ..models import ContentFreshnessStatus
from ..models import DocumentInfo
from ..models import DocumentSummary
from ..models import MetadataListResponse
from ..models import MetadataResponse
from ..models import ModificationHistory
from ..models import OperationStatus
from ..models import PaginatedContent
from ..models import ParagraphDetail
from ..models import SemanticSearchResponse
from ..models import SnapshotsList
from ..models import StatisticsReport
from ..utils.schema_generator import create_flexible_object_schema
from ..utils.schema_generator import create_oneOf_schema
from ..utils.schema_generator import pydantic_list_to_json_schema
from ..utils.schema_generator import pydantic_model_to_json_schema
from ..utils.schema_generator import pydantic_union_to_json_schema

__all__ = [
    # Document tools
    "LIST_DOCUMENTS_SCHEMA",
    "CREATE_DOCUMENT_SCHEMA",
    "DELETE_DOCUMENT_SCHEMA",
    "READ_SUMMARY_SCHEMA",
    "WRITE_SUMMARY_SCHEMA",
    "LIST_SUMMARIES_SCHEMA",
    # Chapter tools
    "LIST_CHAPTERS_SCHEMA",
    "CREATE_CHAPTER_SCHEMA",
    "DELETE_CHAPTER_SCHEMA",
    "WRITE_CHAPTER_CONTENT_SCHEMA",
    # Paragraph tools
    "ADD_PARAGRAPH_SCHEMA",
    "REPLACE_PARAGRAPH_SCHEMA",
    "DELETE_PARAGRAPH_SCHEMA",
    "MOVE_PARAGRAPH_SCHEMA",
    # Content tools
    "READ_CONTENT_SCHEMA",
    "FIND_TEXT_SCHEMA",
    "REPLACE_TEXT_SCHEMA",
    "GET_STATISTICS_SCHEMA",
    "FIND_SIMILAR_TEXT_SCHEMA",
    "FIND_ENTITY_SCHEMA",
    # Metadata tools
    "READ_METADATA_SCHEMA",
    "WRITE_METADATA_SCHEMA",
    "LIST_METADATA_SCHEMA",
    # Safety tools
    "MANAGE_SNAPSHOTS_SCHEMA",
    "CHECK_CONTENT_STATUS_SCHEMA",
    "DIFF_CONTENT_SCHEMA",
    # Overview tools
    "GET_DOCUMENT_OUTLINE_SCHEMA",
    # Discovery tools
    "SEARCH_TOOL_SCHEMA",
    # Registry
    "TOOL_SCHEMAS",
]

# ============================================================================
# DOCUMENT TOOLS (6 tools)
# ============================================================================

LIST_DOCUMENTS_SCHEMA = pydantic_list_to_json_schema(DocumentInfo)
"""Schema for list_documents tool.

Returns list of DocumentInfo objects with optional chapter metadata.
"""

CREATE_DOCUMENT_SCHEMA = pydantic_model_to_json_schema(OperationStatus)
"""Schema for create_document tool.

Returns OperationStatus with success/failure and optional details about created document.
"""

DELETE_DOCUMENT_SCHEMA = pydantic_model_to_json_schema(OperationStatus)
"""Schema for delete_document tool.

Returns OperationStatus indicating success or failure of deletion.
"""

READ_SUMMARY_SCHEMA = pydantic_union_to_json_schema(DocumentSummary, allow_null=True)
"""Schema for read_summary tool.

Returns DocumentSummary object or None if summary doesn't exist.
"""

WRITE_SUMMARY_SCHEMA = pydantic_model_to_json_schema(OperationStatus)
"""Schema for write_summary tool.

Returns OperationStatus confirming summary write operation.
"""

LIST_SUMMARIES_SCHEMA = {
    "type": "array",
    "items": {"type": "string"},
    "description": "List of summary filenames for a document",
}
"""Schema for list_summaries tool.

Returns array of summary file names.
"""

# ============================================================================
# CHAPTER TOOLS (5 tools, including read_chapter_content which is not registered but exists)
# ============================================================================

LIST_CHAPTERS_SCHEMA = {
    "anyOf": [
        {
            "type": "array",
            "items": create_flexible_object_schema("Chapter metadata dict"),
            "description": "List of chapter dictionaries with metadata",
        },
        {"type": "null", "description": "None if document not found"},
    ]
}
"""Schema for list_chapters tool.

Returns list of chapter dicts with metadata or None if document not found.
Returns list[dict[str, Any]] | None.
"""

CREATE_CHAPTER_SCHEMA = pydantic_model_to_json_schema(OperationStatus)
"""Schema for create_chapter tool.

Returns OperationStatus with success/failure and details about created chapter.
"""

DELETE_CHAPTER_SCHEMA = pydantic_model_to_json_schema(OperationStatus)
"""Schema for delete_chapter tool.

Returns OperationStatus indicating success or failure of deletion.
"""

WRITE_CHAPTER_CONTENT_SCHEMA = pydantic_model_to_json_schema(OperationStatus)
"""Schema for write_chapter_content tool.

Returns OperationStatus with diff information showing changes made.
"""

# ============================================================================
# PARAGRAPH TOOLS (4 tools)
# ============================================================================

ADD_PARAGRAPH_SCHEMA = pydantic_model_to_json_schema(OperationStatus)
"""Schema for add_paragraph tool.

Returns OperationStatus with success status and new paragraph index in details.
"""

REPLACE_PARAGRAPH_SCHEMA = pydantic_model_to_json_schema(OperationStatus)
"""Schema for replace_paragraph tool.

Returns OperationStatus with diff information about the replacement.
"""

DELETE_PARAGRAPH_SCHEMA = pydantic_model_to_json_schema(OperationStatus)
"""Schema for delete_paragraph tool.

Returns OperationStatus indicating success or failure of deletion.
"""

MOVE_PARAGRAPH_SCHEMA = pydantic_model_to_json_schema(OperationStatus)
"""Schema for move_paragraph tool.

Returns OperationStatus with diff information about the move operation.
Note: Return type is 'Any' but actually returns OperationStatus.
"""

# ============================================================================
# CONTENT TOOLS (6 tools)
# ============================================================================

READ_CONTENT_SCHEMA = pydantic_union_to_json_schema(
    PaginatedContent, ChapterContent, ParagraphDetail, allow_null=True
)
"""Schema for read_content tool.

Returns content object based on scope:
- document scope: PaginatedContent with pagination metadata
- chapter scope: ChapterContent with metadata
- paragraph scope: ParagraphDetail with metadata
Returns PaginatedContent | ChapterContent | ParagraphDetail | None.
Note: Return type annotated as 'Any' but uses union internally.
"""

FIND_TEXT_SCHEMA = {
    "anyOf": [
        {
            "type": "array",
            "items": create_flexible_object_schema("Search result with matches"),
        },
        {"type": "null"},
    ]
}
"""Schema for find_text tool.

Returns list of search results with context or None if not found.
Return type is 'Any' but returns list of dicts.
"""

REPLACE_TEXT_SCHEMA = pydantic_model_to_json_schema(OperationStatus)
"""Schema for replace_text tool.

Returns OperationStatus with diff information about replacements made.
Return type is 'Any' but actually returns OperationStatus.
"""

GET_STATISTICS_SCHEMA = pydantic_model_to_json_schema(StatisticsReport)
"""Schema for get_statistics tool.

Returns StatisticsReport with document statistics.
Return type is 'Any' but actually returns StatisticsReport.
"""

FIND_SIMILAR_TEXT_SCHEMA = pydantic_model_to_json_schema(SemanticSearchResponse)
"""Schema for find_similar_text tool.

Returns SemanticSearchResponse with similarity-matched content using embeddings.
Return type is 'Any' but actually returns SemanticSearchResponse.
"""

FIND_ENTITY_SCHEMA = {
    "anyOf": [
        {
            "type": "array",
            "items": create_flexible_object_schema("Entity reference"),
        },
        {"type": "null"},
    ]
}
"""Schema for find_entity tool.

Returns list of entity references or None if not found.
Return type is 'Any'.
"""

# ============================================================================
# METADATA TOOLS (3 tools)
# ============================================================================

READ_METADATA_SCHEMA = pydantic_union_to_json_schema(
    MetadataResponse, allow_null=True
)
"""Schema for read_metadata tool.

Returns MetadataResponse with chapter frontmatter, entity data, or timeline.
Returns MetadataResponse | None.
"""

WRITE_METADATA_SCHEMA = pydantic_model_to_json_schema(OperationStatus)
"""Schema for write_metadata tool.

Returns OperationStatus confirming metadata write operation.
"""

LIST_METADATA_SCHEMA = pydantic_union_to_json_schema(
    MetadataListResponse, allow_null=True
)
"""Schema for list_metadata tool.

Returns MetadataListResponse with filtered metadata entries.
Returns MetadataListResponse | None.
"""

# ============================================================================
# SAFETY TOOLS (5 tools)
# ============================================================================

MANAGE_SNAPSHOTS_SCHEMA = {
    "anyOf": [
        pydantic_model_to_json_schema(OperationStatus),
        pydantic_model_to_json_schema(SnapshotsList),
        create_flexible_object_schema("Snapshot action result"),
    ]
}
"""Schema for manage_snapshots tool.

Action-based unified snapshot management with multiple return types:
- create action: OperationStatus with snapshot_id
- list action: SnapshotsList with all snapshots
- restore action: OperationStatus with restoration details
Return type is 'Any'.
"""

CHECK_CONTENT_STATUS_SCHEMA = pydantic_model_to_json_schema(OperationStatus)
"""Schema for check_content_status tool.

Returns OperationStatus with current content status and safety information.
Return type is 'Any' but actually returns OperationStatus.
"""

DIFF_CONTENT_SCHEMA = pydantic_model_to_json_schema(OperationStatus)
"""Schema for diff_content tool.

Returns OperationStatus with unified diff information between versions.
"""

# ============================================================================
# OVERVIEW TOOLS (1 tool)
# ============================================================================

GET_DOCUMENT_OUTLINE_SCHEMA = {
    "anyOf": [
        create_flexible_object_schema("Document outline with structure and metadata"),
        {"type": "null"},
    ]
}
"""Schema for get_document_outline tool.

Returns flexible dict with document outline structure or None if not found.
Returns dict[str, Any] | None.
"""

# ============================================================================
# DISCOVERY TOOLS (1 tool)
# ============================================================================

SEARCH_TOOL_SCHEMA = {
    "type": "tuple",
    "prefixItems": [
        {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"const": "text"},
                    "text": {"type": "string"},
                },
                "required": ["type", "text"],
            },
            "description": "List of TextContent items with tool descriptions",
        },
        {
            "type": "object",
            "additionalProperties": True,
            "description": "Metadata dict with search statistics",
        },
    ],
    "description": "Tuple of (list of TextContent items, metadata dict)",
}
"""Schema for search_tool discovery tool.

Returns tuple of (list[TextContent], dict) with tool descriptions and metadata.
"""

# ============================================================================
# CENTRAL TOOL SCHEMAS REGISTRY
# ============================================================================

TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    # Document Tools (6)
    "list_documents": LIST_DOCUMENTS_SCHEMA,
    "create_document": CREATE_DOCUMENT_SCHEMA,
    "delete_document": DELETE_DOCUMENT_SCHEMA,
    "read_summary": READ_SUMMARY_SCHEMA,
    "write_summary": WRITE_SUMMARY_SCHEMA,
    "list_summaries": LIST_SUMMARIES_SCHEMA,
    # Chapter Tools (4)
    "list_chapters": LIST_CHAPTERS_SCHEMA,
    "create_chapter": CREATE_CHAPTER_SCHEMA,
    "delete_chapter": DELETE_CHAPTER_SCHEMA,
    "write_chapter_content": WRITE_CHAPTER_CONTENT_SCHEMA,
    # Paragraph Tools (4)
    "add_paragraph": ADD_PARAGRAPH_SCHEMA,
    "replace_paragraph": REPLACE_PARAGRAPH_SCHEMA,
    "delete_paragraph": DELETE_PARAGRAPH_SCHEMA,
    "move_paragraph": MOVE_PARAGRAPH_SCHEMA,
    # Content Tools (6)
    "read_content": READ_CONTENT_SCHEMA,
    "find_text": FIND_TEXT_SCHEMA,
    "replace_text": REPLACE_TEXT_SCHEMA,
    "get_statistics": GET_STATISTICS_SCHEMA,
    "find_similar_text": FIND_SIMILAR_TEXT_SCHEMA,
    "find_entity": FIND_ENTITY_SCHEMA,
    # Metadata Tools (3)
    "read_metadata": READ_METADATA_SCHEMA,
    "write_metadata": WRITE_METADATA_SCHEMA,
    "list_metadata": LIST_METADATA_SCHEMA,
    # Safety Tools (3)
    "manage_snapshots": MANAGE_SNAPSHOTS_SCHEMA,
    "check_content_status": CHECK_CONTENT_STATUS_SCHEMA,
    "diff_content": DIFF_CONTENT_SCHEMA,
    # Overview Tools (1)
    "get_document_outline": GET_DOCUMENT_OUTLINE_SCHEMA,
    # Discovery Tools (1)
    "search_tool": SEARCH_TOOL_SCHEMA,
}
"""Central registry mapping all 27 registered MCP tool names to their JSON schemas.

Coverage Summary:
- Document Tools: 6/6 (list, create, delete, read_summary, write_summary, list_summaries)
- Chapter Tools: 4/4 (list, create, delete, write_content)
- Paragraph Tools: 4/4 (add, replace, delete, move)
- Content Tools: 6/6 (read, find_text, replace, statistics, find_similar, find_entity)
- Metadata Tools: 3/3 (read, write, list)
- Safety Tools: 3/3 (manage_snapshots, check_status, diff)
- Overview Tools: 1/1 (get_document_outline)
- Discovery Tools: 1/1 (search_tool)

TOTAL: 27 TOOLS COVERED

Note: check_content_freshness and get_modification_history are helper functions in safety_tools.py
but are not registered as MCP tools (no @mcp_server.tool() decorator).
"""


def get_tool_schema(tool_name: str) -> dict[str, Any] | None:
    """Retrieve schema for a specific tool by name.

    Args:
        tool_name: Name of the MCP tool

    Returns:
        JSON schema dict or None if tool not found
    """
    return TOOL_SCHEMAS.get(tool_name)


def get_all_tool_schemas() -> dict[str, dict[str, Any]]:
    """Get all registered tool schemas.

    Returns:
        Dict mapping all 28 tool names to their schemas
    """
    return TOOL_SCHEMAS.copy()


def list_tools_by_category() -> dict[str, list[str]]:
    """Get tools organized by category.

    Returns:
        Dict mapping category names to lists of tool names (27 registered MCP tools total)
    """
    return {
        "document": [
            "list_documents",
            "create_document",
            "delete_document",
            "read_summary",
            "write_summary",
            "list_summaries",
        ],
        "chapter": [
            "list_chapters",
            "create_chapter",
            "delete_chapter",
            "write_chapter_content",
        ],
        "paragraph": [
            "add_paragraph",
            "replace_paragraph",
            "delete_paragraph",
            "move_paragraph",
        ],
        "content": [
            "read_content",
            "find_text",
            "replace_text",
            "get_statistics",
            "find_similar_text",
            "find_entity",
        ],
        "metadata": [
            "read_metadata",
            "write_metadata",
            "list_metadata",
        ],
        "safety": [
            "manage_snapshots",
            "check_content_status",
            "diff_content",
        ],
        "overview": [
            "get_document_outline",
        ],
        "discovery": [
            "search_tool",
        ],
    }
