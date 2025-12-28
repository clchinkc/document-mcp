"""Tool category modules for Document MCP system.

This package contains MCP tools organized by functional categories:
- document_tools: Document management (create, delete, list, read summary)
- chapter_tools: Chapter operations (create, delete, list, write content)
- paragraph_tools: Paragraph editing (add, replace, delete, move) - 4 tools
- content_tools: Unified content access (read, find, replace, statistics, semantic search)
- safety_tools: Version control (snapshots, status, diff)
- discovery_tools: Tool discovery for defer loading (search_tool)
- metadata_tools: Metadata management (read, write, list metadata)
- overview_tools: Document overview and outline (get_document_outline)

Paragraph Tools (4 tools):
- add_paragraph: Add new paragraph (before/after/end position)
- replace_paragraph: Replace content at specific index
- delete_paragraph: Remove paragraph at index
- move_paragraph: Reorder paragraphs (before target or to end)
"""

from .chapter_tools import register_chapter_tools
from .content_tools import register_content_tools
from .discovery_tools import register_discovery_tools
from .document_tools import register_document_tools
from .metadata_tools import register_metadata_tools
from .overview_tools import register_overview_tools
from .paragraph_tools import register_paragraph_tools
from .safety_tools import register_safety_tools

__all__ = [
    "register_document_tools",
    "register_chapter_tools",
    "register_paragraph_tools",
    "register_content_tools",
    "register_safety_tools",
    "register_discovery_tools",
    "register_metadata_tools",
    "register_overview_tools",
]
