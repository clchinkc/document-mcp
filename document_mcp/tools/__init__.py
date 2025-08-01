"""Tool category modules for Document MCP system.

This package contains the MCP tools organized by functional categories:
- document_tools: Document management (create, delete, list, read summary)
- chapter_tools: Chapter operations (create, delete, list, write content)
- paragraph_tools: Paragraph editing (replace, insert, delete, move, append)
- content_tools: Unified content access (read, find, replace, statistics)
- safety_tools: Version control (snapshots, status, diff)
"""

from .chapter_tools import register_chapter_tools
from .content_tools import register_content_tools
from .document_tools import register_document_tools
from .paragraph_tools import register_paragraph_tools
from .safety_tools import register_safety_tools

__all__ = [
    "register_document_tools",
    "register_chapter_tools",
    "register_paragraph_tools",
    "register_content_tools",
    "register_safety_tools",
]
