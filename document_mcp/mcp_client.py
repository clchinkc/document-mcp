"""Clean client interface for Document MCP tools.

This module provides a clean Python interface to all registered MCP tools,
allowing tests and agents to interact with the document management system
without the complexity of the internal server implementation.

All functions in this module correspond directly to registered MCP tools.
"""

# Import the MCP server to access registered tools
from .doc_tool_server import mcp_server


def _get_mcp_tool(tool_name: str):
    """Get a registered MCP tool function by name."""
    if (
        hasattr(mcp_server, "_tool_manager")
        and hasattr(mcp_server._tool_manager, "_tools")
        and tool_name in mcp_server._tool_manager._tools
    ):
        tool = mcp_server._tool_manager._tools[tool_name]
        if hasattr(tool, "fn"):
            return tool.fn
    raise RuntimeError(f"MCP tool '{tool_name}' not found or not properly registered")


# Document management tools
def list_documents(include_chapters: bool = False):
    """List all available document collections."""
    return _get_mcp_tool("list_documents")(include_chapters)


def read_summary(document_name: str, scope: str = "document", target_name: str = None):
    """Read summary with flexible scope (document, chapter, section)."""
    return _get_mcp_tool("read_summary")(document_name, scope, target_name)


def write_summary(document_name: str, summary_content: str, scope: str = "document", target_name: str = None):
    """Write or update summary with flexible scope."""
    return _get_mcp_tool("write_summary")(document_name, summary_content, scope, target_name)


def list_summaries(document_name: str):
    """List all available summary files for a document."""
    return _get_mcp_tool("list_summaries")(document_name)


def create_document(document_name: str):
    """Create a new document collection."""
    return _get_mcp_tool("create_document")(document_name)


def delete_document(document_name: str):
    """Delete entire document and all chapters."""
    return _get_mcp_tool("delete_document")(document_name)


# Chapter management tools
def create_chapter(document_name: str, chapter_name: str, initial_content: str = ""):
    """Create a new chapter in a document."""
    return _get_mcp_tool("create_chapter")(document_name, chapter_name, initial_content)


def write_chapter_content(
    document_name: str,
    chapter_name: str,
    new_content: str,
    last_known_modified: str | None = None,
    force_write: bool = False,
):
    """Write/update the content of a chapter."""
    return _get_mcp_tool("write_chapter_content")(
        document_name, chapter_name, new_content, last_known_modified, force_write
    )


def delete_chapter(document_name: str, chapter_name: str):
    """Delete a chapter from a document."""
    return _get_mcp_tool("delete_chapter")(document_name, chapter_name)


def list_chapters(document_name: str):
    """List all chapters in a document."""
    return _get_mcp_tool("list_chapters")(document_name)


# Paragraph management tools
def read_paragraph_content(document_name: str, chapter_name: str, paragraph_index: int):
    """Read the content of a specific paragraph."""
    return _get_mcp_tool("read_paragraph_content")(document_name, chapter_name, paragraph_index)


def replace_paragraph(
    document_name: str,
    chapter_name: str,
    paragraph_index: int,
    new_content: str,
    last_known_modified: str | None = None,
    force_write: bool = False,
):
    """Replace content of a specific paragraph."""
    return _get_mcp_tool("replace_paragraph")(
        document_name,
        chapter_name,
        paragraph_index,
        new_content,
        last_known_modified,
        force_write,
    )


def insert_paragraph_before(document_name: str, chapter_name: str, paragraph_index: int, new_content: str):
    """Insert paragraph before specified index."""
    return _get_mcp_tool("insert_paragraph_before")(document_name, chapter_name, paragraph_index, new_content)


def insert_paragraph_after(document_name: str, chapter_name: str, paragraph_index: int, new_content: str):
    """Insert paragraph after specified index."""
    return _get_mcp_tool("insert_paragraph_after")(document_name, chapter_name, paragraph_index, new_content)


def delete_paragraph(document_name: str, chapter_name: str, paragraph_index: int):
    """Delete a specific paragraph."""
    return _get_mcp_tool("delete_paragraph")(document_name, chapter_name, paragraph_index)


def append_paragraph_to_chapter(document_name: str, chapter_name: str, new_content: str):
    """Add paragraph to end of chapter."""
    return _get_mcp_tool("append_paragraph_to_chapter")(document_name, chapter_name, new_content)


def move_paragraph_before(document_name: str, chapter_name: str, source_index: int, target_index: int):
    """Move paragraph to before another paragraph."""
    return _get_mcp_tool("move_paragraph_before")(document_name, chapter_name, source_index, target_index)


def move_paragraph_to_end(document_name: str, chapter_name: str, paragraph_index: int):
    """Move paragraph to end of chapter."""
    return _get_mcp_tool("move_paragraph_to_end")(document_name, chapter_name, paragraph_index)


# Unified content tools (scope-based)
def read_content(
    document_name: str,
    scope: str = "document",
    chapter_name: str | None = None,
    paragraph_index: int | None = None,
    page: int = 1,
    page_size: int = 50000,
):
    """Unified content reading with scope-based targeting and pagination."""
    return _get_mcp_tool("read_content")(document_name, scope, chapter_name, paragraph_index, page, page_size)


def find_text(
    document_name: str,
    search_text: str,
    scope: str = "document",
    chapter_name: str | None = None,
    case_sensitive: bool = False,
    max_results: int = 100,
):
    """Unified text search with scope-based targeting."""
    return _get_mcp_tool("find_text")(
        document_name, search_text, scope, chapter_name, case_sensitive, max_results
    )


def replace_text(
    document_name: str,
    find_text: str,
    replace_text: str,
    scope: str = "document",
    chapter_name: str | None = None,
):
    """Unified text replacement with scope-based targeting."""
    return _get_mcp_tool("replace_text")(document_name, find_text, replace_text, scope, chapter_name)


def get_statistics(document_name: str, scope: str = "document", chapter_name: str | None = None):
    """Unified statistics collection with scope-based targeting."""
    return _get_mcp_tool("get_statistics")(document_name, scope, chapter_name)


def find_similar_text(
    document_name: str,
    query_text: str,
    scope: str = "document",
    chapter_name: str | None = None,
    similarity_threshold: float = 0.7,
    max_results: int = 10,
):
    """Semantic text search with scope-based targeting and similarity scoring."""
    return _get_mcp_tool("find_similar_text")(
        document_name,
        query_text,
        scope,
        chapter_name,
        similarity_threshold,
        max_results,
    )


# Safety and version control tools
def manage_snapshots(
    document_name: str,
    action: str,
    snapshot_id: str | None = None,
    message: str | None = None,
    auto_cleanup: bool = True,
):
    """Unified snapshot management tool."""
    return _get_mcp_tool("manage_snapshots")(document_name, action, snapshot_id, message, auto_cleanup)


def check_content_status(
    document_name: str,
    chapter_name: str | None = None,
    include_history: bool = False,
    time_window: str = "24h",
    last_known_modified: str | None = None,
):
    """Unified content status and modification history checker."""
    return _get_mcp_tool("check_content_status")(
        document_name, chapter_name, include_history, time_window, last_known_modified
    )


def diff_content(
    document_name: str,
    source_type: str = "snapshot",
    source_id: str | None = None,
    target_type: str = "current",
    target_id: str | None = None,
    output_format: str = "unified",
    chapter_name: str | None = None,
):
    """Unified content comparison and diff generation tool."""
    return _get_mcp_tool("diff_content")(
        document_name,
        source_type,
        source_id,
        target_type,
        target_id,
        output_format,
        chapter_name,
    )


# All convenience functions have been removed to ensure tests call MCP tools directly.
# Use the unified tools above with appropriate scope parameters instead.


# Export all available functions
__all__ = [
    # Document management
    "list_documents",
    "create_document",
    "delete_document",
    # Chapter management
    "create_chapter",
    "write_chapter_content",
    "delete_chapter",
    "list_chapters",
    # Paragraph management
    "read_paragraph_content",
    "replace_paragraph",
    "insert_paragraph_before",
    "insert_paragraph_after",
    "delete_paragraph",
    "append_paragraph_to_chapter",
    "move_paragraph_before",
    "move_paragraph_to_end",
    # Unified content tools
    "read_content",
    "find_text",
    "replace_text",
    "get_statistics",
    "find_similar_text",
    # Safety and version control
    "manage_snapshots",
    "check_content_status",
    "diff_content",
]
