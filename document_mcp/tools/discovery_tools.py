"""Discovery tools for Document MCP system.

This module provides the search_tool MCP tool for discovering
available tools on-demand, enabling defer loading for token efficiency.
"""
from __future__ import annotations


from typing import TYPE_CHECKING

from mcp.types import TextContent

if TYPE_CHECKING:
    from mcp.server import FastMCP

# Tool registry with WHAT/WHEN/RETURNS format for discovery
_TOOL_REGISTRY = [
    # Core tools (always loaded)
    {
        "name": "list_documents",
        "what": "List all available documents in the storage.",
        "when": "Use when user asks 'what documents exist?' or needs to see available content.",
        "returns": "List of document names with optional chapter details.",
        "category": "Document Management",
        "priority": "core",
    },
    {
        "name": "create_document",
        "what": "Create a new document directory for organizing chapters.",
        "when": "Use when user wants to start a new book, story, or document.",
        "returns": "Confirmation with document name and creation timestamp.",
        "category": "Document Management",
        "priority": "core",
    },
    {
        "name": "read_content",
        "what": "Read content from document, chapter, or paragraph with pagination.",
        "when": "Use when user asks to read, show, or display content.",
        "returns": "Content text with pagination metadata (page, total_pages, has_more).",
        "category": "Scope-based Content Access",
        "priority": "core",
    },
    {
        "name": "list_chapters",
        "what": "List all chapters in a specific document with optional frontmatter.",
        "when": "Use when user asks 'what chapters are in this book?' or needs structure overview.",
        "returns": "Ordered list of chapter names with metadata and optional frontmatter.",
        "category": "Chapter Management",
        "priority": "core",
    },
    # Document Management (deferred)
    {
        "name": "delete_document",
        "what": "Delete an entire document and all its chapters.",
        "when": "Use when user explicitly asks to remove/delete a document. DESTRUCTIVE.",
        "returns": "Confirmation of deletion with document name.",
        "category": "Document Management",
        "priority": "deferred",
    },
    {
        "name": "read_summary",
        "what": "Read summary files at document, chapter, or section scope.",
        "when": "Use when user asks for summary, overview, or synopsis of content.",
        "returns": "Summary text for the specified scope.",
        "category": "Document Management",
        "priority": "deferred",
    },
    {
        "name": "write_summary",
        "what": "Write or update summary files at document, chapter, or section scope.",
        "when": "Use when user wants to add or update a summary.",
        "returns": "Confirmation with summary location and character count.",
        "category": "Document Management",
        "priority": "deferred",
    },
    {
        "name": "list_summaries",
        "what": "List all available summary files for a document.",
        "when": "Use when user asks 'what summaries exist?'",
        "returns": "List of summary file names with their scopes.",
        "category": "Document Management",
        "priority": "deferred",
    },
    # Chapter Management (deferred)
    {
        "name": "create_chapter",
        "what": "Create a new chapter file with optional frontmatter metadata.",
        "when": "Use when user wants to add a chapter or start writing new content.",
        "returns": "Confirmation with chapter name and initial content length.",
        "category": "Chapter Management",
        "priority": "deferred",
    },
    {
        "name": "delete_chapter",
        "what": "Delete a chapter from a document.",
        "when": "Use when user explicitly asks to remove/delete a chapter. DESTRUCTIVE.",
        "returns": "Confirmation of deletion with chapter name.",
        "category": "Chapter Management",
        "priority": "deferred",
    },
    {
        "name": "write_chapter_content",
        "what": "Overwrite chapter content (preserves existing frontmatter).",
        "when": "Use when user wants to rewrite chapter or replace all content.",
        "returns": "Confirmation with chapter name and new content statistics.",
        "category": "Chapter Management",
        "priority": "deferred",
    },
    # Paragraph Operations (deferred)
    {
        "name": "add_paragraph",
        "what": "Add new paragraph to chapter (append to end, or insert before/after index).",
        "when": "Use when user wants to add text, insert paragraph, or continue writing.",
        "returns": "Confirmation with new paragraph index and content preview.",
        "category": "Paragraph Operations",
        "priority": "deferred",
    },
    {
        "name": "replace_paragraph",
        "what": "Replace a specific paragraph by index with new content.",
        "when": "Use when user wants to edit or fix text at a specific position.",
        "returns": "Confirmation with paragraph index and content previews.",
        "category": "Paragraph Operations",
        "priority": "deferred",
    },
    {
        "name": "delete_paragraph",
        "what": "Delete a specific paragraph by index.",
        "when": "Use when user wants to remove a paragraph. DESTRUCTIVE.",
        "returns": "Confirmation with deleted content preview.",
        "category": "Paragraph Operations",
        "priority": "deferred",
    },
    {
        "name": "move_paragraph",
        "what": "Move a paragraph to a new position (before target index or to end).",
        "when": "Use when user wants to reorder or reorganize paragraphs.",
        "returns": "Confirmation with old and new positions.",
        "category": "Paragraph Operations",
        "priority": "deferred",
    },
    # Content Tools (deferred)
    {
        "name": "find_text",
        "what": "Search for exact text matches within document or chapter scope.",
        "when": "Use when user asks 'where did I mention X?' or needs to find text.",
        "returns": "List of matches with locations and context snippets.",
        "category": "Scope-based Content Access",
        "priority": "deferred",
    },
    {
        "name": "replace_text",
        "what": "Find and replace text across document or chapter scope.",
        "when": "Use when user wants to replace all X with Y or rename something.",
        "returns": "Count of replacements made and affected locations.",
        "category": "Scope-based Content Access",
        "priority": "deferred",
    },
    {
        "name": "get_statistics",
        "what": "Get content statistics for document or chapter scope.",
        "when": "Use when user asks 'how long?' or wants word count/metrics.",
        "returns": "Statistics including word count, character count, paragraph count.",
        "category": "Scope-based Content Access",
        "priority": "deferred",
    },
    {
        "name": "find_similar_text",
        "what": "Semantic search for contextually similar content using AI.",
        "when": "Use when user asks for similar content or related passages.",
        "returns": "List of similar passages with similarity scores.",
        "category": "Scope-based Content Access",
        "priority": "deferred",
    },
    # Version Control (deferred)
    {
        "name": "manage_snapshots",
        "what": "Create, list, or restore document snapshots for version control.",
        "when": "Use when user wants to save version, list backups, or restore.",
        "returns": "Snapshot ID/list or restoration confirmation.",
        "category": "Version Control",
        "priority": "deferred",
    },
    {
        "name": "check_content_status",
        "what": "Check content freshness and modification history.",
        "when": "Use when user asks 'has this changed?' or wants edit history.",
        "returns": "Freshness status, last modified timestamp, optional history.",
        "category": "Version Control",
        "priority": "deferred",
    },
    {
        "name": "diff_content",
        "what": "Compare content between snapshots or current version.",
        "when": "Use when user asks 'what changed?' or wants to compare versions.",
        "returns": "Unified diff showing additions, deletions, and context.",
        "category": "Version Control",
        "priority": "deferred",
    },
    # Metadata Management (deferred)
    {
        "name": "read_metadata",
        "what": "Read chapter frontmatter or document-level metadata (entities, timeline).",
        "when": "Use when user asks about chapter status, character details, or timeline.",
        "returns": "Structured metadata object for the specified scope.",
        "category": "Metadata Management",
        "priority": "deferred",
    },
    {
        "name": "write_metadata",
        "what": "Write or update metadata for chapters or document-level data.",
        "when": "Use when user wants to set chapter status, add character, or update timeline.",
        "returns": "Confirmation with updated metadata summary.",
        "category": "Metadata Management",
        "priority": "deferred",
    },
    {
        "name": "list_metadata",
        "what": "List and filter metadata entries (chapters, entities, timeline).",
        "when": "Use when user asks 'what characters exist?' or 'which chapters are drafts?'",
        "returns": "Filtered list of metadata entries.",
        "category": "Metadata Management",
        "priority": "deferred",
    },
    {
        "name": "find_entity",
        "what": "Find all mentions of an entity using aliases from metadata.",
        "when": "Use when user asks 'where did I mention Marcus?' or entity tracking.",
        "returns": "Aggregated mentions by chapter with first/last mention and context.",
        "category": "Metadata Management",
        "priority": "deferred",
    },
    {
        "name": "get_document_outline",
        "what": "Get comprehensive document structure with chapter metadata.",
        "when": "Use when user asks 'what's the document structure?' or needs an overview.",
        "returns": "Hierarchical outline with chapter status, word counts, entity mentions.",
        "category": "Metadata Management",
        "priority": "deferred",
    },
    # Discovery Tools (meta-tool for searching the registry)
    {
        "name": "search_tool",
        "what": "META-TOOL: Search and discover available MCP tools by capability keyword.",
        "when": "Use when user asks about available tools, operations, or capabilities.",
        "returns": "List of matching tool names with descriptions and usage hints.",
        "category": "Discovery",
        "priority": "discovery",
    },
]


def _search_tools(query: str, category: str | None = None) -> list[dict]:
    """Search tools by query and optional category filter."""
    query_lower = query.lower()
    results = []

    for tool in _TOOL_REGISTRY:
        if category and tool["category"].lower() != category.lower():
            continue

        searchable = f"{tool['name']} {tool['what']} {tool['when']}".lower()
        if query_lower in searchable:
            results.append(tool)

    return results


def register_discovery_tools(mcp: "FastMCP") -> None:
    """Register discovery tools with the MCP server."""

    @mcp.tool()
    def search_tool(
        query: str,
        category: str | None = None,
    ) -> tuple[list[TextContent], dict]:
        """META-TOOL: Search and discover available MCP tools by capability.

        WHAT: Find available MCP tools (NOT documents!) by capability keyword.
        WHEN: Use when user asks about available tools, operations, or capabilities.
              Trigger phrases: "what tools", "which tools", "how can I", "available operations",
              "tools for editing", "tools to manage".
              Example: "What tools can I use to edit paragraphs?" -> search_tool(query="paragraph")
        RETURNS: List of matching tool names with descriptions and usage hints.
        AUTO: Searches tool registry, NOT document content. Use for tool discovery only.

        Args:
            query: Capability to search for (e.g., "paragraph", "metadata", "snapshot", "delete")
            category: Optional filter - "Document Management", "Chapter Management",
                      "Paragraph Operations", "Scope-based Content Access", "Version Control",
                      "Metadata Management"

        Returns:
            Tuple of (TextContent list, result dict) with matching tools
        """
        results = _search_tools(query, category)

        if not results:
            message = f"No tools found matching '{query}'"
            if category:
                message += f" in category '{category}'"
            message += ". Try broader terms like: document, chapter, paragraph, search, snapshot."

            return (
                [TextContent(type="text", text=message)],
                {"matches": [], "query": query, "category": category, "count": 0},
            )

        lines = [f"Found {len(results)} tool(s) matching '{query}':\n"]
        for tool in results:
            lines.append(f"**{tool['name']}** [{tool['category']}]")
            lines.append(f"  WHAT: {tool['what']}")
            lines.append(f"  WHEN: {tool['when']}")
            lines.append("")

        return (
            [TextContent(type="text", text="\n".join(lines))],
            {"matches": results, "query": query, "category": category, "count": len(results)},
        )
