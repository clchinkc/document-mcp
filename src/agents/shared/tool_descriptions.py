"""MCP Tool Registry - Single source of truth for all tool metadata.

This module provides:
- Tool descriptions for agents (WHAT/WHEN/RETURNS/AUTO format)
- Benchmark scenarios for testing tool selection
- DSPy training data for prompt optimization

Usage:
    from src.agents.shared.tool_descriptions import (
        ToolDescriptionManager,
        get_all_scenarios,
        get_dspy_tool_descriptions,
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from enum import Enum


class ToolFormat(Enum):
    """Format styles for tool descriptions based on agent architecture."""

    FULL = "full"
    COMPACT = "compact"
    PLANNER = "planner"
    MINIMAL = "minimal"


class LoadingPriority(Enum):
    """Tool loading priority for defer loading implementation."""

    CORE = "core"
    DISCOVERY = "discovery"
    DEFERRED = "deferred"


@dataclass
class ToolScenario:
    """A benchmark scenario for testing tool selection."""

    query: str
    description: str = ""


@dataclass
class ToolDescription:
    """Tool configuration with description and benchmark scenarios."""

    name: str
    what: str
    when: str
    returns: str
    auto: str = ""
    parameters: dict[str, str] = field(default_factory=dict)
    example: str = ""
    category: str = ""
    planner_signature: str = ""
    loading_priority: LoadingPriority = LoadingPriority.DEFERRED
    scenarios: list[ToolScenario] = field(default_factory=list)

    @property
    def description(self) -> str:
        return f"{self.what} {self.when}"


class ToolDescriptionManager:
    """Single source of truth for all MCP tool metadata."""

    def __init__(self):
        self._tools = self._initialize_tools()

    def _initialize_tools(self) -> list[ToolDescription]:
        """Initialize all 28 MCP tools with descriptions and scenarios."""
        return [
            # === DOCUMENT MANAGEMENT (3 tools) ===
            ToolDescription(
                name="list_documents",
                what="List all available documents in the storage.",
                when="Use when user asks 'what documents exist?', 'show my books', 'list projects'.",
                returns="List of document names with optional chapter details.",
                auto="Use include_chapters=True for metadata, False for fast overview.",
                parameters={"include_chapters": "bool"},
                example="list_documents(include_chapters=True)",
                category="Document Management",
                planner_signature="list_documents(include_chapters: bool = False)",
                loading_priority=LoadingPriority.CORE,
                scenarios=[
                    ToolScenario("What documents do I have?", "List documents"),
                    ToolScenario("Show me all my books", "List documents"),
                ],
            ),
            ToolDescription(
                name="create_document",
                what="Create a new document directory for organizing chapters.",
                when="Use when user wants to start a new book, story, or document.",
                returns="Confirmation with document name and creation timestamp.",
                auto="Creates directory structure with summaries/ and metadata/ folders.",
                parameters={"document_name": "str"},
                example='create_document(document_name="My Book")',
                category="Document Management",
                planner_signature="create_document(document_name: str)",
                loading_priority=LoadingPriority.CORE,
                scenarios=[
                    ToolScenario("Create a new document called 'my_story'", "Create document"),
                    ToolScenario("Start a new book named 'Novel'", "Create document"),
                ],
            ),
            ToolDescription(
                name="delete_document",
                what="Delete an entire document and all its chapters.",
                when="Use when user explicitly asks to remove/delete a document. DESTRUCTIVE.",
                returns="Confirmation of deletion with document name.",
                auto="Removes all chapters, summaries, metadata, and snapshots.",
                parameters={"document_name": "str"},
                example='delete_document(document_name="My Book")',
                category="Document Management",
                planner_signature="delete_document(document_name: str)",
                loading_priority=LoadingPriority.DEFERRED,
                scenarios=[
                    ToolScenario("Delete the document called 'old_draft'", "Delete document"),
                ],
            ),
            # === SUMMARY TOOLS (3 tools) ===
            ToolDescription(
                name="read_summary",
                what="Read summary files at document, chapter, or section scope.",
                when="Use when user asks for 'summary', 'overview', 'synopsis', 'what is this about?'.",
                returns="Summary text for the specified scope.",
                auto="Supports scope='document', 'chapter', or 'section'.",
                parameters={"document_name": "str", "scope": "str", "target": "str"},
                example='read_summary(document_name="My Book", scope="document")',
                category="Document Management",
                planner_signature="read_summary(document_name: str, scope: str = 'document', target: str = None)",
                loading_priority=LoadingPriority.DEFERRED,
                scenarios=[
                    ToolScenario("Show me the summary of my novel", "Read summary"),
                ],
            ),
            ToolDescription(
                name="write_summary",
                what="Write or update summary files at document, chapter, or section scope.",
                when="Use when user wants to add or update a summary, synopsis, or overview.",
                returns="Confirmation with summary location and character count.",
                auto="Creates summaries/ directory if needed.",
                parameters={
                    "document_name": "str",
                    "summary_content": "str",
                    "scope": "str",
                    "target": "str",
                },
                example='write_summary(document_name="My Book", summary_content="This book covers...", scope="document")',
                category="Document Management",
                planner_signature="write_summary(document_name: str, summary_content: str, scope: str = 'document', target: str = None)",
                loading_priority=LoadingPriority.DEFERRED,
                scenarios=[
                    ToolScenario("Write a summary for my novel: 'A tale of two cities...'", "Write summary"),
                ],
            ),
            ToolDescription(
                name="list_summaries",
                what="List all available summary files for a document.",
                when="Use when user asks 'what summaries exist?', 'show available summaries'.",
                returns="List of summary file names with their scopes.",
                auto="Lists document, chapter, and section summaries.",
                parameters={"document_name": "str"},
                example='list_summaries(document_name="My Book")',
                category="Document Management",
                planner_signature="list_summaries(document_name: str)",
                loading_priority=LoadingPriority.DEFERRED,
                scenarios=[
                    ToolScenario("What summaries are available for my novel?", "List summaries"),
                ],
            ),
            # === CHAPTER MANAGEMENT (4 tools) ===
            ToolDescription(
                name="list_chapters",
                what="List all chapters in a specific document with optional frontmatter.",
                when="Use when user asks 'what chapters are in this book?', 'show chapter list', 'table of contents'.",
                returns="Ordered list of chapter names with metadata (word count, paragraph count).",
                auto="Returns chapters in filesystem order (01-, 02-, etc.).",
                parameters={"document_name": "str", "include_frontmatter": "bool"},
                example='list_chapters(document_name="My Book")',
                category="Chapter Management",
                planner_signature="list_chapters(document_name: str, include_frontmatter: bool = False)",
                loading_priority=LoadingPriority.CORE,
                scenarios=[
                    ToolScenario("What chapters are in my novel?", "List chapters"),
                ],
            ),
            ToolDescription(
                name="create_chapter",
                what="Create a new chapter file with optional frontmatter metadata.",
                when="Use when user wants to add a chapter, start a new section, or create new content file.",
                returns="Confirmation with chapter name and initial content length.",
                auto="Adds YAML frontmatter if metadata provided, creates numbered filename.",
                parameters={"document_name": "str", "chapter_name": "str", "initial_content": "str"},
                example='create_chapter(document_name="My Book", chapter_name="01-introduction.md", initial_content="# Introduction")',
                category="Chapter Management",
                planner_signature="create_chapter(document_name: str, chapter_name: str, initial_content: str = '')",
                loading_priority=LoadingPriority.DEFERRED,
                scenarios=[
                    ToolScenario(
                        "Create a new chapter called '04-conclusion.md' in document novel", "Create chapter"
                    ),
                ],
            ),
            ToolDescription(
                name="delete_chapter",
                what="Delete a chapter from a document.",
                when="Use when user explicitly asks to remove/delete a chapter. DESTRUCTIVE.",
                returns="Confirmation of deletion with chapter name.",
                auto="Creates automatic snapshot before deletion.",
                parameters={"document_name": "str", "chapter_name": "str"},
                example='delete_chapter(document_name="My Book", chapter_name="01-introduction.md")',
                category="Chapter Management",
                planner_signature="delete_chapter(document_name: str, chapter_name: str)",
                loading_priority=LoadingPriority.DEFERRED,
                scenarios=[
                    ToolScenario("Delete chapter 02-old.md from novel", "Delete chapter"),
                ],
            ),
            ToolDescription(
                name="write_chapter_content",
                what="Overwrite entire chapter content (preserves existing frontmatter).",
                when="Use when user wants to completely rewrite a chapter or replace all content.",
                returns="Confirmation with chapter name and new content statistics.",
                auto="Creates automatic snapshot, preserves YAML frontmatter if present.",
                parameters={"document_name": "str", "chapter_name": "str", "new_content": "str"},
                example='write_chapter_content(document_name="My Book", chapter_name="01-intro.md", new_content="# New Content")',
                category="Chapter Management",
                planner_signature="write_chapter_content(document_name: str, chapter_name: str, new_content: str)",
                loading_priority=LoadingPriority.DEFERRED,
                scenarios=[
                    ToolScenario(
                        "Rewrite the entire chapter 01-intro.md with this new content", "Overwrite chapter"
                    ),
                ],
            ),
            # === PARAGRAPH OPERATIONS (4 tools) ===
            ToolDescription(
                name="add_paragraph",
                what="Add new paragraph to chapter - use position='end' for append, 'before'/'after' for insert.",
                when="Use when adding new content: 'add paragraph', 'insert text', 'append content'. Set position='end' for end, 'before'/'after' with paragraph_index.",
                returns="Confirmation with new paragraph index and content preview.",
                auto="Creates automatic snapshot. position='end' ignores paragraph_index.",
                parameters={
                    "document_name": "str",
                    "chapter_name": "str",
                    "content": "str",
                    "position": "str",
                    "paragraph_index": "int",
                },
                example='add_paragraph(document_name="My Book", chapter_name="01-intro.md", content="New text", position="end")',
                category="Paragraph Operations",
                planner_signature="add_paragraph(document_name: str, chapter_name: str, content: str, position: str = 'end', paragraph_index: int = None)",
                loading_priority=LoadingPriority.DEFERRED,
                scenarios=[
                    # Basic operations
                    ToolScenario(
                        "In document 'novel', chapter 01-intro.md, insert a new paragraph after paragraph 2",
                        "Insert after position",
                    ),
                    ToolScenario(
                        "Insert a paragraph before the first paragraph in chapter 02-middle.md",
                        "Insert before position",
                    ),
                    ToolScenario(
                        "Add a new paragraph at the end of chapter 02-middle.md in novel",
                        "Append to chapter end",
                    ),
                    # Edge cases
                    ToolScenario(
                        "Document 'empty_doc' has chapter '01-empty.md' with no paragraphs. Add the first paragraph 'Hello World'",
                        "Empty chapter append",
                    ),
                    ToolScenario(
                        "Document 'minimal' has 1 paragraph. Insert before paragraph 0",
                        "Insert at boundary index 0",
                    ),
                    # Adversarial - confusing phrasing
                    ToolScenario(
                        "Put new content right at the very beginning of the chapter, before everything else",
                        "Adversarial: beginning means before 0",
                    ),
                    ToolScenario(
                        "Stick a paragraph in between paragraphs 2 and 3 in chapter intro.md",
                        "Adversarial: between means after 2",
                    ),
                ],
            ),
            ToolDescription(
                name="replace_paragraph",
                what="Replace existing paragraph content at exact index - paragraph count unchanged.",
                when="Use when: 'replace paragraph N', 'edit paragraph N', 'update text at position N'. NOT for adding new paragraphs.",
                returns="Confirmation with paragraph index, old and new content previews.",
                auto="Creates automatic snapshot, validates paragraph index exists.",
                parameters={
                    "document_name": "str",
                    "chapter_name": "str",
                    "paragraph_index": "int",
                    "new_content": "str",
                },
                example='replace_paragraph(document_name="My Book", chapter_name="01-intro.md", paragraph_index=0, new_content="Updated text.")',
                category="Paragraph Operations",
                planner_signature="replace_paragraph(document_name: str, chapter_name: str, paragraph_index: int, new_content: str)",
                loading_priority=LoadingPriority.DEFERRED,
                scenarios=[
                    # Basic operations
                    ToolScenario(
                        "Replace the third paragraph in chapter 01-intro.md with new content",
                        "Replace at index",
                    ),
                    ToolScenario(
                        "Update paragraph 0 in chapter 02-middle.md of novel with revised text",
                        "Replace first paragraph",
                    ),
                    # Edge cases
                    ToolScenario(
                        "Document has 5 paragraphs (0-4). Replace paragraph 4 with new ending",
                        "Replace last paragraph",
                    ),
                    ToolScenario(
                        "Document has unicode content. Replace paragraph 1 containing '日本語' with English",
                        "Unicode content",
                    ),
                    # Adversarial - confusing phrasing
                    ToolScenario(
                        "Edit the content at position 2 to say something different",
                        "Adversarial: edit means replace",
                    ),
                    ToolScenario(
                        "Change what paragraph 3 says without removing it",
                        "Adversarial: change without remove = replace",
                    ),
                ],
            ),
            ToolDescription(
                name="delete_paragraph",
                what="Delete paragraph at index - subsequent paragraphs shift up.",
                when="Use when: 'delete paragraph N', 'remove paragraph N'. DESTRUCTIVE - paragraph is gone.",
                returns="Confirmation with deleted content preview and updated paragraph count.",
                auto="Creates automatic snapshot, shifts subsequent paragraphs.",
                parameters={"document_name": "str", "chapter_name": "str", "paragraph_index": "int"},
                example='delete_paragraph(document_name="My Book", chapter_name="01-intro.md", paragraph_index=2)',
                category="Paragraph Operations",
                planner_signature="delete_paragraph(document_name: str, chapter_name: str, paragraph_index: int)",
                loading_priority=LoadingPriority.DEFERRED,
                scenarios=[
                    # Basic operations
                    ToolScenario(
                        "Delete paragraph 4 from document novel, chapter 03-end.md", "Delete at index"
                    ),
                    ToolScenario(
                        "Remove the first paragraph from chapter 01-intro.md", "Delete first paragraph"
                    ),
                    # Edge cases
                    ToolScenario(
                        "Document has single paragraph (index 0). Delete it to make chapter empty",
                        "Delete only paragraph",
                    ),
                    ToolScenario(
                        "Delete the last paragraph (index 6) from a 7-paragraph chapter",
                        "Delete last paragraph",
                    ),
                    # Adversarial - confusing phrasing
                    ToolScenario(
                        "Get rid of paragraph 2 entirely from the chapter", "Adversarial: get rid = delete"
                    ),
                    ToolScenario(
                        "Eliminate the duplicate paragraph at position 3", "Adversarial: eliminate = delete"
                    ),
                ],
            ),
            ToolDescription(
                name="move_paragraph",
                what="Move paragraph to new position - use target_type='before' or 'end'.",
                when="Use when reordering: 'move paragraph X to before Y' (target_type='before'), 'move to end' (target_type='end').",
                returns="Confirmation with old and new positions.",
                auto="Creates automatic snapshot, handles index recalculation.",
                parameters={
                    "document_name": "str",
                    "chapter_name": "str",
                    "paragraph_index": "int",
                    "target_type": "str",
                    "target_paragraph_index": "int",
                },
                example='move_paragraph(document_name="My Book", chapter_name="01-intro.md", paragraph_index=3, target_type="before", target_paragraph_index=1)',
                category="Paragraph Operations",
                planner_signature="move_paragraph(document_name: str, chapter_name: str, paragraph_index: int, target_type: str = 'before', target_paragraph_index: int = None)",
                loading_priority=LoadingPriority.DEFERRED,
                scenarios=[
                    # Basic operations
                    ToolScenario(
                        "Move paragraph 5 to before paragraph 2 in novel/01-intro.md", "Move to position"
                    ),
                    ToolScenario(
                        "Move paragraph 1 to the end of the chapter in document my_book, chapter 02.md",
                        "Move to end",
                    ),
                    # Edge cases
                    ToolScenario(
                        "Move the first paragraph (0) to the end of the chapter", "Move first to end"
                    ),
                    ToolScenario("Move the last paragraph to before paragraph 0", "Move last to first"),
                    # Adversarial - confusing phrasing
                    ToolScenario(
                        "Reorder paragraph 4 so it appears before paragraph 1",
                        "Adversarial: reorder = move before",
                    ),
                    ToolScenario(
                        "Shift paragraph 3 down to the bottom of the chapter",
                        "Adversarial: shift down/bottom = move to end",
                    ),
                ],
            ),
            # === CONTENT ACCESS (6 tools) ===
            ToolDescription(
                name="read_content",
                what="Read content from document, chapter, or paragraph with pagination.",
                when="Use when user asks to 'read', 'show', 'display', 'get content', 'what does it say'.",
                returns="Content text with pagination metadata (page, total_pages, has_more).",
                auto="50K chars per page. scope='document'|'chapter'|'paragraph'.",
                parameters={
                    "document_name": "str",
                    "scope": "str",
                    "chapter_name": "str",
                    "paragraph_index": "int",
                    "page": "int",
                },
                example='read_content(document_name="My Book", scope="chapter", chapter_name="01-intro.md")',
                category="Content Access",
                planner_signature="read_content(document_name: str, scope: str = 'document', chapter_name: str = None, paragraph_index: int = None, page: int = 1)",
                loading_priority=LoadingPriority.CORE,
                scenarios=[
                    ToolScenario("Read the entire document called 'novel'", "Read full document"),
                    ToolScenario("Show me chapter 02-middle.md from document novel", "Read specific chapter"),
                ],
            ),
            ToolDescription(
                name="find_text",
                what="Search for exact text matches within document or chapter scope.",
                when="Use when user asks 'where did I mention X?', 'find all occurrences of', 'search for text'.",
                returns="List of matches with locations (chapter, paragraph index) and context snippets.",
                auto="Handles case sensitivity option, limits results with max_results.",
                parameters={
                    "document_name": "str",
                    "search_text": "str",
                    "scope": "str",
                    "chapter_name": "str",
                    "case_sensitive": "bool",
                    "max_results": "int",
                },
                example='find_text(document_name="My Book", search_text="protagonist", scope="document")',
                category="Content Access",
                planner_signature="find_text(document_name: str, search_text: str, scope: str = 'document', chapter_name: str = None, case_sensitive: bool = False, max_results: int = 50)",
                loading_priority=LoadingPriority.DEFERRED,
                scenarios=[
                    ToolScenario("Search for the exact phrase 'once upon a time' in my novel", "Text search"),
                ],
            ),
            ToolDescription(
                name="replace_text",
                what="Find and replace text across document or chapter scope.",
                when="Use when user wants to 'replace all X with Y', 'rename character', 'fix typo everywhere'.",
                returns="Count of replacements made and affected locations.",
                auto="Creates automatic snapshot, handles case sensitivity.",
                parameters={
                    "document_name": "str",
                    "find_text": "str",
                    "replace_with": "str",
                    "scope": "str",
                    "chapter_name": "str",
                    "case_sensitive": "bool",
                },
                example='replace_text(document_name="My Book", find_text="John", replace_with="James", scope="document")',
                category="Content Access",
                planner_signature="replace_text(document_name: str, find_text: str, replace_with: str, scope: str = 'document', chapter_name: str = None, case_sensitive: bool = False)",
                loading_priority=LoadingPriority.DEFERRED,
                scenarios=[
                    ToolScenario("Replace 'John' with 'James' throughout the document", "Find and replace"),
                ],
            ),
            ToolDescription(
                name="get_statistics",
                what="Get content statistics for document or chapter scope.",
                when="Use when user asks 'how long?', 'word count', 'how many paragraphs?', 'statistics'.",
                returns="Statistics including word count, character count, paragraph count.",
                auto="Calculates stats for entire document or specific chapter.",
                parameters={"document_name": "str", "scope": "str", "chapter_name": "str"},
                example='get_statistics(document_name="My Book", scope="document")',
                category="Content Access",
                planner_signature="get_statistics(document_name: str, scope: str = 'document', chapter_name: str = None)",
                loading_priority=LoadingPriority.DEFERRED,
                scenarios=[
                    ToolScenario("How many words are in my novel?", "Word count request"),
                ],
            ),
            ToolDescription(
                name="find_similar_text",
                what="Semantic search for contextually similar content using AI embeddings.",
                when="Use when user asks for 'similar content', 'related passages', 'find themes like', 'semantic search'.",
                returns="List of similar passages with similarity scores and context.",
                auto="Uses Gemini embeddings, configurable threshold and max_results.",
                parameters={
                    "document_name": "str",
                    "query_text": "str",
                    "scope": "str",
                    "chapter_name": "str",
                    "similarity_threshold": "float",
                    "max_results": "int",
                },
                example='find_similar_text(document_name="My Book", query_text="character development themes", scope="document")',
                category="Content Access",
                planner_signature="find_similar_text(document_name: str, query_text: str, scope: str = 'document', chapter_name: str = None, similarity_threshold: float = 0.7, max_results: int = 10)",
                loading_priority=LoadingPriority.DEFERRED,
                scenarios=[
                    ToolScenario(
                        "Find content similar to 'the hero's journey' in my book", "Semantic search"
                    ),
                ],
            ),
            ToolDescription(
                name="find_entity",
                what="Find all mentions of an entity using aliases from metadata.",
                when="Use when user asks 'where did I mention Marcus?', 'find all references to the city', 'track character'.",
                returns="Aggregated mentions by chapter with first/last mention, total count, and context.",
                auto="Uses aliases from entities.yaml for comprehensive search.",
                parameters={
                    "document_name": "str",
                    "entity_name": "str",
                    "scope": "str",
                    "chapter_name": "str",
                },
                example='find_entity(document_name="My Book", entity_name="Marcus Chen", scope="document")',
                category="Content Access",
                planner_signature="find_entity(document_name: str, entity_name: str, scope: str = 'document', chapter_name: str = None)",
                loading_priority=LoadingPriority.DEFERRED,
                scenarios=[
                    ToolScenario("Where is Marcus mentioned in the document?", "Entity search"),
                ],
            ),
            # === METADATA MANAGEMENT (3 tools) ===
            ToolDescription(
                name="read_metadata",
                what="Read YAML frontmatter from chapters or document metadata files (entities, timeline).",
                when="Use when user asks about chapter status (draft/revised/complete), POV character, tags, character details, or timeline. NOT for file modification times - use check_content_status.",
                returns="Structured metadata object with YAML fields (status, pov_character, tags) or entity/timeline data.",
                auto="scope='chapter' (frontmatter), 'entity' (entities.yaml), 'timeline' (timeline.yaml).",
                parameters={"document_name": "str", "scope": "str", "target": "str"},
                example='read_metadata(document_name="My Book", scope="chapter", target="01-intro.md")',
                category="Metadata Management",
                planner_signature="read_metadata(document_name: str, scope: str, target: str = None)",
                loading_priority=LoadingPriority.DEFERRED,
                scenarios=[
                    ToolScenario(
                        "Read the YAML frontmatter metadata from chapter 02.md", "Read chapter metadata"
                    ),
                ],
            ),
            ToolDescription(
                name="write_metadata",
                what="Write or update YAML frontmatter for chapters or document-level metadata.",
                when="Use when user wants to set chapter status (draft/revised/complete), add character, update POV, or modify timeline.",
                returns="Confirmation with updated metadata summary.",
                auto="Creates automatic snapshot, supports chapters/entities/timeline.",
                parameters={
                    "document_name": "str",
                    "scope": "str",
                    "target": "str",
                    "status": "str",
                    "pov_character": "str",
                    "tags": "list[str]",
                },
                example='write_metadata(document_name="My Book", scope="chapter", target="01-intro.md", status="revised")',
                category="Metadata Management",
                planner_signature="write_metadata(document_name: str, scope: str, target: str = None, status: str = None, pov_character: str = None, tags: list[str] = None)",
                loading_priority=LoadingPriority.DEFERRED,
                scenarios=[
                    ToolScenario(
                        "Set the status of chapter 01-intro.md to 'revised'", "Write chapter metadata"
                    ),
                ],
            ),
            ToolDescription(
                name="list_metadata",
                what="List and filter metadata entries for chapters, entities, or timeline.",
                when="Use when user asks 'what characters exist?', 'which chapters are drafts?', 'list all entities', 'show timeline'.",
                returns="Filtered list of metadata entries with counts.",
                auto="Supports filtering by status, type, pov_character.",
                parameters={
                    "document_name": "str",
                    "scope": "str",
                    "filter_status": "str",
                    "filter_type": "str",
                },
                example='list_metadata(document_name="My Book", scope="chapters", filter_status="draft")',
                category="Metadata Management",
                planner_signature="list_metadata(document_name: str, scope: str, filter_status: str = None, filter_type: str = None)",
                loading_priority=LoadingPriority.DEFERRED,
                scenarios=[
                    ToolScenario(
                        "List all chapters that have status='draft' in their metadata", "Filter by metadata"
                    ),
                ],
            ),
            # === VERSION CONTROL (3 tools) ===
            ToolDescription(
                name="manage_snapshots",
                what="Create, list, restore, or delete document snapshots for version control.",
                when="Use when user wants to 'save version', 'create backup', 'list backups', 'restore previous version', 'undo changes'.",
                returns="Snapshot ID/list or restoration confirmation.",
                auto="action='create'|'list'|'restore'|'delete'. Auto-cleanup old snapshots.",
                parameters={"document_name": "str", "action": "str", "snapshot_id": "str", "message": "str"},
                example='manage_snapshots(document_name="My Book", action="create", message="First draft complete")',
                category="Version Control",
                planner_signature="manage_snapshots(document_name: str, action: str, snapshot_id: str = None, message: str = None)",
                loading_priority=LoadingPriority.DEFERRED,
                scenarios=[
                    ToolScenario("Create a backup of my novel before I make changes", "Create snapshot"),
                    ToolScenario("List all versions of my document", "List snapshots"),
                    ToolScenario("Restore my novel to the previous version", "Restore snapshot"),
                ],
            ),
            ToolDescription(
                name="check_content_status",
                what="Check content freshness, file modification timestamps, and edit history.",
                when="Use when user asks 'has this changed?', 'modification history', 'when was this last edited?', 'is this stale?'. NOT for YAML metadata status - use read_metadata.",
                returns="Freshness status (fresh/stale), last modified timestamp, optional history.",
                auto="Calculates freshness based on time window, tracks file-level modifications.",
                parameters={
                    "document_name": "str",
                    "chapter_name": "str",
                    "include_history": "bool",
                    "time_window": "str",
                },
                example='check_content_status(document_name="My Book", chapter_name="01-intro.md", include_history=True)',
                category="Version Control",
                planner_signature="check_content_status(document_name: str, chapter_name: str = None, include_history: bool = False, time_window: str = '24h')",
                loading_priority=LoadingPriority.DEFERRED,
                scenarios=[
                    ToolScenario(
                        "Has chapter 01-intro.md been modified recently?", "Check modification status"
                    ),
                ],
            ),
            ToolDescription(
                name="diff_content",
                what="Compare content between snapshots, current version, or files.",
                when="Use when user asks 'what changed?', 'show differences', 'compare versions', 'diff'.",
                returns="Unified diff output showing additions, deletions, and context.",
                auto="source_type/target_type='snapshot'|'current'. Supports unified/side-by-side formats.",
                parameters={
                    "document_name": "str",
                    "source_type": "str",
                    "source_id": "str",
                    "target_type": "str",
                    "target_id": "str",
                },
                example='diff_content(document_name="My Book", source_type="snapshot", source_id="snap_1", target_type="current")',
                category="Version Control",
                planner_signature="diff_content(document_name: str, source_type: str = 'snapshot', source_id: str = None, target_type: str = 'current', target_id: str = None)",
                loading_priority=LoadingPriority.DEFERRED,
                scenarios=[
                    ToolScenario("What changed in chapter 01.md since the last backup?", "Compare versions"),
                ],
            ),
            # === OVERVIEW (1 tool) ===
            ToolDescription(
                name="get_document_outline",
                what="Get comprehensive document structure with chapter metadata and entity counts.",
                when="Use when user asks 'what's the document structure?', 'give me an overview', 'show outline', 'table of contents with stats'.",
                returns="Hierarchical outline with chapter status, word counts, paragraph counts, entity mentions.",
                auto="Combines chapter metadata, statistics, and entity tracking.",
                parameters={
                    "document_name": "str",
                    "include_metadata": "bool",
                    "include_entity_counts": "bool",
                },
                example='get_document_outline(document_name="My Book", include_metadata=True)',
                category="Overview",
                planner_signature="get_document_outline(document_name: str, include_metadata: bool = True, include_entity_counts: bool = True)",
                loading_priority=LoadingPriority.DEFERRED,
                scenarios=[
                    ToolScenario("Show me the outline of my novel", "Document outline"),
                ],
            ),
            # === DISCOVERY (1 tool) ===
            ToolDescription(
                name="search_tool",
                what="META-TOOL: Search and discover available MCP tools by capability keyword.",
                when="Use when user asks about available tools, operations, or capabilities. Triggers: 'what tools', 'which tools', 'how can I', 'available operations'.",
                returns="List of matching tool names with descriptions and usage hints.",
                auto="Searches tool registry, NOT document content. Use for tool discovery only.",
                parameters={"query": "str", "category": "str"},
                example='search_tool(query="paragraph", category="Paragraph Operations")',
                category="Discovery",
                planner_signature="search_tool(query: str, category: str = None)",
                loading_priority=LoadingPriority.DISCOVERY,
                scenarios=[
                    ToolScenario(
                        "Help me find which MCP tool to use for paragraph editing", "Search for tools"
                    ),
                    ToolScenario(
                        "Search the tool catalog for metadata-related capabilities", "Category search"
                    ),
                ],
            ),
        ]

    # === QUERY METHODS ===

    def get_tool(self, name: str) -> ToolDescription | None:
        """Get a specific tool by name."""
        for tool in self._tools:
            if tool.name == name:
                return tool
        return None

    def get_tools_by_category(self) -> dict[str, list[ToolDescription]]:
        """Get tools organized by category."""
        categories: dict[str, list[ToolDescription]] = {}
        for tool in self._tools:
            if tool.category not in categories:
                categories[tool.category] = []
            categories[tool.category].append(tool)
        return categories

    def get_tools_by_priority(self) -> dict[LoadingPriority, list[ToolDescription]]:
        """Get tools organized by loading priority."""
        priorities: dict[LoadingPriority, list[ToolDescription]] = {p: [] for p in LoadingPriority}
        for tool in self._tools:
            priorities[tool.loading_priority].append(tool)
        return priorities

    def get_core_tools(self) -> list[ToolDescription]:
        """Get only core tools that should always be loaded."""
        return [t for t in self._tools if t.loading_priority == LoadingPriority.CORE]

    def get_all_tool_names(self) -> set[str]:
        """Get set of all tool names."""
        return {t.name for t in self._tools}

    # === FORMAT GENERATION ===

    def format_tool(self, tool: ToolDescription, format_type: ToolFormat) -> str:
        """Format a single tool description."""
        if format_type == ToolFormat.FULL:
            return f"""### {tool.name}
WHAT: {tool.what}
WHEN: {tool.when}
RETURNS: {tool.returns}
AUTO: {tool.auto}
Example: {tool.example}"""
        elif format_type == ToolFormat.COMPACT:
            return f"- {tool.name}: {tool.what}"
        elif format_type == ToolFormat.PLANNER:
            return f"- {tool.planner_signature}"
        else:  # MINIMAL
            return f"- {tool.name}"

    def get_formatted_tools(
        self, format_type: ToolFormat = ToolFormat.FULL, categories: list[str] | None = None
    ) -> str:
        """Get formatted tool descriptions."""
        lines = []
        tools_by_category = self.get_tools_by_category()

        for category, tools in sorted(tools_by_category.items()):
            if categories and category not in categories:
                continue
            lines.append(f"\n## {category}")
            for tool in tools:
                lines.append(self.format_tool(tool, format_type))

        return "\n".join(lines)

    # === SCENARIO EXPORT ===

    def get_all_scenarios(self) -> list[dict]:
        """Get all scenarios for benchmarking."""
        scenarios = []
        for tool in self._tools:
            for scenario in tool.scenarios:
                scenarios.append(
                    {
                        "query": scenario.query,
                        "expected_tool": tool.name,
                        "category": tool.category,
                        "description": scenario.description,
                    }
                )
        return scenarios

    def get_scenario_stats(self) -> dict:
        """Get statistics about scenario coverage."""
        tools_with_scenarios = {t.name for t in self._tools if t.scenarios}
        all_tools = {t.name for t in self._tools}
        total_scenarios = sum(len(t.scenarios) for t in self._tools)

        return {
            "total_scenarios": total_scenarios,
            "tools_with_scenarios": len(tools_with_scenarios),
            "total_tools": len(all_tools),
            "tools_missing_scenarios": sorted(all_tools - tools_with_scenarios),
            "categories": sorted({t.category for t in self._tools}),
        }

    # === DSPY EXPORT ===

    def get_dspy_descriptions(self) -> str:
        """Generate DSPy-format tool descriptions."""
        lines = ["Available MCP Tools (28 total):\n"]
        tools_by_category = self.get_tools_by_category()

        for category, tools in sorted(tools_by_category.items()):
            lines.append(f"\n{category.upper()}:")
            for tool in tools:
                lines.append(f"- {tool.name}: {tool.what}")

        return "\n".join(lines)

    def get_dspy_trainset(self) -> list[dict]:
        """Get training data for DSPy optimization."""
        return [{"query": s["query"], "expected_tool": s["expected_tool"]} for s in self.get_all_scenarios()]


# === MODULE-LEVEL CONVENIENCE FUNCTIONS ===

# Singleton instance
_manager: ToolDescriptionManager | None = None


def get_manager() -> ToolDescriptionManager:
    """Get the singleton ToolDescriptionManager instance."""
    global _manager
    if _manager is None:
        _manager = ToolDescriptionManager()
    return _manager


def get_all_scenarios() -> list[dict]:
    """Get all benchmark scenarios."""
    return get_manager().get_all_scenarios()


def get_dspy_tool_descriptions() -> str:
    """Get DSPy-format tool descriptions."""
    return get_manager().get_dspy_descriptions()


def get_dspy_trainset() -> list[dict]:
    """Get DSPy training data."""
    return get_manager().get_dspy_trainset()


def get_all_tool_names() -> set[str]:
    """Get set of all tool names."""
    return get_manager().get_all_tool_names()


def get_tool_descriptions_for_agent(agent_type: str = "simple") -> str:
    """Get formatted tool descriptions for a specific agent type.

    Based on DSPy benchmark results (Dec 2024), FULL format achieves 100% tool
    selection accuracy vs 97.3% for COMPACT. Both agents now use FULL format.
    """
    mgr = get_manager()
    # FULL format recommended for all agents based on benchmark results
    # full: 100%, compact: 97.3%, minimal: 94.6%
    return mgr.get_formatted_tools(ToolFormat.FULL)
