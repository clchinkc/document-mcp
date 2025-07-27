"""Shared tool descriptions for all document management agents.

This module provides structured tool descriptions that can be dynamically formatted
for different agent architectures. Instead of hardcoding tool descriptions in prompts,
agents request format-specific descriptions that match their operational patterns.
"""

from dataclasses import dataclass
from enum import Enum


class ToolFormat(Enum):
    """Format styles for tool descriptions based on agent architecture."""

    FULL = "full"  # Detailed examples with full parameter syntax
    COMPACT = "compact"  # Brief descriptions with simplified syntax
    PLANNER = "planner"  # Type-annotated signatures for planning
    MINIMAL = "minimal"  # Tool names only for reference


@dataclass
class ToolDescription:
    """Structured representation of a tool description."""

    name: str
    description: str
    parameters: dict[str, str]  # parameter_name: type_hint
    example: str
    category: str
    planner_signature: str  # Type signature for planner agent


class ToolDescriptionManager:
    """Manages tool descriptions for dynamic prompt generation across all agents."""

    def __init__(self):
        """Initialize the tool description registry."""
        self._tools = self._initialize_tools()

    def _initialize_tools(self) -> list[ToolDescription]:
        """Initialize all tool descriptions with standardized format."""
        return [
            # Document Management (6 tools)
            ToolDescription(
                name="list_documents",
                description="Lists all available documents",
                parameters={},
                example="list_documents()",
                category="Document Management",
                planner_signature="list_documents()",
            ),
            ToolDescription(
                name="create_document",
                description="Creates a new document directory",
                parameters={"document_name": "str"},
                example='create_document(document_name="My Book")',
                category="Document Management",
                planner_signature="create_document(document_name: str)",
            ),
            ToolDescription(
                name="delete_document",
                description="Deletes an entire document",
                parameters={"document_name": "str"},
                example='delete_document(document_name="My Book")',
                category="Document Management",
                planner_signature="delete_document(document_name: str)",
            ),
            ToolDescription(
                name="read_summary",
                description="Reads summary files with flexible scope (document, chapter, section). **Use this to read summaries**",
                parameters={"document_name": "str", "scope": "str", "target_name": "str | None"},
                example='read_summary(document_name="My Book", scope="document")',
                category="Document Management",
                planner_signature="read_summary(document_name: str, scope: str = 'document', target_name: str | None = None)",
            ),
            ToolDescription(
                name="write_summary",
                description="Writes or updates summary files with flexible scope (document, chapter, section)",
                parameters={
                    "document_name": "str",
                    "summary_content": "str",
                    "scope": "str",
                    "target_name": "str | None",
                },
                example='write_summary(document_name="My Book", summary_content="This book covers...", scope="document")',
                category="Document Management",
                planner_signature="write_summary(document_name: str, summary_content: str, scope: str = 'document', target_name: str | None = None)",
            ),
            ToolDescription(
                name="list_summaries",
                description="Lists all available summary files for a document",
                parameters={"document_name": "str"},
                example='list_summaries(document_name="My Book")',
                category="Document Management",
                planner_signature="list_summaries(document_name: str)",
            ),
            # Chapter Management (5 tools)
            ToolDescription(
                name="list_chapters",
                description="Lists all chapters in a document",
                parameters={"document_name": "str"},
                example='list_chapters(document_name="My Book")',
                category="Chapter Management",
                planner_signature="list_chapters(document_name: str)",
            ),
            ToolDescription(
                name="create_chapter",
                description="Creates a new chapter",
                parameters={
                    "document_name": "str",
                    "chapter_name": "str",
                    "initial_content": "str",
                },
                example='create_chapter(document_name="My Book", chapter_name="01-introduction.md", initial_content="# Introduction")',
                category="Chapter Management",
                planner_signature="create_chapter(document_name: str, chapter_name: str, initial_content: str)",
            ),
            ToolDescription(
                name="delete_chapter",
                description="Deletes a chapter",
                parameters={"document_name": "str", "chapter_name": "str"},
                example='delete_chapter(document_name="My Book", chapter_name="01-introduction.md")',
                category="Chapter Management",
                planner_signature="delete_chapter(document_name: str, chapter_name: str)",
            ),
            ToolDescription(
                name="write_chapter_content",
                description="Overwrites chapter content",
                parameters={
                    "document_name": "str",
                    "chapter_name": "str",
                    "new_content": "str",
                },
                example='write_chapter_content(document_name="My Book", chapter_name="01-intro.md", new_content="# New Content")',
                category="Chapter Management",
                planner_signature="write_chapter_content(document_name: str, chapter_name: str, new_content: str)",
            ),
            # Paragraph Operations (6 tools)
            ToolDescription(
                name="append_paragraph_to_chapter",
                description="Adds content to end of chapter",
                parameters={
                    "document_name": "str",
                    "chapter_name": "str",
                    "paragraph_content": "str",
                },
                example='append_paragraph_to_chapter(document_name="My Book", chapter_name="01-intro.md", paragraph_content="New paragraph.")',
                category="Paragraph Operations",
                planner_signature="append_paragraph_to_chapter(document_name: str, chapter_name: str, paragraph_content: str)",
            ),
            ToolDescription(
                name="replace_paragraph",
                description="Replaces a specific paragraph",
                parameters={
                    "document_name": "str",
                    "chapter_name": "str",
                    "paragraph_index": "int",
                    "new_content": "str",
                },
                example='replace_paragraph(document_name="My Book", chapter_name="01-intro.md", paragraph_index=0, new_content="Updated text.")',
                category="Paragraph Operations",
                planner_signature="replace_paragraph(document_name: str, chapter_name: str, paragraph_index: int, new_content: str)",
            ),
            ToolDescription(
                name="insert_paragraph_before",
                description="Inserts paragraph before specified index",
                parameters={
                    "document_name": "str",
                    "chapter_name": "str",
                    "paragraph_index": "int",
                    "new_content": "str",
                },
                example='insert_paragraph_before(document_name="My Book", chapter_name="01-intro.md", paragraph_index=1, new_content="New paragraph.")',
                category="Paragraph Operations",
                planner_signature="insert_paragraph_before(document_name: str, chapter_name: str, paragraph_index: int, new_content: str)",
            ),
            ToolDescription(
                name="insert_paragraph_after",
                description="Inserts paragraph after specified index",
                parameters={
                    "document_name": "str",
                    "chapter_name": "str",
                    "paragraph_index": "int",
                    "new_content": "str",
                },
                example='insert_paragraph_after(document_name="My Book", chapter_name="01-intro.md", paragraph_index=0, new_content="New paragraph.")',
                category="Paragraph Operations",
                planner_signature="insert_paragraph_after(document_name: str, chapter_name: str, paragraph_index: int, new_content: str)",
            ),
            ToolDescription(
                name="delete_paragraph",
                description="Deletes a specific paragraph",
                parameters={
                    "document_name": "str",
                    "chapter_name": "str",
                    "paragraph_index": "int",
                },
                example='delete_paragraph(document_name="My Book", chapter_name="01-intro.md", paragraph_index=2)',
                category="Paragraph Operations",
                planner_signature="delete_paragraph(document_name: str, chapter_name: str, paragraph_index: int)",
            ),
            ToolDescription(
                name="move_paragraph_before",
                description="Moves paragraph to new position",
                parameters={
                    "document_name": "str",
                    "chapter_name": "str",
                    "move_paragraph_index": "int",
                    "target_paragraph_index": "int",
                },
                example='move_paragraph_before(document_name="My Book", chapter_name="01-intro.md", paragraph_to_move_index=3, target_paragraph_index=1)',
                category="Paragraph Operations",
                planner_signature="move_paragraph_before(document_name: str, chapter_name: str, move_paragraph_index: int, target_paragraph_index: int)",
            ),
            ToolDescription(
                name="move_paragraph_to_end",
                description="Moves paragraph to end",
                parameters={
                    "document_name": "str",
                    "chapter_name": "str",
                    "paragraph_index": "int",
                },
                example='move_paragraph_to_end(document_name="My Book", chapter_name="01-intro.md", paragraph_to_move_index=0)',
                category="Paragraph Operations",
                planner_signature="move_paragraph_to_end(document_name: str, chapter_name: str, paragraph_index: int)",
            ),
            # Scope-based Content Access (4 tools - replacing 9 individual tools)
            ToolDescription(
                name="read_content",
                description="Flexible content reading with scope-based targeting. Use scope='document' for full document, scope='chapter' for specific chapter, scope='paragraph' for specific paragraph. Essential for content access operations.",
                parameters={
                    "document_name": "str",
                    "scope": "str",
                    "chapter_name": "Optional[str]",
                    "paragraph_index": "Optional[int]",
                },
                example='read_content(document_name="My Book", scope="chapter", chapter_name="01-intro.md")',
                category="Scope-based Content Access",
                planner_signature="read_content(document_name: str, scope: str = 'document', chapter_name: str = None, paragraph_index: int = None)",
            ),
            ToolDescription(
                name="find_text",
                description="Text search with scope-based targeting and case sensitivity control. Use scope='document' to search entire document, scope='chapter' to search specific chapter. Returns locations of matching text.",
                parameters={
                    "document_name": "str",
                    "search_text": "str",
                    "scope": "str",
                    "chapter_name": "Optional[str]",
                    "case_sensitive": "bool",
                },
                example='find_text(document_name="My Book", search_text="search term", scope="document", case_sensitive=False)',
                category="Scope-based Content Access",
                planner_signature="find_text(document_name: str, search_text: str, scope: str = 'document', chapter_name: str = None, case_sensitive: bool = False)",
            ),
            ToolDescription(
                name="replace_text",
                description="Text replacement with scope-based targeting. Use scope='document' to replace across entire document, scope='chapter' to replace within specific chapter. Performs find-and-replace operations efficiently.",
                parameters={
                    "document_name": "str",
                    "find_text": "str",
                    "replace_text": "str",
                    "scope": "str",
                    "chapter_name": "Optional[str]",
                },
                example='replace_text(document_name="My Book", find_text="old", replace_text="new", scope="document")',
                category="Scope-based Content Access",
                planner_signature="replace_text(document_name: str, find_text: str, replace_text: str, scope: str = 'document', chapter_name: str = None)",
            ),
            ToolDescription(
                name="get_statistics",
                description="Statistics collection with scope-based targeting. Use scope='document' for full document metrics, scope='chapter' for specific chapter metrics. Returns word count, paragraph count, and other content statistics.",
                parameters={
                    "document_name": "str",
                    "scope": "str",
                    "chapter_name": "Optional[str]",
                },
                example='get_statistics(document_name="My Book", scope="document")',
                category="Scope-based Content Access",
                planner_signature="get_statistics(document_name: str, scope: str = 'document', chapter_name: str = None)",
            ),
            ToolDescription(
                name="find_similar_text",
                description="Semantic text search with scope-based targeting and similarity scoring. Use scope='document' to search entire document, scope='chapter' to search specific chapter. Returns contextually similar content based on meaning rather than exact matches.",
                parameters={
                    "document_name": "str",
                    "query_text": "str",
                    "scope": "str",
                    "chapter_name": "Optional[str]",
                    "similarity_threshold": "float",
                    "max_results": "int",
                },
                example='find_similar_text(document_name="My Book", query_text="character development themes", scope="document", similarity_threshold=0.7)',
                category="Scope-based Content Access",
                planner_signature="find_similar_text(document_name: str, query_text: str, scope: str = 'document', chapter_name: str = None, similarity_threshold: float = 0.7, max_results: int = 10)",
            ),
            # Version Control Tools (3 tools replacing 6)
            ToolDescription(
                name="manage_snapshots",
                description="Comprehensive snapshot management: create, list, or restore snapshots with action-based interface",
                parameters={
                    "document_name": "str",
                    "action": "str",
                    "snapshot_id": "str",
                    "message": "str",
                    "auto_cleanup": "bool",
                },
                example='manage_snapshots(document_name="My Book", action="create", message="First draft complete")',
                category="Version Control",
                planner_signature="manage_snapshots(document_name: str, action: str, snapshot_id: str = None, message: str = None, auto_cleanup: bool = True)",
            ),
            ToolDescription(
                name="check_content_status",
                description="Content status checker: freshness validation with optional history tracking",
                parameters={
                    "document_name": "str",
                    "chapter_name": "str",
                    "include_history": "bool",
                    "time_window": "str",
                    "last_known_modified": "str",
                },
                example='check_content_status(document_name="My Book", chapter_name="01-intro.md", include_history=True)',
                category="Version Control",
                planner_signature="check_content_status(document_name: str, chapter_name: str = None, include_history: bool = False, time_window: str = '24h', last_known_modified: str = None)",
            ),
            ToolDescription(
                name="diff_content",
                description="Content comparison: flexible diff between snapshots, current content, and files",
                parameters={
                    "document_name": "str",
                    "source_type": "str",
                    "source_id": "str",
                    "target_type": "str",
                    "target_id": "str",
                    "output_format": "str",
                    "chapter_name": "str",
                },
                example='diff_content(document_name="My Book", source_type="snapshot", source_id="snap_1", target_type="current")',
                category="Version Control",
                planner_signature="diff_content(document_name: str, source_type: str = 'snapshot', source_id: str = None, target_type: str = 'current', target_id: str = None, output_format: str = 'unified', chapter_name: str = None)",
            ),
            # Batch Operations (1 tool)
            ToolDescription(
                name="batch_apply_operations",
                description="""Execute multiple document operations atomically with comprehensive safety and rollback.

INTELLIGENCE FEATURES:
• Automatic dependency resolution - operations execute in correct order regardless of definition order
• Smart conflict detection - prevents contradictory operations (e.g., delete then modify same content)
• Automatic snapshot creation - every batch gets a restoration checkpoint
• Granular rollback - failed batches automatically restore to pre-execution state
• User operation tracking - all changes attributed and logged for easy restoration

WHEN TO USE BATCHES:
[OK] Multi-step document creation (document + chapters + content)
[OK] Bulk content editing (character renaming, formatting changes)
[OK] Complex reorganization (moving/restructuring multiple elements)
[OK] Multi-document operations requiring consistency
[OK] Any workflow where partial completion would leave incomplete state

WHEN TO USE INDIVIDUAL OPERATIONS:
[X] Single, simple edits (one paragraph change)
[X] Exploratory operations where you need to observe results
[X] Trial-and-error workflows requiring intermediate feedback
[X] Operations that depend on external input or validation""",
                parameters={
                    "operations": "List[Dict] - List of operations, each with: operation_type (str), target (Dict), parameters (Dict), order (int), operation_id (str), depends_on (List[str], optional)",
                    "atomic": "bool - True: all succeed or all rollback (recommended for most use cases)",
                    "validate_only": "bool - True: dry-run validation without execution (test complex batches first)",
                    "snapshot_before": "bool - True: create named restoration point (automatic for edit operations)",
                    "continue_on_error": "bool - False: stop on first error (safer), True: continue despite failures",
                    "execution_mode": "str - 'sequential' (default, safer) or 'parallel_safe' (faster for independent ops)",
                },
                example="""batch_apply_operations(
    operations=[
        {
            "operation_type": "create_document",
            "target": {},
            "parameters": {"document_name": "Science Fiction Novel"},
            "order": 1,
            "operation_id": "create_doc"
        },
        {
            "operation_type": "create_chapter",
            "target": {"document_name": "Science Fiction Novel"},
            "parameters": {
                "chapter_name": "01-introduction.md",
                "initial_content": "# The Journey Begins\\n\\nIn a galaxy far away..."
            },
            "order": 2,
            "operation_id": "create_intro",
            "depends_on": ["create_doc"]
        }
    ],
    atomic=True,
    snapshot_before=True
)""",
                category="Batch Operations",
                planner_signature="batch_apply_operations(operations: List[Dict], atomic: bool = True, validate_only: bool = False, snapshot_before: bool = False, continue_on_error: bool = False, execution_mode: str = 'sequential')",
            ),
        ]

    def get_tools_by_category(self) -> dict[str, list[ToolDescription]]:
        """Get tools organized by category."""
        categories = {}
        for tool in self._tools:
            if tool.category not in categories:
                categories[tool.category] = []
            categories[tool.category].append(tool)
        return categories

    def get_tool_descriptions_text(self, format_type: ToolFormat = ToolFormat.FULL) -> str:
        """Generate tool descriptions text for prompt inclusion."""
        if format_type == ToolFormat.COMPACT:
            return self._generate_compact_format()
        elif format_type == ToolFormat.MINIMAL:
            return self._generate_minimal_format()
        elif format_type == ToolFormat.PLANNER:
            return self._generate_planner_format()
        else:
            return self._generate_full_format()

    def _generate_full_format(self) -> str:
        """Generate full format tool descriptions (ReAct agent format)."""
        categories = self.get_tools_by_category()
        sections = []

        for category, tools in categories.items():
            sections.append(f"**{category}:**")
            for tool in tools:
                sections.append(f"- `{tool.example}` - {tool.description}")

        return "\n".join(sections)

    def _generate_compact_format(self) -> str:
        """Generate compact format tool descriptions (Simple agent format)."""
        lines = []
        for tool in self._tools:
            lines.append(f"- `{tool.example}`: {tool.description}")
        return "\n".join(lines)

    def _generate_minimal_format(self) -> str:
        """Generate minimal format tool descriptions."""
        tool_names = [tool.name for tool in self._tools]
        return f"Available tools: {', '.join(tool_names)}"

    def _generate_planner_format(self) -> str:
        """Generate planner format tool descriptions with type hints."""
        categories = self.get_tools_by_category()
        sections = []

        for category, tools in categories.items():
            sections.append(f"**{category} ({len(tools)} tools):**")
            for tool in tools:
                sections.append(f"- `{tool.planner_signature}` - {tool.description}")

        return "\n".join(sections)

    def get_tool_count(self) -> int:
        """Get total number of tools."""
        return len(self._tools)

    def get_category_count(self) -> int:
        """Get number of tool categories."""
        return len({tool.category for tool in self._tools})

    def get_token_estimate(self, format_type: ToolFormat = ToolFormat.FULL) -> int:
        """Estimate token count for tool descriptions (rough approximation)."""
        text = self.get_tool_descriptions_text(format_type)
        # Rough estimate: 1 token ≈ 4 characters
        return len(text) // 4

    def compare_formats(self) -> dict[str, tuple[int, int]]:
        """Compare character and token usage across different formats."""
        formats = {}
        for format_type in ToolFormat:
            text = self.get_tool_descriptions_text(format_type)
            formats[format_type.value] = (
                len(text),
                self.get_token_estimate(format_type),
            )
        return formats


# Global instance
tool_manager = ToolDescriptionManager()


def get_tool_descriptions_for_agent(agent_type: str) -> str:
    """Get format-appropriate tool descriptions for the specified agent architecture.

    Each agent type uses a different format optimized for its operational pattern:
    - simple: Compact format for single-operation workflows
    - react: Full format with examples for multi-step reasoning
    - planner: Type-annotated format for plan generation
    """
    format_map = {
        "simple": ToolFormat.COMPACT,
        "react": ToolFormat.FULL,
        "planner": ToolFormat.PLANNER,
    }

    format_type = format_map.get(agent_type.lower(), ToolFormat.FULL)
    return tool_manager.get_tool_descriptions_text(format_type)
