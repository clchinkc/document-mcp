"""Tool Definitions for Benchmarking.

This module defines tool sets for A/B comparison:
1. Different tool implementations (4-tool vs 8-tool vs 2-tool)
2. Different description styles for the same tools

The MCP server's default is 4 paragraph tools.
Benchmarks can test alternative implementations to compare accuracy and usability.
"""

from dataclasses import dataclass
from dataclasses import field


@dataclass
class ToolDescription:
    """Description of a single tool."""

    name: str
    what: str
    when: str

    def to_prompt(self) -> str:
        """Format for LLM prompt."""
        return f"- {self.name}: {self.what} {self.when}"


@dataclass
class ToolSet:
    """A set of tools for benchmarking.

    Attributes:
        name: Human-readable name
        tool_count: Number of tools in this set
        tools: List of tool descriptions
        tool_mapping: Maps operation -> tool name for this set
    """

    name: str
    tool_count: int
    tools: list[ToolDescription]
    tool_mapping: dict[str, str] = field(default_factory=dict)

    def to_prompt(self) -> str:
        """Format all tools for LLM prompt."""
        header = f"Available MCP tools for paragraph operations ({self.name}):\n"
        return header + "\n".join(tool.to_prompt() for tool in self.tools)

    @property
    def tool_names(self) -> list[str]:
        """Get list of tool names."""
        return [tool.name for tool in self.tools]

    def get_expected_tool(self, operation: str) -> str:
        """Get expected tool name for an operation.

        Args:
            operation: Semantic operation (insert_before, insert_after, append,
                      replace, delete, move_before, move_to_end)

        Returns:
            Tool name for this tool set
        """
        return self.tool_mapping.get(operation, operation)


# =============================================================================
# TOOL SET: DEFAULT (4 TOOLS) - Current MCP server implementation
# =============================================================================

DEFAULT_4_TOOLS = ToolSet(
    name="Default (4 tools)",
    tool_count=4,
    tools=[
        ToolDescription(
            name="add_paragraph",
            what="Add new paragraph at position: BEFORE target, AFTER target, or at END.",
            when=(
                "Use for ALL new paragraph additions. "
                "position='before'|'after'|'end'. "
                "index required for 'before'/'after', omit for 'end'."
            ),
        ),
        ToolDescription(
            name="replace_paragraph",
            what="REPLACE/OVERWRITE content at index - paragraph count unchanged.",
            when="Use for: 'replace N', 'edit N', 'change N'. NOT for adding new.",
        ),
        ToolDescription(
            name="delete_paragraph",
            what="DELETE/REMOVE at index - subsequent paragraphs shift up.",
            when="Use for: 'delete N', 'remove N'. DESTRUCTIVE.",
        ),
        ToolDescription(
            name="move_paragraph",
            what="Move paragraph to new position (before or after target).",
            when=(
                "Use for: 'move X before Y', 'move X after Y', 'move X to end'. "
                "source_index required. destination='before'|'after'. "
                "target_index=null with 'after' moves to end."
            ),
        ),
    ],
    tool_mapping={
        "insert_before": "add_paragraph",
        "insert_after": "add_paragraph",
        "append": "add_paragraph",
        "replace": "replace_paragraph",
        "delete": "delete_paragraph",
        "move_before": "move_paragraph",
        "move_after": "move_paragraph",
        "move_to_end": "move_paragraph",
    },
)


# =============================================================================
# TOOL SET: ATOMIC (9 TOOLS) - One tool per operation
# =============================================================================

ATOMIC_8_TOOLS = ToolSet(
    name="Atomic (9 tools)",
    tool_count=9,
    tools=[
        ToolDescription(
            name="insert_paragraph_before",
            what="Insert new paragraph BEFORE specified index.",
            when="Use ONLY when inserting BEFORE/ABOVE an existing paragraph.",
        ),
        ToolDescription(
            name="insert_paragraph_after",
            what="Insert new paragraph AFTER specified index.",
            when="Use ONLY when inserting AFTER/BELOW an existing paragraph.",
        ),
        ToolDescription(
            name="append_paragraph_to_chapter",
            what="Append new paragraph at END of chapter.",
            when="Use ONLY when adding to the END. No index needed.",
        ),
        ToolDescription(
            name="replace_paragraph",
            what="REPLACE content at index. Count unchanged.",
            when="Use for 'replace N', 'edit N', 'change N'.",
        ),
        ToolDescription(
            name="delete_paragraph",
            what="DELETE at index. Subsequent paragraphs shift up.",
            when="Use for 'delete N', 'remove N'. DESTRUCTIVE.",
        ),
        ToolDescription(
            name="read_paragraph",
            what="Read content of a single paragraph.",
            when="Use to read specific paragraph content.",
        ),
        ToolDescription(
            name="move_paragraph_before",
            what="Move paragraph BEFORE a target index.",
            when="Use for 'move X before Y'.",
        ),
        ToolDescription(
            name="move_paragraph_after",
            what="Move paragraph AFTER a target index.",
            when="Use for 'move X after Y'.",
        ),
        ToolDescription(
            name="move_paragraph_to_end",
            what="Move paragraph to END of chapter.",
            when="Use for 'move X to end'.",
        ),
    ],
    tool_mapping={
        "insert_before": "insert_paragraph_before",
        "insert_after": "insert_paragraph_after",
        "append": "append_paragraph_to_chapter",
        "replace": "replace_paragraph",
        "delete": "delete_paragraph",
        "move_before": "move_paragraph_before",
        "move_after": "move_paragraph_after",
        "move_to_end": "move_paragraph_to_end",
    },
)


# =============================================================================
# TOOL SET: CONSOLIDATED (2 TOOLS) - Maximum consolidation
# =============================================================================

CONSOLIDATED_2_TOOLS = ToolSet(
    name="Consolidated (2 tools)",
    tool_count=2,
    tools=[
        ToolDescription(
            name="modify_paragraph",
            what="Unified paragraph modification: insert, replace, delete, or append.",
            when=(
                "operation='insert_before'|'insert_after'|'append'|'replace'|'delete'. "
                "index required except for 'append'. content required except for 'delete'."
            ),
        ),
        ToolDescription(
            name="move_paragraph",
            what="Move paragraph to new position.",
            when="source_index + destination='before'|'to_end' + optional target_index.",
        ),
    ],
    tool_mapping={
        "insert_before": "modify_paragraph",
        "insert_after": "modify_paragraph",
        "append": "modify_paragraph",
        "replace": "modify_paragraph",
        "delete": "modify_paragraph",
        "move_before": "move_paragraph",
        "move_to_end": "move_paragraph",
    },
)


# =============================================================================
# DESCRIPTION STYLE VARIANTS (for same 4-tool implementation)
# =============================================================================

MINIMAL_4_TOOLS = ToolSet(
    name="Minimal (4 tools)",
    tool_count=4,
    tools=[
        ToolDescription(
            name="add_paragraph",
            what="Add paragraph (before/after index, or at end).",
            when="position + optional index.",
        ),
        ToolDescription(
            name="replace_paragraph",
            what="Replace paragraph at index.",
            when="index + new_content.",
        ),
        ToolDescription(
            name="delete_paragraph",
            what="Delete paragraph at index.",
            when="index only.",
        ),
        ToolDescription(
            name="move_paragraph",
            what="Move paragraph (before target or to end).",
            when="source + destination + optional target.",
        ),
    ],
    tool_mapping=DEFAULT_4_TOOLS.tool_mapping,
)

VERBOSE_4_TOOLS = ToolSet(
    name="Verbose (4 tools)",
    tool_count=4,
    tools=[
        ToolDescription(
            name="add_paragraph",
            what="Add a new paragraph to the chapter at a specified position. Supports three modes: 'before' inserts above the target index (pushes existing content down), 'after' inserts below the target index, 'end' appends to the end of the chapter.",
            when="Use this tool whenever you need to add NEW content to a chapter. Required params: content (the text), position ('before'|'after'|'end'). If position is 'before' or 'after', also provide paragraph_index.",
        ),
        ToolDescription(
            name="replace_paragraph",
            what="Replace the content of an existing paragraph at the specified index. The total number of paragraphs remains unchanged. This overwrites the existing text completely.",
            when="Use when editing or updating existing content. DO NOT use for adding new paragraphs. Required params: paragraph_index, new_content.",
        ),
        ToolDescription(
            name="delete_paragraph",
            what="Permanently remove a paragraph from the chapter. All paragraphs after the deleted one shift up (their indices decrease by 1). This is a destructive operation.",
            when="Use when content needs to be removed entirely. Required param: paragraph_index. WARNING: Cannot be undone without snapshots.",
        ),
        ToolDescription(
            name="move_paragraph",
            what="Reorder paragraphs by moving one to a new position. Supports moving before another paragraph or to the end of the chapter.",
            when="Use for reordering content. Required: source_index, destination ('before'|'to_end'). If destination is 'before', also provide target_index.",
        ),
    ],
    tool_mapping=DEFAULT_4_TOOLS.tool_mapping,
)


# =============================================================================
# TOOL SET REGISTRY
# =============================================================================

# Tool implementations (different tool counts)
TOOL_IMPLEMENTATIONS = {
    "default": DEFAULT_4_TOOLS,
    "4-tool": DEFAULT_4_TOOLS,
    "8-tool": ATOMIC_8_TOOLS,
    "atomic": ATOMIC_8_TOOLS,
    "2-tool": CONSOLIDATED_2_TOOLS,
    "consolidated": CONSOLIDATED_2_TOOLS,
}

# Description styles (same 4 tools, different descriptions)
DESCRIPTION_STYLES = {
    "default": DEFAULT_4_TOOLS,
    "minimal": MINIMAL_4_TOOLS,
    "verbose": VERBOSE_4_TOOLS,
}

# Combined registry
TOOL_SETS = {
    **TOOL_IMPLEMENTATIONS,
    "minimal": MINIMAL_4_TOOLS,
    "verbose": VERBOSE_4_TOOLS,
}


def get_tool_set(name: str = "default") -> ToolSet:
    """Get a tool set by name.

    Args:
        name: Tool set name. Options:
            - Implementation: 'default', '4-tool', '8-tool', 'atomic', '2-tool', 'consolidated'
            - Description style: 'minimal', 'verbose'

    Returns:
        ToolSet for benchmarking
    """
    if name not in TOOL_SETS:
        raise ValueError(f"Unknown tool set: {name}. Available: {list(TOOL_SETS.keys())}")
    return TOOL_SETS[name]


def get_implementation_tool_sets() -> dict[str, ToolSet]:
    """Get tool sets that represent different implementations (different tool counts)."""
    return {
        "4-tool": DEFAULT_4_TOOLS,
        "8-tool": ATOMIC_8_TOOLS,
        "2-tool": CONSOLIDATED_2_TOOLS,
    }


def get_description_style_tool_sets() -> dict[str, ToolSet]:
    """Get tool sets that represent different description styles (same tools)."""
    return {
        "default": DEFAULT_4_TOOLS,
        "minimal": MINIMAL_4_TOOLS,
        "verbose": VERBOSE_4_TOOLS,
    }


# Backward compatibility aliases
ATOMIC_PARAGRAPH_TOOLS = ATOMIC_8_TOOLS
CONSOLIDATED_PARAGRAPH_TOOLS = CONSOLIDATED_2_TOOLS
DEFAULT_TOOLS = DEFAULT_4_TOOLS
MINIMAL_TOOLS = MINIMAL_4_TOOLS
VERBOSE_TOOLS = VERBOSE_4_TOOLS
