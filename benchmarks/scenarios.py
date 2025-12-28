"""MCP Tool Selection Benchmark Scenarios.

Two scenario sets are provided:

1. PARAGRAPH SCENARIOS: Test paragraph manipulation tool selection
   (add, replace, delete, move) across complexity levels:
   - Level 1: Single tool selection (basic paragraph operations)
   - Level 2: Sequential operations (2-3 tool chains)
   - Level 3: Complex reasoning (context-dependent)
   - Level 4: Ambiguous (multiple valid solutions)
   - Level 5: Edge cases (boundary conditions)
   - Level 6: Adversarial (intentionally confusing phrasing)

2. COMPREHENSIVE SCENARIOS: All 28 MCP tools imported from the tool registry.

Use get_all_tools_trainset() for comprehensive tool selection optimization.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from enum import Enum

from src.agents.shared.tool_descriptions import get_all_scenarios as _get_tool_scenarios


class ComplexityLevel(Enum):
    """Scenario complexity levels."""

    SIMPLE = 1  # Single tool selection
    SEQUENTIAL = 2  # 2-3 tool chain
    COMPLEX = 3  # Requires reasoning about context
    AMBIGUOUS = 4  # Multiple valid solutions


@dataclass
class Scenario:
    """A single benchmark scenario.

    Attributes:
        query: The user's natural language query
        operation: Semantic operation (insert_before, insert_after, append, replace, delete, move_before, move_to_end)
        category: Category for grouping (insert, replace, append, move, delete)
        level: Complexity level
    """

    query: str
    operation: str  # Semantic operation that maps to tool names via ToolSet.get_expected_tool()
    category: str = "paragraph"
    level: ComplexityLevel = ComplexityLevel.SIMPLE

    @property
    def clean_query(self) -> str:
        """Get query without context prefix."""
        if ". " in self.query:
            return self.query.split(". ")[-1]
        return self.query

    def get_expected_tool(self, tool_set_name: str = "default") -> str:
        """Get expected tool for a specific tool set.

        Args:
            tool_set_name: Name of the tool set (default, 4-tool, 8-tool, 2-tool, etc.)

        Returns:
            Expected tool name for this scenario with the given tool set
        """
        from .tools import get_tool_set

        tool_set = get_tool_set(tool_set_name)
        return tool_set.get_expected_tool(self.operation)


@dataclass
class SequentialScenario:
    """A multi-step scenario requiring sequential tool calls."""

    query: str
    expected_sequence: list[str]  # Expected tools in order
    category: str = "sequential"
    level: ComplexityLevel = ComplexityLevel.SEQUENTIAL
    context_required: bool = False
    reasoning_steps: list[str] = field(default_factory=list)

    @property
    def clean_query(self) -> str:
        if ". " in self.query:
            return self.query.split(". ")[-1]
        return self.query


@dataclass
class AmbiguousScenario:
    """A scenario with multiple valid tool choices."""

    query: str
    valid_tools: list[str]  # All acceptable tools
    preferred_tool: str  # Most appropriate choice
    category: str = "ambiguous"
    level: ComplexityLevel = ComplexityLevel.AMBIGUOUS
    ambiguity_reason: str = ""


# Document context prefix for scenarios
DOC_CONTEXT = (
    "Document 'novel' has chapter '01-intro.md' with paragraphs: "
    "[0: '# Intro', 1: 'Paragraph one.', 2: 'Paragraph two.', 3: 'Paragraph three.']. "
)


# =============================================================================
# PARAGRAPH OPERATION SCENARIOS
# =============================================================================

PARAGRAPH_SCENARIOS = [
    # Insert BEFORE operations
    Scenario(
        query=f"{DOC_CONTEXT}Insert 'New text' BEFORE paragraph 2",
        operation="insert_before",
        category="insert",
    ),
    Scenario(
        query=f"{DOC_CONTEXT}Put 'New text' above paragraph 1",
        operation="insert_before",
        category="insert",
    ),
    # Insert AFTER operations
    Scenario(
        query=f"{DOC_CONTEXT}Add 'New text' AFTER paragraph 3",
        operation="insert_after",
        category="insert",
    ),
    Scenario(
        query=f"{DOC_CONTEXT}Add 'New text' below paragraph 2",
        operation="insert_after",
        category="insert",
    ),
    # Replace operations
    Scenario(
        query=f"{DOC_CONTEXT}Replace paragraph 2 with 'Updated text'",
        operation="replace",
        category="replace",
    ),
    Scenario(
        query=f"{DOC_CONTEXT}Edit paragraph 3 to say 'New content'",
        operation="replace",
        category="replace",
    ),
    # Append operations (add to END)
    Scenario(
        query=f"{DOC_CONTEXT}Add 'New paragraph' at the end",
        operation="append",
        category="append",
    ),
    Scenario(
        query=f"{DOC_CONTEXT}Continue writing with 'More content' at the end",
        operation="append",
        category="append",
    ),
    # Move BEFORE operations
    Scenario(
        query=f"{DOC_CONTEXT}Move paragraph 3 to before paragraph 1",
        operation="move_before",
        category="move",
    ),
    # Move TO END operations
    Scenario(
        query=f"{DOC_CONTEXT}Push paragraph 2 to the end",
        operation="move_to_end",
        category="move",
    ),
]


# =============================================================================
# LEVEL 2: SEQUENTIAL SCENARIOS (Multi-step operations)
# =============================================================================

SEQUENTIAL_SCENARIOS = [
    SequentialScenario(
        query="Create a new chapter 'epilogue' and add a paragraph saying 'The end'",
        expected_sequence=["create_chapter", "add_paragraph"],
        category="create_and_write",
        reasoning_steps=["Create the chapter first", "Then add content to it"],
    ),
    SequentialScenario(
        query="Find all paragraphs mentioning 'Marcus' and move the first one to the end",
        expected_sequence=["find_text", "move_paragraph"],
        category="search_and_modify",
        context_required=True,
        reasoning_steps=["Search for mentions", "Identify first match", "Move it"],
    ),
    SequentialScenario(
        query="Read chapter 2, then replace paragraph 3 with a summary",
        expected_sequence=["read_content", "replace_paragraph"],
        category="read_and_modify",
        context_required=True,
    ),
    SequentialScenario(
        query="Delete paragraph 5 and then move paragraph 3 to the end",
        expected_sequence=["delete_paragraph", "move_paragraph"],
        category="multi_modify",
        reasoning_steps=["Delete first (indices will shift)", "Then move"],
    ),
    SequentialScenario(
        query="Copy the content from chapter 1 paragraph 2 to the end of chapter 3",
        expected_sequence=["read_content", "add_paragraph"],
        category="copy_content",
        context_required=True,
    ),
]


# =============================================================================
# LEVEL 3: COMPLEX SCENARIOS (Reasoning required)
# =============================================================================

COMPLEX_CONTEXT = (
    "Document 'novel' has: chapter '01-intro.md' (paragraphs 0-4), "
    "chapter '02-rising.md' (paragraphs 0-6), chapter '03-climax.md' (paragraphs 0-3). "
    "Entity 'Marcus' appears in paragraphs 2,4 of chapter 1 and paragraph 3 of chapter 2. "
)

COMPLEX_SCENARIOS = [
    Scenario(
        query=f"{COMPLEX_CONTEXT}Add a new paragraph about Marcus's backstory right after his first mention",
        operation="insert_after",
        category="context_reasoning",
        level=ComplexityLevel.COMPLEX,
    ),
    Scenario(
        query=f"{COMPLEX_CONTEXT}The chapter 2 paragraph about Marcus should come before the chapter 1 mentions. Fix this.",
        operation="move_before",
        category="context_reasoning",
        level=ComplexityLevel.COMPLEX,
    ),
    Scenario(
        query="Document has chapters 1-5. Chapter 3 is too long. Split paragraph 4 into two paragraphs.",
        operation="insert_after",
        category="inference",
        level=ComplexityLevel.COMPLEX,
    ),
    Scenario(
        query="The last paragraph of each chapter should be a cliffhanger. Add one to chapter 2 if missing.",
        operation="append",
        category="conditional",
        level=ComplexityLevel.COMPLEX,
    ),
    Scenario(
        query="Paragraph 3 was accidentally duplicated as paragraph 4. Remove the duplicate.",
        operation="delete",
        category="dedup",
        level=ComplexityLevel.COMPLEX,
    ),
]


# =============================================================================
# LEVEL 4: AMBIGUOUS SCENARIOS (Multiple valid solutions)
# =============================================================================

AMBIGUOUS_SCENARIOS = [
    AmbiguousScenario(
        query=f"{DOC_CONTEXT}Add some text about the sunset at the end of the intro chapter",
        valid_tools=["add_paragraph", "add_paragraph"],
        preferred_tool="add_paragraph",
        ambiguity_reason="'At the end' could mean append or insert after last paragraph",
    ),
    AmbiguousScenario(
        query=f"{DOC_CONTEXT}Update paragraph 2 to include more details",
        valid_tools=["replace_paragraph", "add_paragraph"],
        preferred_tool="replace_paragraph",
        ambiguity_reason="'Update' could mean replace content or add after it",
    ),
    AmbiguousScenario(
        query=f"{DOC_CONTEXT}The paragraph about weather should come earlier",
        valid_tools=["move_paragraph", "move_paragraph"],
        preferred_tool="move_paragraph",
        ambiguity_reason="'Earlier' is relative - needs context to determine target position",
    ),
    AmbiguousScenario(
        query=f"{DOC_CONTEXT}Fix paragraph 3 - it doesn't flow well",
        valid_tools=["replace_paragraph", "delete_paragraph", "add_paragraph"],
        preferred_tool="replace_paragraph",
        ambiguity_reason="'Fix flow' could mean rewrite, remove, or reorder",
    ),
    AmbiguousScenario(
        query="Add a transition between chapters 2 and 3",
        valid_tools=["add_paragraph", "add_paragraph"],
        preferred_tool="add_paragraph",
        ambiguity_reason="Could add to end of ch2 or beginning of ch3",
    ),
]


# =============================================================================
# LEVEL 5: EDGE CASE SCENARIOS (Boundary conditions, special characters)
# =============================================================================
# NOTE: Edge cases test agent behavior in unusual situations. Empty document
# scenarios may cause agents to perform multiple read operations. The benchmark
# runner has a 2-minute timeout per scenario as a safeguard.


@dataclass
class EdgeCaseScenario:
    """A scenario testing edge cases and boundary conditions."""

    query: str
    operation: str  # Semantic operation that maps to tool names via ToolSet.get_expected_tool()
    category: str
    edge_case_type: str  # empty, boundary, special_chars, unicode, large_content
    expected_behavior: str  # What should happen (success, error, specific handling)

    def get_expected_tool(self, tool_set_name: str = "default") -> str:
        """Get expected tool for a specific tool set."""
        from .tools import get_tool_set

        tool_set = get_tool_set(tool_set_name)
        return tool_set.get_expected_tool(self.operation)


# Edge case contexts
EMPTY_DOC_CONTEXT = "Document 'empty_doc' has chapter '01-empty.md' with no paragraphs (completely empty). "

SINGLE_PARAGRAPH_CONTEXT = (
    "Document 'minimal' has chapter '01-single.md' with exactly 1 paragraph: "
    "[0: 'The only paragraph in this chapter.']. "
)

UNICODE_CONTEXT = (
    "Document 'international' has chapter '01-unicode.md' with paragraphs: "
    "[0: '日本語テスト', 1: 'Ελληνικά κείμενο', 2: '中文内容', 3: 'العربية']. "
)

SPECIAL_CHARS_CONTEXT = (
    "Document 'special' has chapter '01-special.md' with paragraphs: "
    "[0: 'Normal text', 1: 'Text with **markdown** and `code`', "
    "2: 'Text with <html> tags', 3: 'Text with \"quotes\" and 'apostrophes'']. "
)

EDGE_CASE_SCENARIOS = [
    # Empty document operations
    EdgeCaseScenario(
        query=f"{EMPTY_DOC_CONTEXT}Add the first paragraph 'Hello World' to this empty chapter",
        operation="append",
        category="empty_document",
        edge_case_type="empty",
        expected_behavior="success - creates first paragraph",
    ),
    EdgeCaseScenario(
        query=f"{EMPTY_DOC_CONTEXT}Delete paragraph 0 from the empty chapter",
        operation="delete",
        category="empty_document",
        edge_case_type="empty",
        expected_behavior="error - no paragraphs to delete",
    ),
    EdgeCaseScenario(
        query=f"{EMPTY_DOC_CONTEXT}Replace paragraph 0 with new content",
        operation="replace",
        category="empty_document",
        edge_case_type="empty",
        expected_behavior="error - no paragraph at index 0",
    ),
    # Single paragraph boundary operations
    EdgeCaseScenario(
        query=f"{SINGLE_PARAGRAPH_CONTEXT}Insert text before the only paragraph",
        operation="insert_before",
        category="boundary",
        edge_case_type="boundary",
        expected_behavior="success - inserts at position 0",
    ),
    EdgeCaseScenario(
        query=f"{SINGLE_PARAGRAPH_CONTEXT}Move paragraph 0 to the end",
        operation="move_to_end",
        category="boundary",
        edge_case_type="boundary",
        expected_behavior="success - no-op, already at end",
    ),
    EdgeCaseScenario(
        query=f"{SINGLE_PARAGRAPH_CONTEXT}Move paragraph 0 before paragraph 5",
        operation="move_before",
        category="boundary",
        edge_case_type="boundary",
        expected_behavior="error - target index 5 out of bounds",
    ),
    # Unicode content handling
    EdgeCaseScenario(
        query=f"{UNICODE_CONTEXT}Replace paragraph 0 with 'こんにちは世界'",
        operation="replace",
        category="unicode",
        edge_case_type="unicode",
        expected_behavior="success - handles Japanese characters",
    ),
    EdgeCaseScenario(
        query=f"{UNICODE_CONTEXT}Insert 'Привет мир' after the Arabic paragraph",
        operation="insert_after",
        category="unicode",
        edge_case_type="unicode",
        expected_behavior="success - handles Cyrillic after Arabic",
    ),
    EdgeCaseScenario(
        query=f"{UNICODE_CONTEXT}Add emoji content 🎉🚀✨ at the end",
        operation="append",
        category="unicode",
        edge_case_type="unicode",
        expected_behavior="success - handles emoji characters",
    ),
    # Special characters handling
    EdgeCaseScenario(
        query=f"{SPECIAL_CHARS_CONTEXT}Replace paragraph 1 preserving markdown formatting",
        operation="replace",
        category="special_chars",
        edge_case_type="special_chars",
        expected_behavior="success - preserves markdown",
    ),
    EdgeCaseScenario(
        query=f"{SPECIAL_CHARS_CONTEXT}Delete the paragraph with HTML tags",
        operation="delete",
        category="special_chars",
        edge_case_type="special_chars",
        expected_behavior="success - removes paragraph 2",
    ),
    # First/last paragraph operations
    EdgeCaseScenario(
        query=f"{DOC_CONTEXT}Insert new content before the very first paragraph (index 0)",
        operation="insert_before",
        category="boundary",
        edge_case_type="boundary",
        expected_behavior="success - inserts at beginning",
    ),
    EdgeCaseScenario(
        query=f"{DOC_CONTEXT}Delete the last paragraph (index 3)",
        operation="delete",
        category="boundary",
        edge_case_type="boundary",
        expected_behavior="success - removes last paragraph",
    ),
]


# =============================================================================
# LEVEL 6: ADVERSARIAL SCENARIOS (Confusing, contradictory, malformed)
# =============================================================================


@dataclass
class AdversarialScenario:
    """A scenario designed to confuse or trick the LLM."""

    query: str
    operation: str  # Semantic operation that maps to tool names via ToolSet.get_expected_tool()
    category: str
    adversarial_type: str  # confusing_terms, contradictory, negative_index, out_of_bounds
    trap_description: str  # Why this might confuse the LLM

    def get_expected_tool(self, tool_set_name: str = "default") -> str:
        """Get expected tool for a specific tool set."""
        from .tools import get_tool_set

        tool_set = get_tool_set(tool_set_name)
        return tool_set.get_expected_tool(self.operation)


ADVERSARIAL_SCENARIOS = [
    # Confusing similar terms
    AdversarialScenario(
        query=f"{DOC_CONTEXT}Put new text ABOVE paragraph 2 (not below, I said above!)",
        operation="insert_before",
        category="confusing_terms",
        adversarial_type="confusing_terms",
        trap_description="Emphasizes 'above' to test if LLM correctly maps to 'before'",
    ),
    AdversarialScenario(
        query=f"{DOC_CONTEXT}Add text UNDERNEATH paragraph 1 (below it, after it)",
        operation="insert_after",
        category="confusing_terms",
        adversarial_type="confusing_terms",
        trap_description="Uses 'underneath' which is less common than 'below/after'",
    ),
    AdversarialScenario(
        query=f"{DOC_CONTEXT}I want to INSERT something FOLLOWING paragraph 2",
        operation="insert_after",
        category="confusing_terms",
        adversarial_type="confusing_terms",
        trap_description="Uses 'following' instead of 'after'",
    ),
    AdversarialScenario(
        query=f"{DOC_CONTEXT}Place text PRECEDING paragraph 3",
        operation="insert_before",
        category="confusing_terms",
        adversarial_type="confusing_terms",
        trap_description="Uses 'preceding' instead of 'before'",
    ),
    # Contradictory instructions
    AdversarialScenario(
        query=f"{DOC_CONTEXT}Delete paragraph 2 by replacing it with empty content",
        operation="delete",
        category="contradictory",
        adversarial_type="contradictory",
        trap_description="Says 'delete' but also mentions 'replacing' - should prioritize delete",
    ),
    AdversarialScenario(
        query=f"{DOC_CONTEXT}Append new content at position 1 (at the end)",
        operation="append",
        category="contradictory",
        adversarial_type="contradictory",
        trap_description="Mentions position 1 but says 'at the end' - append takes no index",
    ),
    AdversarialScenario(
        query=f"{DOC_CONTEXT}Move paragraph 3 to before paragraph 3",
        operation="move_before",
        category="contradictory",
        adversarial_type="contradictory",
        trap_description="Move to same position - should handle gracefully (no-op)",
    ),
    # Negative and out-of-bounds indices
    AdversarialScenario(
        query=f"{DOC_CONTEXT}Delete paragraph -1 (the last one)",
        operation="delete",
        category="negative_index",
        adversarial_type="negative_index",
        trap_description="Uses Python-style negative index - should interpret as last",
    ),
    AdversarialScenario(
        query=f"{DOC_CONTEXT}Replace paragraph 100 with new text",
        operation="replace",
        category="out_of_bounds",
        adversarial_type="out_of_bounds",
        trap_description="Index 100 is far out of bounds for a 4-paragraph document",
    ),
    AdversarialScenario(
        query=f"{DOC_CONTEXT}Insert before paragraph 999",
        operation="insert_before",
        category="out_of_bounds",
        adversarial_type="out_of_bounds",
        trap_description="Extremely out-of-bounds target index",
    ),
    # Double negatives and complex phrasing
    AdversarialScenario(
        query=f"{DOC_CONTEXT}Don't NOT add text after paragraph 2",
        operation="insert_after",
        category="double_negative",
        adversarial_type="contradictory",
        trap_description="Double negative that resolves to 'add after paragraph 2'",
    ),
    AdversarialScenario(
        query=f"{DOC_CONTEXT}Remove... no wait, replace paragraph 1 with 'New text'",
        operation="replace",
        category="correction",
        adversarial_type="contradictory",
        trap_description="User corrects themselves mid-sentence",
    ),
    # Ambiguous scope references
    AdversarialScenario(
        query=f"{DOC_CONTEXT}Delete the paragraph (you know which one I mean)",
        operation="delete",
        category="ambiguous_reference",
        adversarial_type="confusing_terms",
        trap_description="No specific index given - LLM should ask for clarification or pick reasonable default",
    ),
    AdversarialScenario(
        query=f"{DOC_CONTEXT}Move that paragraph to the other place",
        operation="move_before",
        category="ambiguous_reference",
        adversarial_type="confusing_terms",
        trap_description="Completely vague references - tests if LLM handles ambiguity",
    ),
    # Mixed operation requests
    AdversarialScenario(
        query=f"{DOC_CONTEXT}Either delete paragraph 2 or replace it, whichever is easier",
        operation="delete",
        category="multiple_options",
        adversarial_type="contradictory",
        trap_description="Offers two valid options - LLM must pick one",
    ),
]


# =============================================================================
# COMPREHENSIVE TOOL SCENARIOS (All 28 MCP Tools)
# =============================================================================
# Scenarios are imported from the tool registry at tool_descriptions.py.
# The ToolScenario class is kept for backward compatibility.


@dataclass
class ToolScenario:
    """A scenario for any MCP tool with direct tool name mapping."""

    query: str
    tool: str  # Direct tool name (no translation needed)
    category: str
    description: str = ""


def _build_all_tool_scenarios() -> list[ToolScenario]:
    """Build ALL_TOOL_SCENARIOS from tool registry."""
    scenarios = _get_tool_scenarios()
    return [
        ToolScenario(
            query=s["query"],
            tool=s["expected_tool"],
            category=s["category"],
            description=s.get("description", ""),
        )
        for s in scenarios
    ]


ALL_TOOL_SCENARIOS = _build_all_tool_scenarios()


def get_all_tools_trainset() -> list[dict]:
    """Get comprehensive training set covering all 28 MCP tools.

    Returns:
        List of dicts with 'query', 'expected_tool', 'category', and 'description'
    """
    return _get_tool_scenarios()


def get_all_tools_stats() -> dict:
    """Get statistics about all-tools coverage."""
    scenarios = _get_tool_scenarios()
    tools = {s["expected_tool"] for s in scenarios}
    categories = {s["category"] for s in scenarios}
    return {
        "total_scenarios": len(scenarios),
        "unique_tools": len(tools),
        "categories": sorted(categories),
        "tools_covered": sorted(tools),
    }


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def get_scenarios(
    category: str | None = None,
    level: ComplexityLevel | None = None,
) -> list[Scenario]:
    """Get scenarios, optionally filtered by category and/or level."""
    scenarios = PARAGRAPH_SCENARIOS + COMPLEX_SCENARIOS
    if category is not None:
        scenarios = [s for s in scenarios if s.category == category]
    if level is not None:
        scenarios = [s for s in scenarios if s.level == level]
    return scenarios


def get_sequential_scenarios() -> list[SequentialScenario]:
    """Get all sequential (Level 2) scenarios."""
    return SEQUENTIAL_SCENARIOS


def get_ambiguous_scenarios() -> list[AmbiguousScenario]:
    """Get all ambiguous (Level 4) scenarios."""
    return AMBIGUOUS_SCENARIOS


def get_edge_case_scenarios() -> list[EdgeCaseScenario]:
    """Get all edge case (Level 5) scenarios."""
    return EDGE_CASE_SCENARIOS


def get_adversarial_scenarios() -> list[AdversarialScenario]:
    """Get all adversarial (Level 6) scenarios."""
    return ADVERSARIAL_SCENARIOS


def get_all_scenarios_by_level() -> dict[str, list]:
    """Get all scenarios organized by complexity level."""
    return {
        "level_1_simple": PARAGRAPH_SCENARIOS,
        "level_2_sequential": SEQUENTIAL_SCENARIOS,
        "level_3_complex": COMPLEX_SCENARIOS,
        "level_4_ambiguous": AMBIGUOUS_SCENARIOS,
        "level_5_edge_case": EDGE_CASE_SCENARIOS,
        "level_6_adversarial": ADVERSARIAL_SCENARIOS,
    }


def get_dspy_trainset(
    level: ComplexityLevel | None = None,
    tool_set_name: str = "default",
    include_edge_cases: bool = False,
    include_adversarial: bool = False,
) -> list[dict]:
    """Export scenarios for DSPy optimization.

    Args:
        level: Complexity level filter (None = Level 1 only for backward compat)
        tool_set_name: Which tool set to get expected tools for
        include_edge_cases: Include Level 5 edge case scenarios
        include_adversarial: Include Level 6 adversarial scenarios

    Returns:
        List of dicts with 'query', 'expected_tool', 'operation', and 'level' keys
    """
    # Start with base scenarios
    if level is None or level == ComplexityLevel.SIMPLE:
        base_scenarios = PARAGRAPH_SCENARIOS
    elif level == ComplexityLevel.COMPLEX:
        base_scenarios = PARAGRAPH_SCENARIOS + COMPLEX_SCENARIOS
    else:
        base_scenarios = PARAGRAPH_SCENARIOS

    results = [
        {
            "query": s.query,  # Use full query with document context for consistency
            "expected_tool": s.get_expected_tool(tool_set_name),
            "operation": s.operation,
            "level": s.level.value,
        }
        for s in base_scenarios
    ]

    # Add edge case scenarios if requested
    if include_edge_cases:
        for s in EDGE_CASE_SCENARIOS:
            results.append(
                {
                    "query": s.query,
                    "expected_tool": s.get_expected_tool(tool_set_name),
                    "operation": s.operation,
                    "level": 5,
                }
            )

    # Add adversarial scenarios if requested
    if include_adversarial:
        for s in ADVERSARIAL_SCENARIOS:
            results.append(
                {
                    "query": s.query,
                    "expected_tool": s.get_expected_tool(tool_set_name),
                    "operation": s.operation,
                    "level": 6,
                }
            )

    return results


def get_scenario_stats() -> dict:
    """Get statistics about available scenarios."""
    return {
        "level_1_simple": len(PARAGRAPH_SCENARIOS),
        "level_2_sequential": len(SEQUENTIAL_SCENARIOS),
        "level_3_complex": len(COMPLEX_SCENARIOS),
        "level_4_ambiguous": len(AMBIGUOUS_SCENARIOS),
        "level_5_edge_case": len(EDGE_CASE_SCENARIOS),
        "level_6_adversarial": len(ADVERSARIAL_SCENARIOS),
        "total": (
            len(PARAGRAPH_SCENARIOS)
            + len(SEQUENTIAL_SCENARIOS)
            + len(COMPLEX_SCENARIOS)
            + len(AMBIGUOUS_SCENARIOS)
            + len(EDGE_CASE_SCENARIOS)
            + len(ADVERSARIAL_SCENARIOS)
        ),
    }


def get_benchmark_scenarios(
    include_edge_cases: bool = False,
    include_adversarial: bool = False,
    tool_set_name: str = "default",
) -> list[dict]:
    """Get scenarios formatted for benchmark testing.

    Args:
        include_edge_cases: Include Level 5 edge case scenarios
        include_adversarial: Include Level 6 adversarial scenarios
        tool_set_name: Which tool set to get expected tools for (default, 4-tool, 8-tool, 2-tool)

    Returns:
        List of dicts with query, expected_tool, operation, level, and category
    """
    scenarios = []

    # Level 1: Simple paragraph scenarios
    for s in PARAGRAPH_SCENARIOS:
        scenarios.append(
            {
                "query": s.query,
                "expected_tool": s.get_expected_tool(tool_set_name),
                "operation": s.operation,
                "level": 1,
                "category": s.category,
            }
        )

    # Level 5: Edge case scenarios
    if include_edge_cases:
        for s in EDGE_CASE_SCENARIOS:
            scenarios.append(
                {
                    "query": s.query,
                    "expected_tool": s.get_expected_tool(tool_set_name),
                    "operation": s.operation,
                    "level": 5,
                    "category": f"edge_{s.edge_case_type}",
                }
            )

    # Level 6: Adversarial scenarios
    if include_adversarial:
        for s in ADVERSARIAL_SCENARIOS:
            scenarios.append(
                {
                    "query": s.query,
                    "expected_tool": s.get_expected_tool(tool_set_name),
                    "operation": s.operation,
                    "level": 6,
                    "category": f"adversarial_{s.adversarial_type}",
                }
            )

    return scenarios


def get_benchmark_scenarios_for_tool_comparison(
    tool_set_names: list[str],
    include_edge_cases: bool = False,
    include_adversarial: bool = False,
) -> dict[str, list[dict]]:
    """Get scenarios for comparing multiple tool sets.

    Args:
        tool_set_names: List of tool set names to compare (e.g., ['4-tool', '8-tool', '2-tool'])
        include_edge_cases: Include Level 5 edge case scenarios
        include_adversarial: Include Level 6 adversarial scenarios

    Returns:
        Dict mapping tool set name to list of scenarios with expected tools for that set
    """
    return {
        name: get_benchmark_scenarios(
            include_edge_cases=include_edge_cases,
            include_adversarial=include_adversarial,
            tool_set_name=name,
        )
        for name in tool_set_names
    }
