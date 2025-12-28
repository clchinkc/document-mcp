"""Tool Selection Benchmark Tests for Document MCP.

This module benchmarks agent tool selection accuracy across all tool categories,
with particular focus on paragraph tools to inform consolidation decisions.

Metrics collected:
- Tool selection accuracy by category
- Error rate (wrong tool, wrong parameters)
- Token usage per operation type
- Round-trip count for multi-step workflows
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any

import pytest


@dataclass
class ToolSelectionScenario:
    """A scenario for testing tool selection."""

    query: str
    expected_tool: str
    category: str
    description: str = ""
    expected_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolSelectionResult:
    """Result of a tool selection test."""

    scenario: ToolSelectionScenario
    selected_tool: str | None
    correct: bool
    error_message: str | None = None
    token_usage: int = 0
    execution_time: float = 0.0


@dataclass
class BenchmarkResults:
    """Aggregated benchmark results."""

    results: list[ToolSelectionResult]

    @property
    def total_scenarios(self) -> int:
        return len(self.results)

    @property
    def correct_selections(self) -> int:
        return sum(1 for r in self.results if r.correct)

    @property
    def accuracy(self) -> float:
        if not self.results:
            return 0.0
        return self.correct_selections / self.total_scenarios

    @property
    def error_rate(self) -> float:
        return 1.0 - self.accuracy

    @property
    def total_tokens(self) -> int:
        return sum(r.token_usage for r in self.results)

    @property
    def average_tokens(self) -> float:
        if not self.results:
            return 0.0
        return self.total_tokens / self.total_scenarios

    def by_category(self) -> dict[str, BenchmarkResults]:
        """Group results by category."""
        categories: dict[str, list[ToolSelectionResult]] = {}
        for r in self.results:
            cat = r.scenario.category
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(r)
        return {cat: BenchmarkResults(results) for cat, results in categories.items()}

    def report(self) -> str:
        """Generate a human-readable report."""
        lines = [
            "=" * 60,
            "TOOL SELECTION BENCHMARK RESULTS",
            "=" * 60,
            f"Total Scenarios: {self.total_scenarios}",
            f"Correct Selections: {self.correct_selections}",
            f"Accuracy: {self.accuracy:.1%}",
            f"Error Rate: {self.error_rate:.1%}",
            f"Total Tokens: {self.total_tokens}",
            f"Average Tokens/Scenario: {self.average_tokens:.1f}",
            "",
            "By Category:",
        ]

        for category, cat_results in self.by_category().items():
            lines.append(f"  {category}:")
            lines.append(
                f"    Accuracy: {cat_results.accuracy:.1%} ({cat_results.correct_selections}/{cat_results.total_scenarios})"
            )
            lines.append(f"    Avg Tokens: {cat_results.average_tokens:.1f}")

        # List incorrect selections
        incorrect = [r for r in self.results if not r.correct]
        if incorrect:
            lines.extend(["", "Incorrect Selections:"])
            for r in incorrect:
                lines.append(f"  - Query: {r.scenario.query[:50]}...")
                lines.append(f"    Expected: {r.scenario.expected_tool}, Got: {r.selected_tool}")
                if r.error_message:
                    lines.append(f"    Error: {r.error_message}")

        lines.append("=" * 60)
        return "\n".join(lines)


# Tool Selection Scenarios organized by category
# Note: Using consolidated 4-tool API names (add_paragraph, move_paragraph)
# The actual MCP server uses add_paragraph with position='before'/'after'/'end'
# and move_paragraph with destination='before'/'after'
PARAGRAPH_TOOL_SCENARIOS = [
    ToolSelectionScenario(
        query="In document 'novel', chapter 01-intro.md (which has 3 paragraphs), insert a new paragraph with text 'New content' after paragraph 2",
        expected_tool="add_paragraph",
        category="Paragraph Operations",
        description="Insert after specific position",
    ),
    ToolSelectionScenario(
        query="In document 'novel', chapter 02-middle.md (which exists), insert a paragraph with text 'Opening' before the first paragraph",
        expected_tool="add_paragraph",
        category="Paragraph Operations",
        description="Insert before specific position",
    ),
    ToolSelectionScenario(
        query="Replace the third paragraph in chapter 01-intro.md with new content",
        expected_tool="replace_paragraph",
        category="Paragraph Operations",
        description="Replace at specific index",
    ),
    ToolSelectionScenario(
        query="Delete paragraph 4 from document novel, chapter 03-end.md",
        expected_tool="delete_paragraph",
        category="Paragraph Operations",
        description="Delete at specific index",
    ),
    ToolSelectionScenario(
        query="Add a new paragraph at the end of chapter 02-middle.md in novel",
        expected_tool="add_paragraph",
        category="Paragraph Operations",
        description="Append to chapter end",
    ),
    ToolSelectionScenario(
        query="Move paragraph 5 to before paragraph 2 in novel/01-intro.md",
        expected_tool="move_paragraph",
        category="Paragraph Operations",
        description="Move to specific position",
    ),
    ToolSelectionScenario(
        query="Move paragraph 1 to the end of the chapter in document my_book, chapter 02.md",
        expected_tool="move_paragraph",
        category="Paragraph Operations",
        description="Move to chapter end",
    ),
    ToolSelectionScenario(
        query="Continue the story by adding this text at the end of chapter 01.md",
        expected_tool="add_paragraph",
        category="Paragraph Operations",
        description="Natural append language",
    ),
]

CONTENT_TOOL_SCENARIOS = [
    ToolSelectionScenario(
        query="Read the entire document called 'novel'",
        expected_tool="read_content",
        category="Content Access",
        description="Read full document",
    ),
    ToolSelectionScenario(
        query="Show me chapter 02-middle.md from document novel",
        expected_tool="read_content",
        category="Content Access",
        description="Read specific chapter",
    ),
    ToolSelectionScenario(
        query="Find all mentions of 'Marcus' in the novel",
        expected_tool="find_text",
        category="Content Access",
        description="Text search",
    ),
    ToolSelectionScenario(
        query="Replace 'John' with 'James' throughout the document",
        expected_tool="replace_text",
        category="Content Access",
        description="Find and replace",
    ),
    ToolSelectionScenario(
        query="How many words are in my novel?",
        expected_tool="get_statistics",
        category="Content Access",
        description="Word count request",
    ),
    ToolSelectionScenario(
        query="Find content similar to 'the hero's journey' in my book",
        expected_tool="find_similar_text",
        category="Content Access",
        description="Semantic search",
    ),
]

DOCUMENT_TOOL_SCENARIOS = [
    ToolSelectionScenario(
        query="Create a new document called 'my_story'",
        expected_tool="create_document",
        category="Document Management",
        description="Create document",
    ),
    ToolSelectionScenario(
        query="What documents do I have?",
        expected_tool="list_documents",
        category="Document Management",
        description="List documents",
    ),
    ToolSelectionScenario(
        query="Delete the document called 'old_draft'",
        expected_tool="delete_document",
        category="Document Management",
        description="Delete document",
    ),
    ToolSelectionScenario(
        query="Show me the summary of my novel",
        expected_tool="read_summary",
        category="Document Management",
        description="Read summary",
    ),
    ToolSelectionScenario(
        query="Write a summary for my novel: 'A tale of two cities...'",
        expected_tool="write_summary",
        category="Document Management",
        description="Write summary",
    ),
    ToolSelectionScenario(
        query="What summaries are available for my novel?",
        expected_tool="list_summaries",
        category="Document Management",
        description="List summaries",
    ),
]

CHAPTER_TOOL_SCENARIOS = [
    ToolSelectionScenario(
        query="Create a new chapter called '04-conclusion.md' in document novel",
        expected_tool="create_chapter",
        category="Chapter Management",
        description="Create chapter",
    ),
    ToolSelectionScenario(
        query="What chapters are in my novel?",
        expected_tool="list_chapters",
        category="Chapter Management",
        description="List chapters",
    ),
    ToolSelectionScenario(
        query="Delete chapter 02-old.md from novel",
        expected_tool="delete_chapter",
        category="Chapter Management",
        description="Delete chapter",
    ),
    ToolSelectionScenario(
        query="Rewrite the entire chapter 01-intro.md with this new content",
        expected_tool="write_chapter_content",
        category="Chapter Management",
        description="Overwrite chapter",
    ),
]

METADATA_TOOL_SCENARIOS = [
    ToolSelectionScenario(
        query="Set the status of chapter 01-intro.md to 'revised'",
        expected_tool="write_metadata",
        category="Metadata Management",
        description="Write chapter metadata",
    ),
    ToolSelectionScenario(
        query="What's the status of chapter 02.md?",
        expected_tool="read_metadata",
        category="Metadata Management",
        description="Read chapter metadata",
    ),
    ToolSelectionScenario(
        query="Show me all draft chapters",
        expected_tool="list_metadata",
        category="Metadata Management",
        description="Filter by metadata",
    ),
    ToolSelectionScenario(
        query="Where is Marcus mentioned in the document?",
        expected_tool="find_entity",
        category="Metadata Management",
        description="Entity search",
    ),
    ToolSelectionScenario(
        query="Show me the outline of my novel",
        expected_tool="get_document_outline",
        category="Metadata Management",
        description="Document outline",
    ),
]

VERSION_CONTROL_SCENARIOS = [
    ToolSelectionScenario(
        query="Create a backup of my novel before I make changes",
        expected_tool="manage_snapshots",
        category="Version Control",
        description="Create snapshot",
    ),
    ToolSelectionScenario(
        query="List all versions of my document",
        expected_tool="manage_snapshots",
        category="Version Control",
        description="List snapshots",
    ),
    ToolSelectionScenario(
        query="Restore my novel to the previous version",
        expected_tool="manage_snapshots",
        category="Version Control",
        description="Restore snapshot",
    ),
    ToolSelectionScenario(
        query="Has chapter 01-intro.md been modified recently?",
        expected_tool="check_content_status",
        category="Version Control",
        description="Check status",
    ),
    ToolSelectionScenario(
        query="What changed in chapter 01.md since the last backup?",
        expected_tool="diff_content",
        category="Version Control",
        description="Compare versions",
    ),
]

DISCOVERY_TOOL_SCENARIOS = [
    ToolSelectionScenario(
        query="What tools can I use to edit paragraphs?",
        expected_tool="search_tool",
        category="Discovery",
        description="Search for tools",
    ),
    ToolSelectionScenario(
        query="Show me available metadata operations",
        expected_tool="search_tool",
        category="Discovery",
        description="Category search",
    ),
]

# All scenarios combined
ALL_SCENARIOS = (
    PARAGRAPH_TOOL_SCENARIOS
    + CONTENT_TOOL_SCENARIOS
    + DOCUMENT_TOOL_SCENARIOS
    + CHAPTER_TOOL_SCENARIOS
    + METADATA_TOOL_SCENARIOS
    + VERSION_CONTROL_SCENARIOS
    + DISCOVERY_TOOL_SCENARIOS
)


def check_api_key_available() -> bool:
    """Check if a real API key is available for benchmark testing."""
    api_keys = ["OPENROUTER_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY"]
    for key in api_keys:
        value = os.environ.get(key, "").strip()
        if value and value != "test_key" and not value.startswith("sk-test"):
            return True
    return False


@pytest.fixture
def test_docs_root():
    """Provide clean temporary directory for document storage."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


class TestToolSelectionScenarios:
    """Test that scenarios are properly defined."""

    def test_all_scenarios_have_required_fields(self):
        """Verify all scenarios have required fields."""
        for scenario in ALL_SCENARIOS:
            assert scenario.query, "Scenario must have a query"
            assert scenario.expected_tool, "Scenario must have expected_tool"
            assert scenario.category, "Scenario must have category"

    def test_paragraph_scenarios_coverage(self):
        """Verify paragraph tool scenarios cover all 4 consolidated tools."""
        # Using the consolidated 4-tool API:
        # - add_paragraph: for before/after/end insertion
        # - replace_paragraph: for content replacement
        # - delete_paragraph: for deletion
        # - move_paragraph: for before/after repositioning
        expected_tools = {
            "add_paragraph",
            "replace_paragraph",
            "delete_paragraph",
            "move_paragraph",
        }

        covered_tools = {s.expected_tool for s in PARAGRAPH_TOOL_SCENARIOS}

        # Note: read_paragraph is handled by read_content with scope="paragraph"
        missing = expected_tools - covered_tools
        assert not missing, f"Missing scenarios for tools: {missing}"

    def test_category_distribution(self):
        """Verify scenarios are distributed across categories."""
        categories = {s.category for s in ALL_SCENARIOS}
        expected_categories = {
            "Paragraph Operations",
            "Content Access",
            "Document Management",
            "Chapter Management",
            "Metadata Management",
            "Version Control",
            "Discovery",
        }

        missing = expected_categories - categories
        assert not missing, f"Missing categories: {missing}"

    def test_scenario_count_by_category(self):
        """Report scenario count by category."""
        category_counts: dict[str, int] = {}
        for s in ALL_SCENARIOS:
            category_counts[s.category] = category_counts.get(s.category, 0) + 1

        print("\nScenario Distribution:")
        for cat, count in sorted(category_counts.items()):
            print(f"  {cat}: {count}")

        # Verify minimum coverage (8 scenarios covering all 4 paragraph tools)
        assert category_counts.get("Paragraph Operations", 0) >= 4, "Need more paragraph scenarios"

    def test_all_28_tools_covered(self):
        """Verify all 28 MCP tools have at least one benchmark scenario."""
        # Complete list of all 28 MCP tools
        all_mcp_tools = {
            # Document management (3)
            "list_documents",
            "create_document",
            "delete_document",
            # Summary tools (3)
            "read_summary",
            "write_summary",
            "list_summaries",
            # Chapter management (4)
            "list_chapters",
            "create_chapter",
            "delete_chapter",
            "write_chapter_content",
            # Paragraph management (4)
            "add_paragraph",
            "replace_paragraph",
            "delete_paragraph",
            "move_paragraph",
            # Content tools (6)
            "read_content",
            "find_text",
            "replace_text",
            "get_statistics",
            "find_similar_text",
            "find_entity",
            # Metadata tools (3)
            "read_metadata",
            "write_metadata",
            "list_metadata",
            # Safety tools (3)
            "manage_snapshots",
            "check_content_status",
            "diff_content",
            # Overview tools (1)
            "get_document_outline",
            # Discovery tools (1)
            "search_tool",
        }

        covered_tools = {s.expected_tool for s in ALL_SCENARIOS}

        missing = all_mcp_tools - covered_tools
        extra = covered_tools - all_mcp_tools

        assert not missing, f"Missing scenarios for {len(missing)} tools: {sorted(missing)}"
        assert not extra, f"Unknown tools in scenarios: {sorted(extra)}"
        assert len(all_mcp_tools) == 28, f"Expected 28 tools, got {len(all_mcp_tools)}"

        print(f"\n✓ All {len(all_mcp_tools)} MCP tools have benchmark scenarios")


@pytest.mark.evaluation
class TestToolSelectionBenchmark:
    """Benchmark tests for tool selection accuracy.

    These tests require real API keys and make actual LLM calls
    to measure how well agents select the correct tools.
    """

    @pytest.mark.skipif(not check_api_key_available(), reason="Real API key required for benchmark testing")
    @pytest.mark.asyncio
    async def test_simple_agent_paragraph_tool_selection(self, test_docs_root):
        """Benchmark simple agent's paragraph tool selection accuracy."""
        from src.agents.simple_agent.main import initialize_agent_and_mcp_server
        from src.agents.simple_agent.main import process_single_user_query

        os.environ["DOCUMENT_ROOT_DIR"] = str(test_docs_root)

        # Create test document with chapters
        doc_dir = test_docs_root / "novel"
        doc_dir.mkdir(parents=True)
        (doc_dir / "01-intro.md").write_text(
            "# Introduction\n\nParagraph one.\n\nParagraph two.\n\nParagraph three."
        )
        (doc_dir / "02-middle.md").write_text("# Middle\n\nMiddle content.\n\nMore content.")

        results: list[ToolSelectionResult] = []

        agent, mcp_server = await initialize_agent_and_mcp_server()

        async with agent.run_mcp_servers():
            # Test ALL scenarios across ALL categories for comprehensive coverage
            test_scenarios = ALL_SCENARIOS  # Full 37 scenarios across 7 categories
            for scenario in test_scenarios:
                try:
                    response, metrics = await process_single_user_query(
                        agent, scenario.query, collect_metrics=True
                    )

                    # Extract selected tool from metrics.tool_names (primary source)
                    # or from response.details JSON (fallback)
                    selected_tool = None
                    if metrics and metrics.tool_names:
                        # Filter out internal tools like 'final_result'
                        real_tools = [t for t in metrics.tool_names if t != "final_result"]

                        if len(real_tools) == 1:
                            # Single tool called - use it
                            selected_tool = real_tools[0]
                        elif scenario.expected_tool in real_tools:
                            # Multiple tools but expected is among them - count as correct
                            selected_tool = scenario.expected_tool
                        elif real_tools:
                            # Multiple tools, expected not among them - use first one
                            selected_tool = real_tools[0]
                    elif response and response.details:
                        # Fallback: parse JSON details to get tool name as key
                        import json

                        try:
                            details_dict = json.loads(response.details)
                            if isinstance(details_dict, dict) and details_dict:
                                selected_tool = next(iter(details_dict.keys()))
                        except (json.JSONDecodeError, TypeError):
                            pass

                    correct = selected_tool == scenario.expected_tool

                    results.append(
                        ToolSelectionResult(
                            scenario=scenario,
                            selected_tool=selected_tool,
                            correct=correct,
                            token_usage=metrics.token_usage if metrics else 0,
                            execution_time=metrics.execution_time if metrics else 0.0,
                        )
                    )

                except Exception as e:
                    results.append(
                        ToolSelectionResult(
                            scenario=scenario,
                            selected_tool=None,
                            correct=False,
                            error_message=str(e),
                        )
                    )

        benchmark = BenchmarkResults(results)
        print(f"\n{benchmark.report()}")

        # Identify timeout vs wrong-selection failures
        timeouts = [r for r in results if r.selected_tool is None and r.execution_time > 60]
        wrong_selections = [r for r in results if r.selected_tool is not None and not r.correct]

        if timeouts:
            print(f"\n⚠️  TIMEOUT FAILURES ({len(timeouts)}):")
            for t in timeouts:
                print(f"  - {t.scenario.query[:50]}... ({t.execution_time:.1f}s)")

        if wrong_selections:
            print(f"\n❌ WRONG SELECTIONS ({len(wrong_selections)}):")
            for w in wrong_selections:
                print(f"  - Expected: {w.scenario.expected_tool}, Got: {w.selected_tool}")

        # Assertions - evaluation tests should track results, not enforce thresholds
        assert benchmark.total_scenarios > 0, "Should have tested some scenarios"
        # Log accuracy for tracking but don't fail - this is evaluation data
        print(f"\n📊 BENCHMARK ACCURACY: {benchmark.accuracy:.1%}")

    @pytest.mark.skipif(not check_api_key_available(), reason="Real API key required for benchmark testing")
    @pytest.mark.asyncio
    async def test_react_agent_paragraph_tool_selection(self, test_docs_root):
        """Benchmark react agent's paragraph tool selection accuracy."""
        from src.agents.react_agent.main import run_react_agent_with_metrics

        os.environ["DOCUMENT_ROOT_DIR"] = str(test_docs_root)

        # Create test document with chapters
        doc_dir = test_docs_root / "novel"
        doc_dir.mkdir(parents=True)
        (doc_dir / "01-intro.md").write_text(
            "# Introduction\n\nParagraph one.\n\nParagraph two.\n\nParagraph three."
        )

        results: list[ToolSelectionResult] = []

        for scenario in PARAGRAPH_TOOL_SCENARIOS:  # ALL paragraph scenarios
            try:
                history, metrics = await run_react_agent_with_metrics(scenario.query, max_steps=3)

                # Extract selected tool from metrics.tool_names (populated by MetricsCollectionContext)
                selected_tool = None
                if metrics and metrics.tool_names:
                    # Filter out internal tools like 'final_result'
                    real_tools = [t for t in metrics.tool_names if t != "final_result"]

                    if len(real_tools) == 1:
                        # Single tool called - use it
                        selected_tool = real_tools[0]
                    elif scenario.expected_tool in real_tools:
                        # Multiple tools but expected is among them - count as correct
                        selected_tool = scenario.expected_tool
                    elif real_tools:
                        # Multiple tools, expected not among them - use first one
                        selected_tool = real_tools[0]
                elif history:
                    # Fallback: extract from history action strings
                    for step in history:
                        if step.get("action"):
                            action_str = str(step["action"])
                            if "(" in action_str:
                                selected_tool = action_str.split("(")[0].strip()
                                break

                correct = selected_tool == scenario.expected_tool

                results.append(
                    ToolSelectionResult(
                        scenario=scenario,
                        selected_tool=selected_tool,
                        correct=correct,
                        token_usage=metrics.token_usage if metrics else 0,
                        execution_time=metrics.execution_time if metrics else 0.0,
                    )
                )

            except Exception as e:
                results.append(
                    ToolSelectionResult(
                        scenario=scenario,
                        selected_tool=None,
                        correct=False,
                        error_message=str(e),
                    )
                )

        benchmark = BenchmarkResults(results)
        print(f"\n{benchmark.report()}")

        # Identify timeout vs wrong-selection failures
        timeouts = [r for r in results if r.selected_tool is None and r.execution_time > 60]
        if timeouts:
            print(f"\n⚠️  TIMEOUT FAILURES ({len(timeouts)}):")
            for t in timeouts:
                print(f"  - {t.scenario.query[:50]}... ({t.execution_time:.1f}s)")

        assert benchmark.total_scenarios > 0
        print(f"\n📊 REACT BENCHMARK ACCURACY: {benchmark.accuracy:.1%}")


class TestBenchmarkResultsAnalysis:
    """Unit tests for BenchmarkResults analysis."""

    def test_accuracy_calculation(self):
        """Test accuracy calculation."""
        scenario = ToolSelectionScenario(query="test", expected_tool="test_tool", category="test")

        results = [
            ToolSelectionResult(scenario=scenario, selected_tool="test_tool", correct=True),
            ToolSelectionResult(scenario=scenario, selected_tool="test_tool", correct=True),
            ToolSelectionResult(scenario=scenario, selected_tool="wrong_tool", correct=False),
        ]

        benchmark = BenchmarkResults(results)

        assert benchmark.total_scenarios == 3
        assert benchmark.correct_selections == 2
        assert benchmark.accuracy == pytest.approx(0.667, rel=0.01)
        assert benchmark.error_rate == pytest.approx(0.333, rel=0.01)

    def test_by_category_grouping(self):
        """Test grouping results by category."""
        scenarios = [
            ToolSelectionScenario(query="a", expected_tool="t1", category="cat1"),
            ToolSelectionScenario(query="b", expected_tool="t2", category="cat1"),
            ToolSelectionScenario(query="c", expected_tool="t3", category="cat2"),
        ]

        results = [
            ToolSelectionResult(scenario=scenarios[0], selected_tool="t1", correct=True),
            ToolSelectionResult(scenario=scenarios[1], selected_tool="t2", correct=True),
            ToolSelectionResult(scenario=scenarios[2], selected_tool="wrong", correct=False),
        ]

        benchmark = BenchmarkResults(results)
        by_cat = benchmark.by_category()

        assert "cat1" in by_cat
        assert "cat2" in by_cat
        assert by_cat["cat1"].accuracy == 1.0
        assert by_cat["cat2"].accuracy == 0.0

    def test_token_metrics(self):
        """Test token usage metrics."""
        scenario = ToolSelectionScenario(query="test", expected_tool="test_tool", category="test")

        results = [
            ToolSelectionResult(scenario=scenario, selected_tool="t", correct=True, token_usage=100),
            ToolSelectionResult(scenario=scenario, selected_tool="t", correct=True, token_usage=150),
            ToolSelectionResult(scenario=scenario, selected_tool="t", correct=True, token_usage=200),
        ]

        benchmark = BenchmarkResults(results)

        assert benchmark.total_tokens == 450
        assert benchmark.average_tokens == 150.0

    def test_empty_results(self):
        """Test handling of empty results."""
        benchmark = BenchmarkResults([])

        assert benchmark.total_scenarios == 0
        assert benchmark.accuracy == 0.0
        assert benchmark.average_tokens == 0.0

    def test_report_generation(self):
        """Test report generation."""
        scenario = ToolSelectionScenario(
            query="test query", expected_tool="expected_tool", category="Test Category"
        )

        results = [
            ToolSelectionResult(
                scenario=scenario, selected_tool="expected_tool", correct=True, token_usage=100
            ),
            ToolSelectionResult(
                scenario=scenario, selected_tool="wrong_tool", correct=False, token_usage=120
            ),
        ]

        benchmark = BenchmarkResults(results)
        report = benchmark.report()

        assert "TOOL SELECTION BENCHMARK RESULTS" in report
        assert "Accuracy: 50.0%" in report
        assert "Test Category" in report
        assert "Incorrect Selections:" in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
