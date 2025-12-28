"""Benchmark Runner for Tool Selection Comparison.

This module runs benchmarks to test how well different tool configurations
help LLMs select the correct paragraph tools.

Configuration is file-based:
- benchmarks/tool_sets/   - Tool set definitions (4-tool.yaml, 8-tool.yaml, etc.)
- benchmarks/descriptions/ - Description styles (default.yaml, minimal.yaml, verbose.yaml)
- benchmarks/models/       - Model configs (gpt-5-mini.yaml, claude-4.5-haiku.yaml, etc.)

To compare configurations, run the benchmark twice with different settings
and compare the results yourself.

Usage:
    python -m benchmarks.runner --list              # List available configs
    python -m benchmarks.runner --tool-set 4-tool   # Run with 4-tool
    python -m benchmarks.runner --tool-set 8-tool   # Run with 8-tool
    python -m benchmarks.runner --model gpt-5-mini  # Run with specific model

Adding new configurations:
    Create a new YAML file in the appropriate directory.
    The benchmark will auto-detect it by filename.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from pathlib import Path
from typing import Any

from .config_loader import get_model_from_config
from .config_loader import get_tool_set_from_config
from .config_loader import print_available_configs
from .metrics import BenchmarkMetrics
from .scenarios import get_benchmark_scenarios
from .scenarios import get_scenario_stats


@dataclass
class BenchmarkResult:
    """Result of a single benchmark scenario."""

    query: str
    expected_tool: str
    selected_tool: str | None
    correct: bool
    level: int
    category: str
    input_tokens: int = 0
    output_tokens: int = 0
    execution_time: float = 0.0
    error: str | None = None


@dataclass
class BenchmarkReport:
    """Complete benchmark report."""

    tool_set_name: str  # Can be tool implementation (4-tool, 8-tool) or description style
    results: list[BenchmarkResult]
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None

    @property
    def total_scenarios(self) -> int:
        return len(self.results)

    @property
    def correct_count(self) -> int:
        return sum(1 for r in self.results if r.correct)

    @property
    def accuracy(self) -> float:
        return self.correct_count / self.total_scenarios if self.results else 0.0

    @property
    def total_input_tokens(self) -> int:
        return sum(r.input_tokens for r in self.results)

    @property
    def total_output_tokens(self) -> int:
        return sum(r.output_tokens for r in self.results)

    @property
    def avg_input_tokens(self) -> float:
        return self.total_input_tokens / self.total_scenarios if self.results else 0.0

    @property
    def avg_output_tokens(self) -> float:
        return self.total_output_tokens / self.total_scenarios if self.results else 0.0

    def by_level(self) -> dict[int, list[BenchmarkResult]]:
        """Group results by complexity level."""
        levels: dict[int, list[BenchmarkResult]] = {}
        for r in self.results:
            levels.setdefault(r.level, []).append(r)
        return levels

    def by_category(self) -> dict[str, list[BenchmarkResult]]:
        """Group results by category."""
        categories: dict[str, list[BenchmarkResult]] = {}
        for r in self.results:
            categories.setdefault(r.category, []).append(r)
        return categories

    def to_metrics(self) -> BenchmarkMetrics:
        """Convert to BenchmarkMetrics for comparison."""
        metrics = BenchmarkMetrics(
            accuracy=self.accuracy,
            input_tokens=self.total_input_tokens,
            output_tokens=self.total_output_tokens,
            error_count=sum(1 for r in self.results if r.error),
            num_examples=self.total_scenarios,
        )
        for r in self.results:
            if r.selected_tool:
                metrics.add_confusion(r.expected_tool, r.selected_tool)
        return metrics

    def report(self) -> str:
        """Generate human-readable report."""
        lines = [
            "=" * 70,
            f"BENCHMARK REPORT: {self.tool_set_name.upper()}",
            "=" * 70,
        ]

        lines.extend(
            [
                f"Total Scenarios: {self.total_scenarios}",
                f"Correct: {self.correct_count} ({self.accuracy:.1%})",
                f"Tokens: {self.avg_input_tokens:.0f} in / {self.avg_output_tokens:.0f} out (avg)",
                "",
                "BY COMPLEXITY LEVEL:",
            ]
        )

        level_names = {
            1: "Simple",
            2: "Sequential",
            3: "Complex",
            4: "Ambiguous",
            5: "Edge Case",
            6: "Adversarial",
        }

        for level, level_results in sorted(self.by_level().items()):
            correct = sum(1 for r in level_results if r.correct)
            total = len(level_results)
            acc = correct / total if total else 0
            name = level_names.get(level, f"Level {level}")
            lines.append(f"  Level {level} ({name}): {acc:.1%} ({correct}/{total})")

        lines.extend(["", "BY CATEGORY:"])
        for cat, cat_results in sorted(self.by_category().items()):
            correct = sum(1 for r in cat_results if r.correct)
            total = len(cat_results)
            acc = correct / total if total else 0
            lines.append(f"  {cat}: {acc:.1%} ({correct}/{total})")

        # Show incorrect selections
        incorrect = [r for r in self.results if not r.correct]
        if incorrect:
            lines.extend(["", "INCORRECT SELECTIONS:"])
            for r in incorrect[:10]:
                query_preview = r.query[:50] + "..." if len(r.query) > 50 else r.query
                lines.append(f"  - {query_preview}")
                lines.append(f"    Expected: {r.expected_tool}, Got: {r.selected_tool}")
                if r.error:
                    lines.append(f"    Error: {r.error}")
            if len(incorrect) > 10:
                lines.append(f"  ... and {len(incorrect) - 10} more")

        lines.append("=" * 70)
        return "\n".join(lines)

    def to_json(self) -> dict:
        """Export to JSON-serializable dict."""
        return {
            "tool_set_name": self.tool_set_name,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "summary": {
                "total_scenarios": self.total_scenarios,
                "correct_count": self.correct_count,
                "accuracy": self.accuracy,
                "avg_input_tokens": self.avg_input_tokens,
                "avg_output_tokens": self.avg_output_tokens,
            },
            "by_level": {
                str(level): {
                    "total": len(results),
                    "correct": sum(1 for r in results if r.correct),
                    "accuracy": sum(1 for r in results if r.correct) / len(results) if results else 0,
                }
                for level, results in self.by_level().items()
            },
            "results": [
                {
                    "query": r.query[:100],
                    "expected_tool": r.expected_tool,
                    "selected_tool": r.selected_tool,
                    "correct": r.correct,
                    "level": r.level,
                    "category": r.category,
                    "input_tokens": r.input_tokens,
                    "output_tokens": r.output_tokens,
                    "error": r.error,
                }
                for r in self.results
            ],
        }


def check_api_key_available() -> bool:
    """Check if API key is available for benchmarking."""
    for key in ["OPENROUTER_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY"]:
        value = os.environ.get(key, "").strip()
        if value and value != "test_key" and not value.startswith("sk-test"):
            return True
    return False


async def run_benchmark(
    tool_set_name: str = "4-tool",
    model_name: str | None = None,
    include_edge_cases: bool = True,
    include_adversarial: bool = True,
    save_results: bool = True,
) -> BenchmarkReport:
    """Run benchmark with a specific tool set.

    Args:
        tool_set_name: Tool set to use (loads from benchmarks/tool_sets/<name>.yaml)
        model_name: Model to use (loads from benchmarks/models/<name>.yaml)
        include_edge_cases: Include Level 5 edge case scenarios
        include_adversarial: Include Level 6 adversarial scenarios
        save_results: Save results to file

    Returns:
        BenchmarkReport with all results
    """
    from src.agents.simple_agent.main import initialize_agent_and_mcp_server
    from src.agents.simple_agent.main import process_single_user_query

    # Load tool set from file-based config
    tool_set = get_tool_set_from_config(tool_set_name)

    # Load model config if specified
    model_config = None
    if model_name:
        model_config = get_model_from_config(model_name)

    # Create temp directory for test documents
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        os.environ["DOCUMENT_ROOT_DIR"] = str(tmp_path)

        # Create test documents matching scenario contexts
        _create_test_documents(tmp_path)

        # Get scenarios with expected tools for this tool set
        all_scenarios = get_benchmark_scenarios(
            include_edge_cases=include_edge_cases,
            include_adversarial=include_adversarial,
            tool_set_name=tool_set_name,
        )

        report = BenchmarkReport(tool_set_name=tool_set_name, results=[])

        print(f"\n{'=' * 60}")
        print(f"Running benchmark: {tool_set_name.upper()} ({len(all_scenarios)} scenarios)")
        print("=" * 60)
        print(f"Tool Set: {tool_set.name} ({tool_set.tool_count} tools)")
        if model_config:
            print(
                f"Model: {model_config.get('name', model_name)} ({model_config.get('provider', 'unknown')})"
            )

        agent, _ = await initialize_agent_and_mcp_server()

        async with agent.run_mcp_servers():
            for i, scenario in enumerate(all_scenarios, 1):
                print(f"[{i}/{len(all_scenarios)}] ", end="", flush=True)

                result = await _run_single_scenario(agent, scenario, process_single_user_query)
                report.results.append(result)

                status = "OK" if result.correct else "WRONG"
                print(f"[{status}] {result.category}")

        report.completed_at = datetime.now()

        # Save results if requested
        if save_results:
            results_dir = Path("benchmarks/results")
            results_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_suffix = f"_{model_name}" if model_name else ""
            results_file = results_dir / f"{tool_set_name}{model_suffix}_{timestamp}.json"

            with open(results_file, "w") as f:
                json.dump(report.to_json(), f, indent=2)
            print(f"\nResults saved to: {results_file}")

        return report


# Default timeout per scenario (seconds)
SCENARIO_TIMEOUT = 120  # 2 minutes max per scenario


async def _run_single_scenario(
    agent: Any,
    scenario: dict,
    process_fn: Any,
    timeout: float = SCENARIO_TIMEOUT,
) -> BenchmarkResult:
    """Run a single benchmark scenario with timeout protection."""
    try:
        response, metrics = await asyncio.wait_for(
            process_fn(agent, scenario["query"], collect_metrics=True),
            timeout=timeout,
        )

        # Extract selected tool
        selected_tool = None
        if metrics and metrics.tool_names:
            real_tools = [t for t in metrics.tool_names if t != "final_result"]
            if len(real_tools) == 1:
                selected_tool = real_tools[0]
            elif scenario["expected_tool"] in real_tools:
                selected_tool = scenario["expected_tool"]
            elif real_tools:
                selected_tool = real_tools[0]

        correct = selected_tool == scenario["expected_tool"]

        # Token estimates
        total_tokens = metrics.token_usage if metrics else 0
        input_tokens = int(total_tokens * 0.7)
        output_tokens = int(total_tokens * 0.3)

        return BenchmarkResult(
            query=scenario["query"],
            expected_tool=scenario["expected_tool"],
            selected_tool=selected_tool,
            correct=correct,
            level=scenario.get("level", 1),
            category=scenario.get("category", "unknown"),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            execution_time=metrics.execution_time if metrics else 0.0,
        )

    except asyncio.TimeoutError:
        return BenchmarkResult(
            query=scenario["query"],
            expected_tool=scenario["expected_tool"],
            selected_tool=None,
            correct=False,
            level=scenario.get("level", 1),
            category=scenario.get("category", "unknown"),
            error=f"Timeout after {timeout}s",
        )

    except Exception as e:
        return BenchmarkResult(
            query=scenario["query"],
            expected_tool=scenario["expected_tool"],
            selected_tool=None,
            correct=False,
            level=scenario.get("level", 1),
            category=scenario.get("category", "unknown"),
            error=str(e),
        )


def _create_test_documents(tmp_path: Path) -> None:
    """Create test documents matching scenario contexts."""
    # Standard novel document
    novel_dir = tmp_path / "novel"
    novel_dir.mkdir(parents=True)
    (novel_dir / "01-intro.md").write_text("# Intro\n\nParagraph one.\n\nParagraph two.\n\nParagraph three.")

    # Empty document for edge cases
    empty_dir = tmp_path / "empty_doc"
    empty_dir.mkdir(parents=True)
    (empty_dir / "01-empty.md").write_text("")

    # Minimal document (single paragraph)
    minimal_dir = tmp_path / "minimal"
    minimal_dir.mkdir(parents=True)
    (minimal_dir / "01-single.md").write_text("The only paragraph in this chapter.")

    # Unicode document
    unicode_dir = tmp_path / "international"
    unicode_dir.mkdir(parents=True)
    (unicode_dir / "01-unicode.md").write_text("日本語テスト\n\nΕλληνικά κείμενο\n\n中文内容\n\nالعربية")

    # Special characters document
    special_dir = tmp_path / "special"
    special_dir.mkdir(parents=True)
    (special_dir / "01-special.md").write_text(
        "Normal text\n\n"
        "Text with **markdown** and `code`\n\n"
        "Text with <html> tags\n\n"
        "Text with \"quotes\" and 'apostrophes'"
    )


def compare_results(result_files: list[str]) -> None:
    """Compare multiple benchmark result files.

    Args:
        result_files: Paths to result JSON files to compare

    Usage:
        Run benchmarks with different configs, then compare:
        python -m benchmarks.runner --tool-set 4-tool
        python -m benchmarks.runner --tool-set 8-tool
        # Then manually compare the JSON files in benchmarks/results/
    """
    results = []
    for file_path in result_files:
        with open(file_path) as f:
            results.append(json.load(f))

    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    for result in results:
        summary = result.get("summary", {})
        name = result.get("tool_set_name", "unknown")
        accuracy = summary.get("accuracy", 0)
        avg_in = summary.get("avg_input_tokens", 0)
        avg_out = summary.get("avg_output_tokens", 0)
        print(f"  {name:20}: {accuracy:.1%} accuracy, {avg_in:.0f} in / {avg_out:.0f} out tokens")

    print("=" * 70)


def print_scenario_stats() -> None:
    """Print statistics about available scenarios."""
    stats = get_scenario_stats()
    print("\n" + "=" * 50)
    print("BENCHMARK SCENARIO STATISTICS")
    print("=" * 50)
    print(f"Level 1 (Simple):      {stats['level_1_simple']:3d}")
    print(f"Level 2 (Sequential):  {stats['level_2_sequential']:3d}")
    print(f"Level 3 (Complex):     {stats['level_3_complex']:3d}")
    print(f"Level 4 (Ambiguous):   {stats['level_4_ambiguous']:3d}")
    print(f"Level 5 (Edge Case):   {stats['level_5_edge_case']:3d}")
    print(f"Level 6 (Adversarial): {stats['level_6_adversarial']:3d}")
    print("-" * 50)
    print(f"TOTAL:                 {stats['total']:3d}")
    print("=" * 50)


def main():
    """CLI entry point."""
    # Load environment variables from .env file
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    parser = argparse.ArgumentParser(
        description="Benchmark paragraph tool selection",
        epilog="""
Configuration is file-based. Add new configs by creating YAML files:
  - benchmarks/tool_sets/<name>.yaml   (tool set definitions)
  - benchmarks/descriptions/<name>.yaml (description styles)
  - benchmarks/models/<name>.yaml       (model configurations)

Examples:
    python -m benchmarks.runner --list               # List available configs
    python -m benchmarks.runner --tool-set 4-tool    # Run with 4-tool
    python -m benchmarks.runner --tool-set 8-tool    # Run with 8-tool
    python -m benchmarks.runner --model gpt-5-mini   # Use specific model
    python -m benchmarks.runner --no-edge-cases      # Exclude edge cases

To compare configs, run twice and compare results:
    python -m benchmarks.runner --tool-set 4-tool
    python -m benchmarks.runner --tool-set 8-tool
    # Then compare results in benchmarks/results/
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--tool-set",
        type=str,
        metavar="NAME",
        help="Tool set to use (auto-detected from benchmarks/tool_sets/<name>.yaml)",
    )
    parser.add_argument(
        "--model",
        type=str,
        metavar="NAME",
        help="Model to use (auto-detected from benchmarks/models/<name>.yaml)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available configurations",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print scenario statistics",
    )
    parser.add_argument(
        "--no-edge-cases",
        action="store_true",
        help="Exclude Level 5 edge case scenarios",
    )
    parser.add_argument(
        "--no-adversarial",
        action="store_true",
        help="Exclude Level 6 adversarial scenarios",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to file",
    )

    args = parser.parse_args()

    if args.list:
        print_available_configs()
        return

    if args.stats:
        print_scenario_stats()
        return

    # If no action specified, show help
    if not args.tool_set:
        print_scenario_stats()
        print("\n" + "=" * 60)
        print("USAGE")
        print("=" * 60)
        print("\nRun a benchmark:")
        print("  python -m benchmarks.runner --tool-set 4-tool")
        print("  python -m benchmarks.runner --tool-set 8-tool --model gpt-5-mini")
        print("\nList available configs:")
        print("  python -m benchmarks.runner --list")
        print("\nOptions:")
        print("  --no-edge-cases    Exclude Level 5 edge case scenarios")
        print("  --no-adversarial   Exclude Level 6 adversarial scenarios")
        print("  --no-save          Don't save results to file")
        print("\nResults are saved to benchmarks/results/ for comparison.")
        return

    if not check_api_key_available():
        print("ERROR: No API key found. Set OPENROUTER_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY")
        sys.exit(1)

    include_edge = not args.no_edge_cases
    include_adv = not args.no_adversarial

    report = asyncio.run(
        run_benchmark(
            tool_set_name=args.tool_set,
            model_name=args.model,
            include_edge_cases=include_edge,
            include_adversarial=include_adv,
            save_results=not args.no_save,
        )
    )
    print(f"\n{report.report()}")


if __name__ == "__main__":
    main()
