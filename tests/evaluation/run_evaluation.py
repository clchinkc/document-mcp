#!/usr/bin/env python3
"""Standalone evaluation runner for document-mcp agents.

This script runs the evaluation suite and generates performance reports
for both Simple and React agents across various test scenarios.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import tempfile

from tests.evaluation.config import get_evaluation_config
from tests.evaluation.config import get_test_scenarios
from tests.evaluation.evaluation_utils import compare_agent_performance
from tests.evaluation.evaluation_utils import generate_performance_summary
from tests.evaluation.test_agent_performance import AgentTestRunner
from tests.evaluation.test_agent_performance import print_performance_report


async def run_single_scenario(runner: AgentTestRunner, scenario: dict, agent_type: str):
    """Run a single test scenario for a given agent type."""
    try:
        query = scenario["query"]
        expected_operations = scenario.get("expected_operations", [])

        # Since we're using real LLM evaluation only, use the unified run_agent_test method
        metrics = await runner.run_agent_test(agent_type, query)

        return metrics

    except Exception as e:
        print(
            f"Error running scenario '{scenario['name']}' with {agent_type} agent: {e}"
        )
        from tests.evaluation.test_agent_performance import AgentPerformanceMetrics

        error_metrics = AgentPerformanceMetrics()
        error_metrics.success = False
        error_metrics.error_message = str(e)
        return error_metrics


async def run_evaluation_suite(
    categories: list[str] = None, use_real_llm: bool = False
):
    """Run the complete evaluation suite."""
    print("=" * 60)
    print("DOCUMENT-MCP AGENT EVALUATION SUITE")
    print("=" * 60)

    config = get_evaluation_config()

    # Get test scenarios
    if categories:
        scenarios = []
        for category in categories:
            scenarios.extend(get_test_scenarios(category))
    else:
        scenarios = get_test_scenarios()

    print(f"Running {len(scenarios)} test scenarios...")
    print(f"Using {'Real' if use_real_llm else 'Mock'} LLM")
    print("-" * 60)

    # Results storage
    all_results = {"simple": [], "react": [], "planner": [], "comparisons": []}

    with tempfile.TemporaryDirectory() as tmp_dir:
        docs_root = Path(tmp_dir)
        runner = AgentTestRunner(docs_root)

        # Run each scenario for all agent types
        for i, scenario in enumerate(scenarios, 1):
            print(f"\n[{i}/{len(scenarios)}] Running scenario: {scenario['name']}")
            print(f"Description: {scenario['description']}")
            print(f"Query: {scenario['query']}")

            # Run with Simple Agent
            print("  - Testing Simple Agent...")
            simple_metrics = await run_single_scenario(runner, scenario, "simple")
            all_results["simple"].append(simple_metrics)

            # Run with React Agent
            print("  - Testing React Agent...")
            react_metrics = await run_single_scenario(runner, scenario, "react")
            all_results["react"].append(react_metrics)

            # Run with Planner Agent
            print("  - Testing Planner Agent...")
            planner_metrics = await run_single_scenario(runner, scenario, "planner")
            all_results["planner"].append(planner_metrics)

            # Compare results (for now just Simple vs React, later can expand)
            comparison = compare_agent_performance(simple_metrics, react_metrics)
            all_results["comparisons"].append(
                {"scenario": scenario["name"], "comparison": comparison}
            )

            # Print quick summary
            simple_status = "✓" if simple_metrics.success else "✗"
            react_status = "✓" if react_metrics.success else "✗"
            planner_status = "✓" if planner_metrics.success else "✗"
            print(
                f"  - Results: Simple {simple_status} ({simple_metrics.execution_time:.2f}s), "
                f"React {react_status} ({react_metrics.execution_time:.2f}s), "
                f"Planner {planner_status} ({planner_metrics.execution_time:.2f}s)"
            )

    # Generate comprehensive report
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    # Simple Agent Report
    print("\n--- SIMPLE AGENT PERFORMANCE ---")
    print_performance_report(all_results["simple"], "Simple Agent")
    simple_summary = generate_performance_summary(all_results["simple"])

    # React Agent Report
    print("\n--- REACT AGENT PERFORMANCE ---")
    print_performance_report(all_results["react"], "React Agent")
    react_summary = generate_performance_summary(all_results["react"])

    # Planner Agent Report
    print("\n--- PLANNER AGENT PERFORMANCE ---")
    print_performance_report(all_results["planner"], "Planner Agent")
    planner_summary = generate_performance_summary(all_results["planner"])

    # Comparison Report
    print("\n--- AGENT COMPARISON ---")
    print_comparison_report(
        all_results["comparisons"], simple_summary, react_summary, planner_summary
    )

    # Save results if configured
    if config.save_metrics_to_file:
        save_results_to_file(all_results, config.metrics_file_path)
        print(f"\nResults saved to {config.metrics_file_path}")

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


def print_comparison_report(
    comparisons: list[dict],
    simple_summary: dict,
    react_summary: dict,
    planner_summary: dict,
):
    """Print a detailed comparison report between agents."""
    print(f"Total Scenarios: {len(comparisons)}")
    print(f"Simple Agent Success Rate: {simple_summary['success_rate']:.2%}")
    print(f"React Agent Success Rate: {react_summary['success_rate']:.2%}")
    print(f"Planner Agent Success Rate: {planner_summary['success_rate']:.2%}")

    # Token efficiency comparison
    simple_avg_tokens = simple_summary["average_token_usage"]
    react_avg_tokens = react_summary["average_token_usage"]
    planner_avg_tokens = planner_summary["average_token_usage"]
    react_token_ratio = (
        react_avg_tokens / simple_avg_tokens if simple_avg_tokens > 0 else 0
    )
    planner_token_ratio = (
        planner_avg_tokens / simple_avg_tokens if simple_avg_tokens > 0 else 0
    )

    print("\nToken Usage Comparison:")
    print(f"  Simple Agent Average: {simple_avg_tokens:.1f} tokens")
    print(f"  React Agent Average: {react_avg_tokens:.1f} tokens")
    print(f"  Planner Agent Average: {planner_avg_tokens:.1f} tokens")
    print(f"  React/Simple Ratio: {react_token_ratio:.1f}x")
    print(f"  Planner/Simple Ratio: {planner_token_ratio:.1f}x")

    # Speed comparison
    simple_avg_time = simple_summary["average_execution_time"]
    react_avg_time = react_summary["average_execution_time"]
    planner_avg_time = planner_summary["average_execution_time"]
    react_time_ratio = react_avg_time / simple_avg_time if simple_avg_time > 0 else 0
    planner_time_ratio = (
        planner_avg_time / simple_avg_time if simple_avg_time > 0 else 0
    )

    print("\nExecution Time Comparison:")
    print(f"  Simple Agent Average: {simple_avg_time:.2f}s")
    print(f"  React Agent Average: {react_avg_time:.2f}s")
    print(f"  Planner Agent Average: {planner_avg_time:.2f}s")
    print(f"  React/Simple Ratio: {react_time_ratio:.1f}x")
    print(f"  Planner/Simple Ratio: {planner_time_ratio:.1f}x")

    # Performance buckets
    print("\nPerformance Distribution:")
    print(
        f"  Simple Agent - Fast: {simple_summary['performance_buckets']['fast']}, "
        f"Medium: {simple_summary['performance_buckets']['medium']}, "
        f"Slow: {simple_summary['performance_buckets']['slow']}"
    )
    print(
        f"  React Agent - Fast: {react_summary['performance_buckets']['fast']}, "
        f"Medium: {react_summary['performance_buckets']['medium']}, "
        f"Slow: {react_summary['performance_buckets']['slow']}"
    )
    print(
        f"  Planner Agent - Fast: {planner_summary['performance_buckets']['fast']}, "
        f"Medium: {planner_summary['performance_buckets']['medium']}, "
        f"Slow: {planner_summary['performance_buckets']['slow']}"
    )

    # Token efficiency buckets
    print("\nToken Efficiency Distribution:")
    print(
        f"  Simple Agent - Efficient: {simple_summary['token_efficiency_buckets']['efficient']}, "
        f"Moderate: {simple_summary['token_efficiency_buckets']['moderate']}, "
        f"Heavy: {simple_summary['token_efficiency_buckets']['heavy']}"
    )
    print(
        f"  React Agent - Efficient: {react_summary['token_efficiency_buckets']['efficient']}, "
        f"Moderate: {react_summary['token_efficiency_buckets']['moderate']}, "
        f"Heavy: {react_summary['token_efficiency_buckets']['heavy']}"
    )
    print(
        f"  Planner Agent - Efficient: {planner_summary['token_efficiency_buckets']['efficient']}, "
        f"Moderate: {planner_summary['token_efficiency_buckets']['moderate']}, "
        f"Heavy: {planner_summary['token_efficiency_buckets']['heavy']}"
    )


def save_results_to_file(results: dict, file_path: str):
    """Save evaluation results to a JSON file."""
    # Convert metrics to dictionaries for JSON serialization
    serializable_results = {
        "simple": [m.to_dict() for m in results["simple"]],
        "react": [m.to_dict() for m in results["react"]],
        "comparisons": results["comparisons"],
    }

    with open(file_path, "w") as f:
        json.dump(serializable_results, f, indent=2)


def main():
    """Main entry point for the evaluation runner."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run document-mcp agent evaluation suite"
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=["basic", "intermediate", "advanced", "complex", "query"],
        help="Test categories to run (default: all)",
    )
    parser.add_argument(
        "--real-llm",
        action="store_true",
        help="Use real LLM instead of mock (requires API keys)",
    )
    parser.add_argument(
        "--save-results", action="store_true", help="Save results to JSON file"
    )

    args = parser.parse_args()

    # Run the evaluation suite
    try:
        asyncio.run(
            run_evaluation_suite(categories=args.categories, use_real_llm=args.real_llm)
        )
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error running evaluation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
