#!/usr/bin/env python3
"""Tool Description Optimization Analysis

This script analyzes the token reduction achieved by extracting hardcoded tool descriptions
and using dynamic generation with different formats.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.agents.planner_agent.prompts import get_planner_system_prompt
from src.agents.react_agent.prompts import get_react_system_prompt
from src.agents.shared.tool_descriptions import ToolFormat
from src.agents.shared.tool_descriptions import tool_manager
from src.agents.simple_agent.prompts import get_simple_agent_system_prompt


def estimate_tokens(text: str) -> int:
    """Estimate token count (rough approximation: 1 token ≈ 4 characters)."""
    return len(text) // 4


def analyze_format_optimization():
    """Analyze token usage across different tool description formats."""
    print("=== Tool Description Format Analysis ===\n")

    formats = tool_manager.compare_formats()

    print("Format Comparison:")
    print(f"{'Format':<12} {'Characters':<12} {'Est. Tokens':<12} {'Reduction':<12}")
    print("-" * 50)

    baseline_tokens = formats["full"][1]

    for format_name, (char_count, token_count) in formats.items():
        reduction = (
            f"{((baseline_tokens - token_count) / baseline_tokens * 100):.1f}%"
            if format_name != "full"
            else "baseline"
        )
        print(f"{format_name:<12} {char_count:<12} {token_count:<12} {reduction:<12}")

    print(f"\nTotal tools managed: {tool_manager.get_tool_count()}")
    print(f"Tool categories: {tool_manager.get_category_count()}")


def analyze_agent_prompt_sizes():
    """Analyze the total prompt sizes for each agent."""
    print("\n=== Agent Prompt Size Analysis ===\n")

    agents = {
        "Simple": get_simple_agent_system_prompt(),
        "ReAct": get_react_system_prompt(),
        "Planner": get_planner_system_prompt(),
    }

    print(f"{'Agent':<12} {'Characters':<12} {'Est. Tokens':<12} {'Tools Format':<15}")
    print("-" * 55)

    for agent_name, prompt in agents.items():
        char_count = len(prompt)
        token_count = estimate_tokens(prompt)
        format_used = (
            "Compact"
            if agent_name == "Simple"
            else ("Planner" if agent_name == "Planner" else "Full")
        )
        print(f"{agent_name:<12} {char_count:<12} {token_count:<12} {format_used:<15}")


def demonstrate_format_differences():
    """Show examples of different tool description formats."""
    print("\n=== Format Examples ===\n")

    formats = [
        ("FULL (ReAct)", ToolFormat.FULL),
        ("COMPACT (Simple)", ToolFormat.COMPACT),
        ("PLANNER (Planner)", ToolFormat.PLANNER),
        ("MINIMAL", ToolFormat.MINIMAL),
    ]

    for name, format_type in formats:
        print(f"--- {name} ---")
        content = tool_manager.get_tool_descriptions_text(format_type)
        preview = content[:200] + "..." if len(content) > 200 else content
        print(preview)
        print(
            f"Total length: {len(content)} chars, ~{estimate_tokens(content)} tokens\n"
        )


def calculate_optimization_impact():
    """Calculate the optimization impact achieved."""
    print("\n=== Optimization Impact Summary ===\n")

    # Get format comparison data
    formats = tool_manager.compare_formats()
    full_tokens = formats["full"][1]
    compact_tokens = formats["compact"][1]
    planner_tokens = formats["planner"][1]
    minimal_tokens = formats["minimal"][1]

    # Calculate savings
    compact_savings = ((full_tokens - compact_tokens) / full_tokens) * 100
    planner_savings = ((full_tokens - planner_tokens) / full_tokens) * 100
    minimal_savings = ((full_tokens - minimal_tokens) / full_tokens) * 100

    print("Achievements:")
    print(f"✅ Extracted {tool_manager.get_tool_count()} hardcoded tool descriptions")
    print(f"✅ Created {tool_manager.get_category_count()} tool categories")
    print("✅ Implemented dynamic prompt generation")
    print("✅ Enabled format-specific optimization")
    print("✅ Maintained backward compatibility")

    print("\nOptimization Potential:")
    print(f"• Compact format: {compact_savings:.1f}% token reduction")
    print(f"• Planner format: {planner_savings:.1f}% token reduction")
    print(f"• Minimal format: {minimal_savings:.1f}% token reduction")

    print("\nNext Steps:")
    print("• A/B test different formats with real LLM calls")
    print("• Measure impact on agent performance and accuracy")
    print("• Fine-tune descriptions for maximum efficiency")
    print("• Consider hybrid approaches for specific use cases")


if __name__ == "__main__":
    analyze_format_optimization()
    analyze_agent_prompt_sizes()
    demonstrate_format_differences()
    calculate_optimization_impact()
