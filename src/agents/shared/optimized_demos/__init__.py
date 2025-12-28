"""Optimized few-shot demonstrations loader.

This module loads DSPy-optimized demonstrations for injection into agent prompts,
providing the learned patterns from MIPROv2 optimization.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def get_demos_dir() -> Path:
    """Get the optimized demos directory."""
    return Path(__file__).parent


def load_demos(variant: str = "full") -> list[dict]:
    """Load optimized demos for a specific variant.

    Args:
        variant: Prompt variant ("compact", "full", "minimal")

    Returns:
        List of demo dictionaries with query, tool_name, and reasoning.
    """
    demos_dir = get_demos_dir()
    latest_file = demos_dir / f"demos_{variant}_latest.json"

    if not latest_file.exists():
        logger.debug(f"No optimized demos found for variant '{variant}'")
        return []

    try:
        with open(latest_file) as f:
            data = json.load(f)
            demos = data.get("demos", [])
            logger.debug(f"Loaded {len(demos)} demos for variant '{variant}'")
            return demos
    except Exception as e:
        logger.warning(f"Failed to load demos: {e}")
        return []


def format_demos_for_prompt(demos: list[dict]) -> str:
    """Format demos as examples for system prompt injection.

    Args:
        demos: List of demo dictionaries.

    Returns:
        Formatted string suitable for prompt injection.
    """
    if not demos:
        return ""

    lines = ["**TOOL SELECTION EXAMPLES (Learned from optimization):**"]

    for i, demo in enumerate(demos, 1):
        query = demo.get("query", "")
        tool = demo.get("tool_name", "")
        reasoning = demo.get("reasoning", "")

        lines.append(f"\nExample {i}:")
        lines.append(f'  Query: "{query}"')
        lines.append(f"  Tool: `{tool}`")
        if reasoning:
            # Truncate long reasoning
            short_reasoning = reasoning[:200] + "..." if len(reasoning) > 200 else reasoning
            lines.append(f"  Reasoning: {short_reasoning}")

    return "\n".join(lines)


def get_optimized_examples_section(variant: str = "full") -> str:
    """Get formatted optimization examples for prompt injection.

    This is the main entry point for agents to get learned examples.

    Args:
        variant: Prompt variant to load demos for.

    Returns:
        Formatted examples section or empty string if no demos available.
    """
    demos = load_demos(variant)
    return format_demos_for_prompt(demos)
