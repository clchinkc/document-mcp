#!/usr/bin/env python3
"""Validate tool consistency across all configuration sources.

This script ensures that tool names, counts, and coverage are consistent
across all the places where tools are defined:
1. MCP server registration (document_mcp/tools/*.py)
2. Agent tool descriptions (src/agents/shared/tool_descriptions.py)
3. Benchmark scenarios (benchmarks/scenarios.py)
4. DSPy optimizer (dspy_optimizer/optimizer.py)
5. Discovery registry (document_mcp/tools/discovery_tools.py)

Run this in CI to catch mismatches early:
    python scripts/validate_tool_consistency.py

Exit codes:
    0 - All consistent
    1 - Mismatch detected
"""

import re
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def get_mcp_registered_tools() -> set[str]:
    """Get tools registered with @mcp_server.tool() or @mcp.tool() decorators."""
    tools = set()
    tool_dirs = [
        PROJECT_ROOT / "document_mcp" / "tools",
    ]

    # Pattern to find tool registrations
    decorator_pattern = re.compile(r"@mcp(?:_server)?\.tool\(\)")
    def_pattern = re.compile(r"def (\w+)\(")

    for tool_dir in tool_dirs:
        if not tool_dir.exists():
            continue
        for py_file in tool_dir.glob("*.py"):
            if py_file.name.startswith("_"):
                continue
            content = py_file.read_text()
            lines = content.split("\n")

            for i, line in enumerate(lines):
                if decorator_pattern.search(line):
                    # Look for the function definition in next few lines
                    for j in range(i + 1, min(i + 5, len(lines))):
                        match = def_pattern.search(lines[j])
                        if match:
                            tools.add(match.group(1))
                            break

    return tools


def get_tool_description_tools() -> set[str]:
    """Get tools defined in ToolDescriptionManager."""
    from src.agents.shared.tool_descriptions import ToolDescriptionManager

    mgr = ToolDescriptionManager()
    return {t.name for t in mgr._tools}


def get_scenario_tools() -> set[str]:
    """Get tools covered by benchmark scenarios."""
    from benchmarks.scenarios import get_all_tools_stats

    return set(get_all_tools_stats()["tools_covered"])


def get_discovery_registry_tools() -> set[str]:
    """Get tools in the discovery registry."""
    # Read the discovery_tools.py file and extract tool names from _TOOL_REGISTRY
    discovery_file = PROJECT_ROOT / "document_mcp" / "tools" / "discovery_tools.py"
    content = discovery_file.read_text()

    tools = set()
    # Find all "name": "tool_name" patterns
    name_pattern = re.compile(r'"name":\s*"(\w+)"')
    for match in name_pattern.finditer(content):
        tools.add(match.group(1))

    return tools


def get_dspy_tools() -> set[str]:
    """Get tools used by DSPy optimizer."""
    dspy_file = PROJECT_ROOT / "dspy_optimizer" / "optimizer.py"
    content = dspy_file.read_text()

    # Check if DSPy imports from tool registry
    if "from src.agents.shared.tool_descriptions import" in content:
        from src.agents.shared.tool_descriptions import ToolDescriptionManager

        mgr = ToolDescriptionManager()
        return {t.name for t in mgr._tools}

    # Fallback: look for inline tool names (legacy format)
    tools = set()
    tool_pattern = re.compile(r"^- (\w+):", re.MULTILINE)
    for match in tool_pattern.finditer(content):
        tools.add(match.group(1))

    return tools


def main() -> int:
    """Run validation and report results."""
    print("=" * 60)
    print("TOOL CONSISTENCY VALIDATION")
    print("=" * 60)

    # Collect tools from all sources
    sources = {
        "MCP Registration": get_mcp_registered_tools(),
        "Tool Descriptions": get_tool_description_tools(),
        "Benchmark Scenarios": get_scenario_tools(),
        "Discovery Registry": get_discovery_registry_tools(),
        "DSPy Optimizer": get_dspy_tools(),
    }

    # Print counts
    print("\nTool counts by source:")
    for name, tools in sources.items():
        print(f"  {name}: {len(tools)} tools")

    # Check consistency
    reference = sources["MCP Registration"]
    all_consistent = True

    print("\n" + "-" * 60)
    print("COMPARISON TO MCP REGISTRATION (authoritative source)")
    print("-" * 60)

    for name, tools in sources.items():
        if name == "MCP Registration":
            continue

        missing = reference - tools
        extra = tools - reference

        if missing or extra:
            all_consistent = False
            print(f"\n❌ {name}:")
            if missing:
                print(f"  Missing tools: {sorted(missing)}")
            if extra:
                print(f"  Extra tools: {sorted(extra)}")
        else:
            print(f"\n✓ {name}: All {len(tools)} tools match")

    print("\n" + "=" * 60)
    if all_consistent:
        print("✓ ALL SOURCES CONSISTENT")
        return 0
    else:
        print("❌ INCONSISTENCIES DETECTED - See above for details")
        return 1


if __name__ == "__main__":
    sys.exit(main())
