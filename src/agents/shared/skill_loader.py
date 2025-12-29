"""Skill loader for Document MCP agents.

This module loads SKILL.md content for injection into agent prompts,
providing consistent tool usage patterns across Claude Code and our agents.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def find_skill_md() -> Path | None:
    """Find the SKILL.md file in the project.

    Searches in order:
    1. .claude/skills/document-mcp/SKILL.md (project skills)
    2. ~/.claude/skills/document-mcp/SKILL.md (user skills)

    Returns:
        Path to SKILL.md or None if not found.
    """
    # Get project root (assumes we're in src/agents/shared/)
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent.parent

    # Check project skills first
    project_skill = project_root / ".claude" / "skills" / "document-mcp" / "SKILL.md"
    if project_skill.exists():
        return project_skill

    # Check user skills
    user_skill = Path.home() / ".claude" / "skills" / "document-mcp" / "SKILL.md"
    if user_skill.exists():
        return user_skill

    return None


def parse_skill_md(content: str) -> dict[str, str]:
    """Parse SKILL.md content into sections.

    Args:
        content: Raw SKILL.md file content.

    Returns:
        Dictionary with section names as keys and content as values.
    """
    sections = {}
    current_section = None
    current_content = []

    # Skip YAML frontmatter
    lines = content.split("\n")
    in_frontmatter = False
    start_idx = 0

    for i, line in enumerate(lines):
        if i == 0 and line.strip() == "---":
            in_frontmatter = True
            continue
        if in_frontmatter and line.strip() == "---":
            start_idx = i + 1
            break

    # Parse sections
    for line in lines[start_idx:]:
        if line.startswith("## "):
            if current_section:
                sections[current_section] = "\n".join(current_content).strip()
            current_section = line[3:].strip()
            current_content = []
        elif current_section:
            current_content.append(line)

    # Save last section
    if current_section:
        sections[current_section] = "\n".join(current_content).strip()

    return sections


def get_skill_content() -> dict[str, str] | None:
    """Load and parse SKILL.md content.

    Returns:
        Parsed sections or None if SKILL.md not found.
    """
    skill_path = find_skill_md()
    if not skill_path:
        logger.debug("SKILL.md not found, using built-in prompts only")
        return None

    try:
        content = skill_path.read_text(encoding="utf-8")
        sections = parse_skill_md(content)
        logger.debug(f"Loaded SKILL.md with {len(sections)} sections")
        return sections
    except Exception as e:
        logger.warning(f"Failed to load SKILL.md: {e}")
        return None


def get_critical_workflows() -> str:
    """Get the Critical Workflows section from SKILL.md.

    This section contains important patterns for tool selection
    that improve agent accuracy.

    Returns:
        Critical workflows content or empty string if not available.
    """
    sections = get_skill_content()
    if not sections:
        return ""

    return sections.get("Critical Workflows", "")


def get_quick_start_examples() -> str:
    """Get the Quick Start section from SKILL.md.

    Returns:
        Quick start examples or empty string if not available.
    """
    sections = get_skill_content()
    if not sections:
        return ""

    return sections.get("Quick Start", "")


def get_best_practices() -> str:
    """Get the Best Practices section from SKILL.md.

    Returns:
        Best practices content or empty string if not available.
    """
    sections = get_skill_content()
    if not sections:
        return ""

    return sections.get("Best Practices", "")


def get_skill_enhanced_prompt_section() -> str:
    """Get a formatted prompt section with key SKILL.md content.

    This provides the most valuable patterns from SKILL.md
    for injection into agent prompts.

    Returns:
        Formatted prompt section or empty string if SKILL.md not available.
    """
    sections = get_skill_content()
    if not sections:
        return ""

    parts = []

    # Add critical workflows - most important for tool selection
    workflows = sections.get("Critical Workflows", "")
    if workflows:
        parts.append("**CRITICAL WORKFLOWS (from SKILL.md):**")
        parts.append(workflows)

    return "\n\n".join(parts)
