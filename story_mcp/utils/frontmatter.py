"""YAML frontmatter parsing and writing utilities.

This module provides functions to parse and write YAML frontmatter
in Markdown chapter files.

Frontmatter format:
---
status: draft
pov_character: Marcus
tags: [action, dialogue]
---

# Chapter content here...
"""

import re
from typing import Any

import yaml  # type: ignore[import-untyped]

# Regex pattern for YAML frontmatter (--- delimited block at start of file)
FRONTMATTER_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)


def parse_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    """Parse YAML frontmatter from Markdown content.

    Args:
        content: Full file content that may contain frontmatter

    Returns:
        Tuple of (metadata dict, content without frontmatter)
        Returns empty dict if no frontmatter present
    """
    if not content or not content.startswith("---"):
        return {}, content

    match = FRONTMATTER_PATTERN.match(content)
    if not match:
        return {}, content

    yaml_block = match.group(1)
    remaining_content = content[match.end() :]

    try:
        metadata = yaml.safe_load(yaml_block)
        if metadata is None:
            metadata = {}
    except yaml.YAMLError:
        # Invalid YAML - return empty metadata but preserve content
        return {}, content

    return metadata, remaining_content


def write_frontmatter(content: str, metadata: dict[str, Any]) -> str:
    """Write YAML frontmatter to content, preserving or replacing existing.

    Args:
        content: File content (may already have frontmatter)
        metadata: Metadata dict to write as frontmatter

    Returns:
        Content with updated frontmatter
    """
    if not metadata:
        # No metadata to write - strip existing frontmatter if present
        _, body = parse_frontmatter(content)
        return body

    # Get content without existing frontmatter
    _, body = parse_frontmatter(content)

    # Generate YAML block
    yaml_str = yaml.dump(metadata, default_flow_style=False, allow_unicode=True, sort_keys=False)

    # Combine frontmatter with content
    return f"---\n{yaml_str}---\n\n{body.lstrip()}"


def update_frontmatter(content: str, updates: dict[str, Any]) -> str:
    """Update specific frontmatter fields, preserving others.

    Args:
        content: File content with optional existing frontmatter
        updates: Dict of fields to add/update (None values remove fields)

    Returns:
        Content with merged frontmatter
    """
    existing, body = parse_frontmatter(content)

    # Merge updates into existing metadata
    merged = existing.copy()
    for key, value in updates.items():
        if value is None:
            merged.pop(key, None)
        else:
            merged[key] = value

    return write_frontmatter(body, merged)


def has_frontmatter(content: str) -> bool:
    """Check if content has YAML frontmatter."""
    if not content or not content.startswith("---"):
        return False
    return bool(FRONTMATTER_PATTERN.match(content))


def get_content_without_frontmatter(content: str) -> str:
    """Get content with frontmatter stripped."""
    _, body = parse_frontmatter(content)
    return body
