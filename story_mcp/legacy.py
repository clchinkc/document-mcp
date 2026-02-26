"""
Legacy CLI entry point for backward compatibility.

This module provides the old 'document-mcp' command that redirects to the new 'story-mcp'.
Users can continue using the old command name during the deprecation period.

DEPRECATION: This entry point will be removed in version 1.0.
Migration: Use 'story-mcp' instead of 'document-mcp'.
"""

import sys
import warnings

from story_mcp.doc_tool_server import main as story_mcp_main


def main_legacy() -> None:
    """Legacy entry point that delegates to the new story_mcp CLI."""
    warnings.warn(
        "The 'document-mcp' command is deprecated. Please use 'story-mcp' instead. "
        "The 'document-mcp' command will be removed in version 1.0.",
        DeprecationWarning,
        stacklevel=1,
    )
    sys.exit(story_mcp_main())


if __name__ == "__main__":
    main_legacy()
