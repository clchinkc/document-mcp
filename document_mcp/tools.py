"""Backward compatibility shim for document_mcp.tools.

This module redirects imports to story_mcp.tools.
DEPRECATED: Use 'from story_mcp import tools' instead.
"""

import warnings

warnings.warn(
    "The 'document_mcp.tools' module is deprecated. "
    "Use 'from story_mcp import tools' or 'from story_mcp.tools import ...' instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export all from story_mcp.tools
from story_mcp.tools import *  # noqa: F401, F403
