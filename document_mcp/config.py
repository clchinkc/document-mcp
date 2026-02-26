"""Backward compatibility shim for document_mcp.config.

This module redirects imports to story_mcp.config.
DEPRECATED: Use 'from story_mcp import config' instead.
"""

import warnings

warnings.warn(
    "The 'document_mcp.config' module is deprecated. "
    "Use 'from story_mcp import config' or 'from story_mcp.config import ...' instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export all from story_mcp.config
from story_mcp.config import *  # noqa: F401, F403
