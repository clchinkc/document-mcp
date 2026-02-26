"""Backward compatibility shim for document_mcp.utils.

This module redirects imports to story_mcp.utils.
DEPRECATED: Use 'from story_mcp import utils' instead.
"""

import warnings

warnings.warn(
    "The 'document_mcp.utils' module is deprecated. "
    "Use 'from story_mcp import utils' or 'from story_mcp.utils import ...' instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export all from story_mcp.utils
from story_mcp.utils import *  # noqa: F401, F403
