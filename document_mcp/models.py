"""Backward compatibility shim for document_mcp.models.

This module redirects imports to story_mcp.models.
DEPRECATED: Use 'from story_mcp import models' instead.
"""

import warnings

warnings.warn(
    "The 'document_mcp.models' module is deprecated. "
    "Use 'from story_mcp import models' or 'from story_mcp.models import ...' instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export all from story_mcp.models
from story_mcp.models import *  # noqa: F401, F403
