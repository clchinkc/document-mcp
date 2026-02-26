"""
Backward compatibility layer for document_mcp.

DEPRECATION NOTICE: document_mcp is deprecated and will be removed in version 1.0 (approximately 6 months from now).

Please migrate to story_mcp:
  Old: from document_mcp import ...
  New: from story_mcp import ...

This compatibility module automatically redirects all document_mcp imports to story_mcp.
Both old and new imports will work during the deprecation period.

Timeline:
  - v0.0.5+: Both document_mcp and story_mcp work (with warnings)
  - v1.0.0: document_mcp is removed, only story_mcp available

See docs/STORY_MCP_MIGRATION.md for migration guide.
"""

import sys
import warnings
from typing import Any

# Issue deprecation warning on import
warnings.warn(
    "The 'document_mcp' package is deprecated and will be removed in version 1.0. "
    "Please migrate to 'story_mcp' (e.g., 'from story_mcp import ...' instead of "
    "'from document_mcp import ...'). See docs/STORY_MCP_MIGRATION.md for migration guide.",
    DeprecationWarning,
    stacklevel=2,
)

# Lazy import handler for backward compatibility
# This avoids loading the full doc_tool_server at import time
def __getattr__(name: str) -> Any:
    """Lazy import from story_mcp for backward compatibility."""
    try:
        # Import from story_mcp on demand
        import story_mcp as story_module
        attr = getattr(story_module, name)
        return attr
    except AttributeError as e:
        raise AttributeError(f"module 'document_mcp' has no attribute '{name}'") from e


def __dir__() -> list[str]:
    """Return list of available attributes."""
    import story_mcp
    return list(dir(story_mcp))
