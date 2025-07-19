"""Tool imports helper for unit tests.

This module provides a centralized place to import all functions needed by unit tests,
making it easier to manage dependencies and avoid import duplication in test files.
"""

# Import public functions from mcp_client
# Helper functions that need to be accessible for unit testing
from document_mcp.helpers import _get_modification_history_path
from document_mcp.helpers import _get_snapshots_path

# Import validation functions for unit testing
from document_mcp.helpers import validate_chapter_name
from document_mcp.helpers import validate_document_name
from document_mcp.helpers import validate_paragraph_index
from document_mcp.helpers import validate_search_query
from document_mcp.mcp_client import find_text
from document_mcp.mcp_client import get_statistics
from document_mcp.mcp_client import read_content
from document_mcp.mcp_client import replace_text

# Import constants and helper functions
from document_mcp.utils.file_operations import DOCS_ROOT_PATH

__all__ = [
    # Public MCP functions
    "find_text",
    "get_statistics",
    "read_content",
    "replace_text",
    # Validation functions
    "validate_chapter_name",
    "validate_document_name",
    "validate_paragraph_index",
    "validate_search_query",
    # Constants and paths
    "DOCS_ROOT_PATH",
    # Helper functions
    "_get_modification_history_path",
    "_get_snapshots_path",
]
