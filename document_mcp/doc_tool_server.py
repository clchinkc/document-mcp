"""MCP Server for Document Management.

This module provides a FastMCP-based MCP server for managing structured Markdown documents.
It exposes tools for creating, reading, updating, and deleting documents and chapters,
as well as for analyzing their content.
"""

import argparse
import datetime
import os
from dataclasses import dataclass
from dataclasses import field
from typing import Any

from dotenv import load_dotenv
from mcp.server import FastMCP

# Local imports for safe operation handling
# Import models
from .models import BatchApplyResult
from .models import BatchOperation
from .models import ChapterContent
from .models import ChapterMetadata
from .models import ContentFreshnessStatus
from .models import DocumentInfo
from .models import DocumentSummary
from .models import FullDocumentContent
from .models import ModificationHistory
from .models import ModificationHistoryEntry
from .models import OperationResult
from .models import OperationStatus
from .models import ParagraphDetail
from .models import SnapshotInfo
from .models import SnapshotsList
from .models import StatisticsReport
from .utils.file_operations import DOCS_ROOT_PATH

# Import metrics functionality
from .metrics_config import METRICS_ENABLED

# Import tool registration functions from modular architecture
from .tools import register_batch_tools
from .tools import register_chapter_tools
from .tools import register_content_tools
from .tools import register_document_tools
from .tools import register_paragraph_tools
from .tools import register_safety_tools

# --- Configuration ---
# Each "document" will be a subdirectory within DOCS_ROOT_DIR.
# Chapters will be .md files within their respective document subdirectory.
# Default for production
_DEFAULT_DOCS_ROOT = ".documents_storage"

# Load environment variables from .env file
load_dotenv()

# Check if running under pytest. If so, allow override via .env for test isolation.
# In production, this is not used and the path is fixed.
if "PYTEST_CURRENT_TEST" in os.environ:
    DOCS_ROOT_DIR_NAME = os.environ.get("DOCUMENT_ROOT_DIR", _DEFAULT_DOCS_ROOT)
else:
    DOCS_ROOT_DIR_NAME = _DEFAULT_DOCS_ROOT

# Use DOCS_ROOT_PATH from utils.file_operations to avoid duplication
# Ensure the root directory exists
DOCS_ROOT_PATH.mkdir(parents=True, exist_ok=True)

# HTTP SSE server configuration
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 3001

# --- Enhanced Automatic Snapshot System Configuration ---


@dataclass
class UserModificationRecord:
    """Enhanced user modification tracking for automatic snapshots."""

    user_id: str
    operation_type: str  # "edit", "create", "delete", "batch"
    affected_scope: str  # "document", "chapter", "paragraph"
    timestamp: datetime.datetime
    snapshot_id: str
    operation_details: dict[str, Any]
    restoration_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SnapshotRetentionPolicy:
    """Intelligent snapshot cleanup with user priority."""

    # High Priority (Keep Longer)
    USER_EDIT_SNAPSHOTS = 30  # days - User-initiated changes
    MILESTONE_SNAPSHOTS = 90  # days - Major document versions
    ERROR_RECOVERY_SNAPSHOTS = 7  # days - Failed operation rollbacks

    # Medium Priority
    BATCH_OPERATION_SNAPSHOTS = 14  # days - Batch operation checkpoints
    CHAPTER_LEVEL_SNAPSHOTS = 14  # days - Chapter modifications

    # Low Priority (Cleanup Frequently)
    PARAGRAPH_LEVEL_SNAPSHOTS = 3  # days - Small edits
    AUTO_BACKUP_SNAPSHOTS = 1  # days - System automated backups


# Global retention policy instance
RETENTION_POLICY = SnapshotRetentionPolicy()

# Track user modifications for better UX
_user_modification_history: list[UserModificationRecord] = []


def get_current_user() -> str:
    """Get current user identifier for tracking modifications."""
    # In production, this would integrate with authentication system
    # For now, return a simple identifier
    return os.environ.get("USER", "system_user")


mcp_server = FastMCP(
    name="DocumentManagementTools", capabilities=["tools", "resources"]
)

# Register tools from modular architecture
register_document_tools(mcp_server)
register_chapter_tools(mcp_server)
register_paragraph_tools(mcp_server)
register_content_tools(mcp_server)
register_safety_tools(mcp_server)
register_batch_tools(mcp_server)

# Export only essential items for MCP server module
__all__ = [
    # Models and types
    "BatchApplyResult",
    "BatchOperation",
    "ChapterContent",
    "ChapterMetadata",
    "ContentFreshnessStatus",
    "DocumentInfo",
    "DocumentSummary",
    "FullDocumentContent",
    "ModificationHistory",
    "ModificationHistoryEntry",
    "OperationResult",
    "OperationStatus",
    "ParagraphDetail",
    "SnapshotInfo",
    "SnapshotsList",
    "StatisticsReport",
    # Constants
    "DOCS_ROOT_PATH",
    # MCP Server (primary export)
    "mcp_server",
    # No internal batch helpers - use batch.registry.execute_batch_operation
]

# --- Main Server Execution ---
def main():
    """Run the main entry point for the server with argument parsing."""
    parser = argparse.ArgumentParser(description="Document MCP Server")
    parser.add_argument(
        "transport",
        choices=["sse", "stdio"],
        default="stdio",
        nargs="?",
        help="Transport: 'sse' for HTTP SSE or 'stdio' for standard I/O (default: stdio)",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"Host to bind to for SSE transport (default: {DEFAULT_HOST})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Port to bind to for SSE transport (default: {DEFAULT_PORT})",
    )

    args = parser.parse_args()

    # This print will show the path used by the subprocess
    root_path = DOCS_ROOT_PATH.resolve()
    print(f"doc_tool_server.py: Initializing with DOCS_ROOT_PATH = {root_path}")
    print(f"Document tool server starting. Tools exposed by '{mcp_server.name}':")
    print(f"Serving tools for root directory: {DOCS_ROOT_PATH.resolve()}")

    # Show metrics status
    try:
        status = "enabled" if METRICS_ENABLED else "disabled"
        print(f"Metrics: {status}")
    except (ImportError, NameError):
        print("Metrics: not available (install prometheus-client)")

    if args.transport == "stdio":
        print(
            "MCP server running with stdio transport. Waiting for client connection..."
        )
        mcp_server.run(transport="stdio")
    else:
        print(f"MCP server running with HTTP SSE transport on {args.host}:{args.port}")
        print(f"SSE endpoint: http://{args.host}:{args.port}/sse")
        print(f"Health endpoint: http://{args.host}:{args.port}/health")
        if METRICS_AVAILABLE:
            print(f"Metrics endpoint: http://{args.host}:{args.port}/metrics")
        # Update server settings before running
        mcp_server.settings.host = args.host
        mcp_server.settings.port = args.port
        mcp_server.run(transport="sse")


if __name__ == "__main__":
    main()
