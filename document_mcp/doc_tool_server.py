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
from pathlib import Path
from typing import Any

from mcp.server import FastMCP

# Local imports for safe operation handling
# Import configuration
from .config import get_settings

# Import metrics functionality
from .metrics_config import METRICS_ENABLED
from .metrics_config import ensure_metrics_initialized
from .models import ChapterContent
from .models import ChapterMetadata
from .models import ContentFreshnessStatus
from .models import DocumentInfo
from .models import DocumentSummary
from .models import FullDocumentContent
from .models import ModificationHistory
from .models import ModificationHistoryEntry
from .models import OperationStatus
from .models import ParagraphDetail
from .models import SnapshotInfo
from .models import SnapshotsList
from .models import StatisticsReport

# Import tool registration functions from modular architecture
from .tools import register_chapter_tools
from .tools import register_content_tools
from .tools import register_document_tools
from .tools import register_paragraph_tools
from .tools import register_safety_tools
from .utils.file_operations import DOCS_ROOT_PATH

# --- Configuration ---
# Get centralized settings
settings = get_settings()

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


mcp_server = FastMCP(name="DocumentManagementTools")

# Register tools from modular architecture
register_document_tools(mcp_server)
register_chapter_tools(mcp_server)
register_paragraph_tools(mcp_server)
register_content_tools(mcp_server)
register_safety_tools(mcp_server)

# Export only essential items for MCP server module
__all__ = [
    # Models and types
    "ChapterContent",
    "ChapterMetadata",
    "ContentFreshnessStatus",
    "DocumentInfo",
    "DocumentSummary",
    "FullDocumentContent",
    "ModificationHistory",
    "ModificationHistoryEntry",
    "OperationStatus",
    "ParagraphDetail",
    "SnapshotInfo",
    "SnapshotsList",
    "StatisticsReport",
    # Constants
    "DOCS_ROOT_PATH",
    # MCP Server (primary export)
    "mcp_server",
]


# --- Main Server Execution ---
def main():
    """Run the main entry point for the server with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Document MCP Server - Manage structured Markdown documents via MCP protocol",
        epilog="""
Examples:
  # Start server with stdio transport (for Claude Code/MCP clients)
  document-mcp stdio

  # Start server with HTTP SSE transport
  document-mcp sse --host localhost --port 3001

  # Test server startup
  document-mcp --help
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "transport",
        choices=["sse", "stdio"],
        default="stdio",
        nargs="?",
        help="Transport protocol: 'stdio' for MCP clients like Claude Code (default), 'sse' for HTTP server",
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

    # Initialize metrics only when server actually starts running (not during --help)
    ensure_metrics_initialized()

    # Debug: Show environment variables received by MCP server subprocess
    import os

    doc_root_env = os.environ.get("DOCUMENT_ROOT_DIR")
    pytest_env = os.environ.get("PYTEST_CURRENT_TEST")
    print(f"[MCP_SERVER_DEBUG] MCP server received DOCUMENT_ROOT_DIR: {doc_root_env}")
    print(f"[MCP_SERVER_DEBUG] MCP server received PYTEST_CURRENT_TEST: {pytest_env}")

    # This print will show the path used by the subprocess
    try:
        root_path = DOCS_ROOT_PATH.resolve()
        print(f"[MCP_SERVER_DEBUG] Initializing with DOCS_ROOT_PATH = {root_path}")
        print(f"Document tool server starting. Tools exposed by '{mcp_server.name}':")
        print(f"Serving tools for root directory: {root_path}")
    except Exception as e:
        print(f"[MCP_SERVER_DEBUG] Error resolving DOCS_ROOT_PATH: {e}")
        print(f"Document tool server starting. Tools exposed by '{mcp_server.name}':")
        print(f"Serving tools for root directory: {DOCS_ROOT_PATH}")

    # Debug: Show platform and current working directory
    import os
    import platform

    print(f"[MCP_SERVER_DEBUG] Platform: {platform.system()}")
    print(f"[MCP_SERVER_DEBUG] Current working directory: {os.getcwd()}")
    print(f"[MCP_SERVER_DEBUG] Python executable: {os.sys.executable}")

    # Check if documents directory will be created in expected location
    if doc_root_env:
        try:
            expected_docs_path = Path(doc_root_env).resolve()
            print(f"[MCP_SERVER_DEBUG] Expected documents path: {expected_docs_path}")
            if expected_docs_path.exists():
                print("[MCP_SERVER_DEBUG] Expected path exists: True")
            else:
                print("[MCP_SERVER_DEBUG] Expected path exists: False")
        except Exception as e:
            print(f"[MCP_SERVER_DEBUG] Error processing expected path {doc_root_env}: {e}")
    else:
        print("[MCP_SERVER_DEBUG] No DOCUMENT_ROOT_DIR env var")

    # Show automatic telemetry status
    try:
        from .metrics_config import get_metrics_summary

        summary = get_metrics_summary()
        if summary["status"] in ["active", "enabled"]:
            print(f"[OK] Automatic telemetry: {summary.get('telemetry_mode', 'active')}")
            print(f"   Service: {summary['service_name']} v{summary['service_version']}")
            print(f"   Environment: {summary['environment']}")
        elif summary["status"] == "shutdown":
            print("[INFO] Telemetry: shutdown after inactivity")
        elif summary["status"] == "disabled":
            print(f"[INFO] Telemetry: {summary.get('reason', 'disabled')}")
        else:
            print(f"[INFO] Telemetry: {summary['status']}")
    except (ImportError, NameError):
        print("[INFO] Telemetry: not available")

    if args.transport == "stdio":
        print("MCP server running with stdio transport. Waiting for client connection...")
        mcp_server.run(transport="stdio")
    else:
        print(f"MCP server running with HTTP SSE transport on {args.host}:{args.port}")
        print(f"SSE endpoint: http://{args.host}:{args.port}/sse")
        print(f"Health endpoint: http://{args.host}:{args.port}/health")
        if METRICS_ENABLED:
            print(f"Metrics endpoint: http://{args.host}:{args.port}/metrics")
        # Update server settings before running
        mcp_server.settings.host = args.host
        mcp_server.settings.port = args.port
        mcp_server.run(transport="sse")


if __name__ == "__main__":
    main()
