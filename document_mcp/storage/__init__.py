"""Storage Abstraction Layer for Document MCP.

Provides a unified interface for document storage that works with both
local filesystem and cloud storage backends (Google Cloud Storage).

The backend is automatically selected based on environment detection:
- Local: Default for development, uses DOCUMENT_ROOT_DIR
- GCS: When K_SERVICE (Cloud Run) or GCS_BUCKET is set

Usage:
    from document_mcp.storage import get_storage

    storage = get_storage()
    content = await storage.read_file("my_doc/01-intro.md")
    await storage.write_file("my_doc/01-intro.md", "# Introduction")
"""

from .base import StorageBackend
from .factory import StorageType
from .factory import get_storage
from .factory import get_storage_sync

__all__ = ["StorageBackend", "get_storage", "get_storage_sync", "StorageType"]
