"""Abstract Base Class for Storage Backends.

Defines the interface that all storage backends must implement.
"""
from __future__ import annotations


from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from datetime import datetime


@dataclass
class FileInfo:
    """Metadata about a stored file."""

    path: str
    size: int
    last_modified: datetime
    is_directory: bool = False
    content_type: str = "text/markdown"


@dataclass
class DirectoryInfo:
    """Metadata about a directory."""

    path: str
    last_modified: datetime
    file_count: int = 0


class StorageBackend(ABC):
    """Abstract base class for storage backends.

    All storage implementations (local filesystem, GCS, etc.) must
    implement this interface for consistent document operations.
    """

    @property
    @abstractmethod
    def backend_type(self) -> str:
        """Return the backend type identifier (e.g., 'local', 'gcs')."""
        pass

    @property
    @abstractmethod
    def root_path(self) -> str:
        """Return the root path/bucket for storage."""
        pass

    # === File Operations ===

    @abstractmethod
    async def read_file(self, path: str) -> str:
        """Read file content as text.

        Args:
            path: Relative path from storage root (e.g., "doc_name/chapter.md")

        Returns:
            File content as string

        Raises:
            FileNotFoundError: If file does not exist
        """
        pass

    @abstractmethod
    async def write_file(self, path: str, content: str) -> None:
        """Write content to a file.

        Creates parent directories if they don't exist.

        Args:
            path: Relative path from storage root
            content: Text content to write
        """
        pass

    @abstractmethod
    async def delete_file(self, path: str) -> bool:
        """Delete a file.

        Args:
            path: Relative path from storage root

        Returns:
            True if file was deleted, False if it didn't exist
        """
        pass

    @abstractmethod
    async def file_exists(self, path: str) -> bool:
        """Check if a file exists.

        Args:
            path: Relative path from storage root

        Returns:
            True if file exists, False otherwise
        """
        pass

    @abstractmethod
    async def get_file_info(self, path: str) -> FileInfo | None:
        """Get metadata about a file.

        Args:
            path: Relative path from storage root

        Returns:
            FileInfo if file exists, None otherwise
        """
        pass

    # === Directory Operations ===

    @abstractmethod
    async def list_directory(self, path: str = "") -> list[str]:
        """List entries in a directory.

        Args:
            path: Relative path from storage root (empty for root)

        Returns:
            List of entry names (not full paths)
        """
        pass

    @abstractmethod
    async def list_files(self, path: str = "", pattern: str = "*.md") -> list[str]:
        """List files matching a pattern in a directory.

        Args:
            path: Relative path from storage root
            pattern: Glob pattern to match (e.g., "*.md")

        Returns:
            List of matching file paths relative to storage root
        """
        pass

    @abstractmethod
    async def directory_exists(self, path: str) -> bool:
        """Check if a directory exists.

        Args:
            path: Relative path from storage root

        Returns:
            True if directory exists, False otherwise
        """
        pass

    @abstractmethod
    async def create_directory(self, path: str) -> None:
        """Create a directory (and parents if needed).

        Args:
            path: Relative path from storage root
        """
        pass

    @abstractmethod
    async def delete_directory(self, path: str, recursive: bool = False) -> bool:
        """Delete a directory.

        Args:
            path: Relative path from storage root
            recursive: If True, delete contents recursively

        Returns:
            True if deleted, False if didn't exist
        """
        pass

    # === Bulk Operations ===

    @abstractmethod
    async def copy_file(self, source: str, destination: str) -> None:
        """Copy a file.

        Args:
            source: Source path relative to storage root
            destination: Destination path relative to storage root
        """
        pass

    @abstractmethod
    async def move_file(self, source: str, destination: str) -> None:
        """Move/rename a file.

        Args:
            source: Source path relative to storage root
            destination: Destination path relative to storage root
        """
        pass

    @abstractmethod
    async def copy_directory(self, source: str, destination: str) -> None:
        """Copy a directory recursively.

        Args:
            source: Source path relative to storage root
            destination: Destination path relative to storage root
        """
        pass

    # === Sync Wrappers (for compatibility) ===

    def read_file_sync(self, path: str) -> str:
        """Synchronous wrapper for read_file."""
        import asyncio

        return asyncio.run(self.read_file(path))

    def write_file_sync(self, path: str, content: str) -> None:
        """Synchronous wrapper for write_file."""
        import asyncio

        asyncio.run(self.write_file(path, content))

    def file_exists_sync(self, path: str) -> bool:
        """Synchronous wrapper for file_exists."""
        import asyncio

        return asyncio.run(self.file_exists(path))

    def list_directory_sync(self, path: str = "") -> list[str]:
        """Synchronous wrapper for list_directory."""
        import asyncio

        return asyncio.run(self.list_directory(path))
