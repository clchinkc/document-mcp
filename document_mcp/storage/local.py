"""Local Filesystem Storage Backend.

Implements the StorageBackend interface using the local filesystem.
This is the default backend for local development.
"""

from __future__ import annotations

import fnmatch
import os
import shutil
from datetime import datetime
from datetime import timezone
from pathlib import Path

from .base import FileInfo
from .base import StorageBackend


class LocalStorageBackend(StorageBackend):
    """Storage backend using the local filesystem.

    Args:
        root_dir: Root directory for document storage.
                  Defaults to DOCUMENT_ROOT_DIR env var or ".documents_storage"
    """

    def __init__(self, root_dir: str | None = None):
        if root_dir:
            self._root = Path(root_dir).resolve()
        else:
            # Use environment variable or default
            root_dir = os.environ.get("DOCUMENT_ROOT_DIR", ".documents_storage")
            self._root = Path(root_dir).resolve()

        # Ensure root directory exists
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def backend_type(self) -> str:
        return "local"

    @property
    def root_path(self) -> str:
        return str(self._root)

    def _full_path(self, path: str) -> Path:
        """Convert relative path to absolute path."""
        return self._root / path

    # === File Operations ===

    async def read_file(self, path: str) -> str:
        full_path = self._full_path(path)
        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return full_path.read_text(encoding="utf-8")

    async def write_file(self, path: str, content: str) -> None:
        full_path = self._full_path(path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding="utf-8")

    async def delete_file(self, path: str) -> bool:
        full_path = self._full_path(path)
        if full_path.exists() and full_path.is_file():
            full_path.unlink()
            return True
        return False

    async def file_exists(self, path: str) -> bool:
        full_path = self._full_path(path)
        return full_path.exists() and full_path.is_file()

    async def get_file_info(self, path: str) -> FileInfo | None:
        full_path = self._full_path(path)
        if not full_path.exists():
            return None

        stat = full_path.stat()
        return FileInfo(
            path=path,
            size=stat.st_size,
            last_modified=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
            is_directory=full_path.is_dir(),
            content_type=self._get_content_type(path),
        )

    def _get_content_type(self, path: str) -> str:
        """Determine content type from file extension."""
        ext = Path(path).suffix.lower()
        content_types = {
            ".md": "text/markdown",
            ".yaml": "text/yaml",
            ".yml": "text/yaml",
            ".json": "application/json",
            ".txt": "text/plain",
        }
        return content_types.get(ext, "application/octet-stream")

    # === Directory Operations ===

    async def list_directory(self, path: str = "") -> list[str]:
        full_path = self._full_path(path)
        if not full_path.exists() or not full_path.is_dir():
            return []

        entries = []
        for item in full_path.iterdir():
            # Skip hidden files/directories
            if not item.name.startswith("."):
                entries.append(item.name)

        return sorted(entries)

    async def list_files(self, path: str = "", pattern: str = "*.md") -> list[str]:
        full_path = self._full_path(path)
        if not full_path.exists() or not full_path.is_dir():
            return []

        files = []
        for item in full_path.iterdir():
            if item.is_file() and fnmatch.fnmatch(item.name, pattern):
                # Return path relative to storage root
                if path:
                    files.append(f"{path}/{item.name}")
                else:
                    files.append(item.name)

        return sorted(files)

    async def directory_exists(self, path: str) -> bool:
        full_path = self._full_path(path)
        return full_path.exists() and full_path.is_dir()

    async def create_directory(self, path: str) -> None:
        full_path = self._full_path(path)
        full_path.mkdir(parents=True, exist_ok=True)

    async def delete_directory(self, path: str, recursive: bool = False) -> bool:
        full_path = self._full_path(path)
        if not full_path.exists():
            return False

        if not full_path.is_dir():
            return False

        if recursive:
            shutil.rmtree(full_path)
        else:
            # Only delete if empty
            try:
                full_path.rmdir()
            except OSError:
                # Directory not empty
                return False

        return True

    # === Bulk Operations ===

    async def copy_file(self, source: str, destination: str) -> None:
        src_path = self._full_path(source)
        dst_path = self._full_path(destination)

        if not src_path.exists():
            raise FileNotFoundError(f"Source file not found: {source}")

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)

    async def move_file(self, source: str, destination: str) -> None:
        src_path = self._full_path(source)
        dst_path = self._full_path(destination)

        if not src_path.exists():
            raise FileNotFoundError(f"Source file not found: {source}")

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(src_path, dst_path)

    async def copy_directory(self, source: str, destination: str) -> None:
        src_path = self._full_path(source)
        dst_path = self._full_path(destination)

        if not src_path.exists():
            raise FileNotFoundError(f"Source directory not found: {source}")

        shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
