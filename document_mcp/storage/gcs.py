"""Google Cloud Storage Backend.

Implements the StorageBackend interface using Google Cloud Storage.
Used when running on GCP (Cloud Run, GCE, etc.).
"""
from __future__ import annotations


import fnmatch
import os
from datetime import datetime
from datetime import timezone

from .base import FileInfo
from .base import StorageBackend


class GCSStorageBackend(StorageBackend):
    """Storage backend using Google Cloud Storage.

    Args:
        bucket_name: GCS bucket name. Defaults to GCS_BUCKET env var.
        prefix: Optional prefix for all paths (e.g., "documents/")
    """

    def __init__(self, bucket_name: str | None = None, prefix: str = ""):
        self._bucket_name = bucket_name or os.environ.get("GCS_BUCKET", "")
        self._prefix = prefix.strip("/")

        if not self._bucket_name:
            raise ValueError(
                "GCS bucket name required. Set GCS_BUCKET environment variable or pass bucket_name parameter."
            )

        # Lazy import to avoid dependency when using local storage
        try:
            from google.cloud import storage

            self._client = storage.Client()
            self._bucket = self._client.bucket(self._bucket_name)
        except ImportError as e:
            raise ImportError(
                "google-cloud-storage package required for GCS backend. "
                "Install with: pip install google-cloud-storage"
            ) from e

    @property
    def backend_type(self) -> str:
        return "gcs"

    @property
    def root_path(self) -> str:
        if self._prefix:
            return f"gs://{self._bucket_name}/{self._prefix}"
        return f"gs://{self._bucket_name}"

    def _full_path(self, path: str) -> str:
        """Convert relative path to GCS object path."""
        if self._prefix:
            return f"{self._prefix}/{path}".strip("/")
        return path.strip("/")

    # === File Operations ===

    async def read_file(self, path: str) -> str:
        blob = self._bucket.blob(self._full_path(path))
        if not blob.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return blob.download_as_text(encoding="utf-8")

    async def write_file(self, path: str, content: str) -> None:
        blob = self._bucket.blob(self._full_path(path))
        blob.upload_from_string(
            content,
            content_type=self._get_content_type(path),
        )

    async def delete_file(self, path: str) -> bool:
        blob = self._bucket.blob(self._full_path(path))
        if blob.exists():
            blob.delete()
            return True
        return False

    async def file_exists(self, path: str) -> bool:
        blob = self._bucket.blob(self._full_path(path))
        return blob.exists()

    async def get_file_info(self, path: str) -> FileInfo | None:
        blob = self._bucket.blob(self._full_path(path))
        if not blob.exists():
            return None

        # Reload to get metadata
        blob.reload()

        return FileInfo(
            path=path,
            size=blob.size or 0,
            last_modified=blob.updated or datetime.now(tz=timezone.utc),
            is_directory=False,  # GCS doesn't have real directories
            content_type=blob.content_type or "application/octet-stream",
        )

    def _get_content_type(self, path: str) -> str:
        """Determine content type from file extension."""
        ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
        content_types = {
            "md": "text/markdown",
            "yaml": "text/yaml",
            "yml": "text/yaml",
            "json": "application/json",
            "txt": "text/plain",
        }
        return content_types.get(ext, "application/octet-stream")

    # === Directory Operations ===

    async def list_directory(self, path: str = "") -> list[str]:
        """List entries in a 'directory' (prefix in GCS).

        GCS doesn't have real directories, so we list unique prefixes
        at the given level.
        """
        prefix = self._full_path(path)
        if prefix and not prefix.endswith("/"):
            prefix += "/"

        # Use delimiter to get 'directory-like' listing
        blobs = self._bucket.list_blobs(prefix=prefix, delimiter="/")

        entries = set()

        # Get file names at this level
        for blob in blobs:
            name = blob.name[len(prefix) :]
            if name and "/" not in name and not name.startswith("."):
                entries.add(name)

        # Get 'subdirectory' names from prefixes
        for prefix_path in blobs.prefixes:
            name = prefix_path[len(prefix) :].rstrip("/")
            if name and not name.startswith("."):
                entries.add(name)

        return sorted(entries)

    async def list_files(self, path: str = "", pattern: str = "*.md") -> list[str]:
        """List files matching a pattern in a 'directory'."""
        prefix = self._full_path(path)
        if prefix and not prefix.endswith("/"):
            prefix += "/"

        blobs = self._bucket.list_blobs(prefix=prefix)

        files = []
        for blob in blobs:
            # Get filename relative to the prefix
            name = blob.name[len(prefix) :] if prefix else blob.name

            # Skip 'directory' markers and hidden files
            if not name or "/" in name or name.startswith("."):
                continue

            if fnmatch.fnmatch(name, pattern):
                if path:
                    files.append(f"{path}/{name}")
                else:
                    files.append(name)

        return sorted(files)

    async def directory_exists(self, path: str) -> bool:
        """Check if a 'directory' exists (has any blobs with prefix)."""
        prefix = self._full_path(path)
        if not prefix.endswith("/"):
            prefix += "/"

        # Check if any blob exists with this prefix
        blobs = self._bucket.list_blobs(prefix=prefix, max_results=1)
        return any(True for _ in blobs)

    async def create_directory(self, path: str) -> None:
        """Create a 'directory' by creating a placeholder blob.

        GCS doesn't have real directories, but we can create
        a zero-byte placeholder to represent an empty directory.
        """
        # In GCS, directories are virtual - they exist when files exist
        # We don't need to explicitly create them, but we can create
        # a placeholder if needed for compatibility
        pass

    async def delete_directory(self, path: str, recursive: bool = False) -> bool:
        """Delete a 'directory' (all blobs with prefix)."""
        prefix = self._full_path(path)
        if not prefix.endswith("/"):
            prefix += "/"

        blobs = list(self._bucket.list_blobs(prefix=prefix))

        if not blobs:
            return False

        if not recursive and len(blobs) > 0:
            # Check if there are actual files (not just directory markers)
            real_files = [b for b in blobs if not b.name.endswith("/")]
            if real_files:
                return False

        # Delete all blobs with this prefix
        for blob in blobs:
            blob.delete()

        return True

    # === Bulk Operations ===

    async def copy_file(self, source: str, destination: str) -> None:
        src_blob = self._bucket.blob(self._full_path(source))
        if not src_blob.exists():
            raise FileNotFoundError(f"Source file not found: {source}")

        dst_blob = self._bucket.blob(self._full_path(destination))
        self._bucket.copy_blob(src_blob, self._bucket, dst_blob.name)

    async def move_file(self, source: str, destination: str) -> None:
        await self.copy_file(source, destination)
        await self.delete_file(source)

    async def copy_directory(self, source: str, destination: str) -> None:
        """Copy all blobs with source prefix to destination prefix."""
        src_prefix = self._full_path(source)
        if not src_prefix.endswith("/"):
            src_prefix += "/"

        dst_prefix = self._full_path(destination)
        if not dst_prefix.endswith("/"):
            dst_prefix += "/"

        blobs = self._bucket.list_blobs(prefix=src_prefix)

        for blob in blobs:
            # Calculate new name
            relative = blob.name[len(src_prefix) :]
            new_name = dst_prefix + relative

            new_blob = self._bucket.blob(new_name)
            self._bucket.copy_blob(blob, self._bucket, new_blob.name)
