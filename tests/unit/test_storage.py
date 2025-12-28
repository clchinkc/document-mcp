"""Unit tests for storage abstraction layer."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from document_mcp.storage import StorageType
from document_mcp.storage import get_storage
from document_mcp.storage.factory import create_storage_backend
from document_mcp.storage.factory import detect_environment
from document_mcp.storage.factory import reset_storage
from document_mcp.storage.local import LocalStorageBackend


class TestEnvironmentDetection:
    """Tests for storage backend environment detection."""

    def test_detect_local_by_default(self):
        """Should detect local storage when no cloud env vars set."""
        with patch.dict(os.environ, {}, clear=True):
            # Ensure no GCS/Cloud Run env vars
            os.environ.pop("K_SERVICE", None)
            os.environ.pop("GCS_BUCKET", None)
            os.environ.pop("STORAGE_BACKEND", None)
            assert detect_environment() == StorageType.LOCAL

    def test_detect_gcs_with_bucket(self):
        """Should detect GCS when GCS_BUCKET is set."""
        with patch.dict(os.environ, {"GCS_BUCKET": "my-bucket"}, clear=False):
            assert detect_environment() == StorageType.GCS

    def test_detect_gcs_on_cloud_run(self):
        """Should detect GCS on Cloud Run with bucket configured."""
        with patch.dict(
            os.environ,
            {"K_SERVICE": "document-mcp", "GCS_BUCKET": "my-bucket"},
            clear=False,
        ):
            assert detect_environment() == StorageType.GCS

    def test_explicit_local_override(self):
        """Should use local when explicitly set."""
        with patch.dict(
            os.environ,
            {"STORAGE_BACKEND": "local", "GCS_BUCKET": "my-bucket"},
            clear=False,
        ):
            assert detect_environment() == StorageType.LOCAL

    def test_explicit_gcs_override(self):
        """Should use GCS when explicitly set."""
        with patch.dict(os.environ, {"STORAGE_BACKEND": "gcs"}, clear=False):
            assert detect_environment() == StorageType.GCS


class TestLocalStorageBackend:
    """Tests for local filesystem storage backend."""

    @pytest.fixture
    def temp_storage(self):
        """Create a temporary storage directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LocalStorageBackend(root_dir=tmpdir)
            yield storage, tmpdir

    @pytest.mark.asyncio
    async def test_backend_type(self, temp_storage):
        """Should report correct backend type."""
        storage, _ = temp_storage
        assert storage.backend_type == "local"

    @pytest.mark.asyncio
    async def test_write_and_read_file(self, temp_storage):
        """Should write and read files correctly."""
        storage, _ = temp_storage

        await storage.write_file("test_doc/chapter.md", "# Hello World")
        content = await storage.read_file("test_doc/chapter.md")

        assert content == "# Hello World"

    @pytest.mark.asyncio
    async def test_file_exists(self, temp_storage):
        """Should correctly detect file existence."""
        storage, _ = temp_storage

        assert not await storage.file_exists("nonexistent.md")

        await storage.write_file("exists.md", "content")
        assert await storage.file_exists("exists.md")

    @pytest.mark.asyncio
    async def test_delete_file(self, temp_storage):
        """Should delete files correctly."""
        storage, _ = temp_storage

        await storage.write_file("to_delete.md", "content")
        assert await storage.file_exists("to_delete.md")

        result = await storage.delete_file("to_delete.md")
        assert result is True
        assert not await storage.file_exists("to_delete.md")

        # Deleting non-existent file should return False
        result = await storage.delete_file("nonexistent.md")
        assert result is False

    @pytest.mark.asyncio
    async def test_list_directory(self, temp_storage):
        """Should list directory contents."""
        storage, _ = temp_storage

        await storage.write_file("doc1/chapter1.md", "content")
        await storage.write_file("doc1/chapter2.md", "content")
        await storage.write_file("doc2/chapter1.md", "content")

        # List root
        root_entries = await storage.list_directory()
        assert set(root_entries) == {"doc1", "doc2"}

        # List subdirectory
        doc1_entries = await storage.list_directory("doc1")
        assert set(doc1_entries) == {"chapter1.md", "chapter2.md"}

    @pytest.mark.asyncio
    async def test_list_files_with_pattern(self, temp_storage):
        """Should list files matching pattern."""
        storage, _ = temp_storage

        await storage.write_file("doc/chapter.md", "content")
        await storage.write_file("doc/metadata.yaml", "key: value")
        await storage.write_file("doc/notes.txt", "notes")

        md_files = await storage.list_files("doc", "*.md")
        assert md_files == ["doc/chapter.md"]

        yaml_files = await storage.list_files("doc", "*.yaml")
        assert yaml_files == ["doc/metadata.yaml"]

    @pytest.mark.asyncio
    async def test_directory_operations(self, temp_storage):
        """Should handle directory operations."""
        storage, _ = temp_storage

        assert not await storage.directory_exists("new_doc")

        await storage.create_directory("new_doc")
        assert await storage.directory_exists("new_doc")

        # Non-recursive delete of non-empty directory should fail
        await storage.write_file("new_doc/file.md", "content")
        result = await storage.delete_directory("new_doc", recursive=False)
        assert result is False

        # Recursive delete should work
        result = await storage.delete_directory("new_doc", recursive=True)
        assert result is True
        assert not await storage.directory_exists("new_doc")

    @pytest.mark.asyncio
    async def test_copy_file(self, temp_storage):
        """Should copy files correctly."""
        storage, _ = temp_storage

        await storage.write_file("source.md", "original content")
        await storage.copy_file("source.md", "destination.md")

        assert await storage.file_exists("source.md")
        assert await storage.file_exists("destination.md")
        assert await storage.read_file("destination.md") == "original content"

    @pytest.mark.asyncio
    async def test_move_file(self, temp_storage):
        """Should move files correctly."""
        storage, _ = temp_storage

        await storage.write_file("old_name.md", "content")
        await storage.move_file("old_name.md", "new_name.md")

        assert not await storage.file_exists("old_name.md")
        assert await storage.file_exists("new_name.md")
        assert await storage.read_file("new_name.md") == "content"

    @pytest.mark.asyncio
    async def test_get_file_info(self, temp_storage):
        """Should return file metadata."""
        storage, _ = temp_storage

        await storage.write_file("info_test.md", "# Heading\n\nParagraph content")
        info = await storage.get_file_info("info_test.md")

        assert info is not None
        assert info.path == "info_test.md"
        assert info.size > 0
        assert info.content_type == "text/markdown"
        assert info.is_directory is False

    @pytest.mark.asyncio
    async def test_file_not_found(self, temp_storage):
        """Should raise FileNotFoundError for missing files."""
        storage, _ = temp_storage

        with pytest.raises(FileNotFoundError):
            await storage.read_file("nonexistent.md")


class TestStorageFactory:
    """Tests for storage backend factory."""

    def test_create_local_backend(self):
        """Should create local backend."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = create_storage_backend(
                storage_type=StorageType.LOCAL,
                root_dir=tmpdir,
            )
            assert backend.backend_type == "local"
            # Compare resolved paths (handles macOS /var -> /private/var symlink)
            assert Path(backend.root_path).resolve() == Path(tmpdir).resolve()

    def test_create_gcs_backend_requires_bucket(self):
        """Should require bucket name for GCS backend."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("GCS_BUCKET", None)
            with pytest.raises(ValueError, match="bucket name required"):
                create_storage_backend(storage_type=StorageType.GCS)

    def test_singleton_pattern(self):
        """Should return same instance on repeated calls."""
        reset_storage()

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"DOCUMENT_ROOT_DIR": tmpdir}):
                storage1 = get_storage()
                storage2 = get_storage()
                assert storage1 is storage2

        reset_storage()  # Clean up


class TestSyncWrappers:
    """Tests for synchronous wrapper methods."""

    def test_read_file_sync(self):
        """Should read file synchronously."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LocalStorageBackend(root_dir=tmpdir)

            # Write file first
            Path(tmpdir, "test.md").write_text("sync content")

            content = storage.read_file_sync("test.md")
            assert content == "sync content"

    def test_file_exists_sync(self):
        """Should check file existence synchronously."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LocalStorageBackend(root_dir=tmpdir)

            assert not storage.file_exists_sync("missing.md")

            Path(tmpdir, "exists.md").write_text("content")
            assert storage.file_exists_sync("exists.md")
