"""Unit tests for embedding cache functionality.

Tests the EmbeddingCache class with mocked file system operations
to ensure proper caching behavior without external dependencies.
"""

import datetime
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from document_mcp.models import ChapterEmbeddingManifest
from document_mcp.models import EmbeddingCacheEntry
from document_mcp.utils.embedding_cache import EmbeddingCache


class TestEmbeddingCache:
    """Test suite for EmbeddingCache functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.cache = EmbeddingCache("models/text-embedding-004")
        self.test_document = "test_document"
        self.test_chapter = "01-test.md"

        # Create test embeddings
        self.test_embeddings = {
            0: np.array([1.0, 0.0, 0.0]),
            1: np.array([0.0, 1.0, 0.0]),
            2: np.array([0.0, 0.0, 1.0]),
        }

        self.test_contents = {
            0: "First paragraph content",
            1: "Second paragraph content",
            2: "Third paragraph content",
        }

    @patch("document_mcp.utils.embedding_cache._get_chapter_embeddings_path")
    @patch("document_mcp.utils.embedding_cache._get_chapter_path")
    def test_get_chapter_embeddings_no_cache(self, mock_chapter_path, mock_embeddings_path):
        """Test getting embeddings when no cache exists."""
        # Setup mocks
        mock_embeddings_path.return_value = Path("/fake/embeddings/path")
        mock_chapter_path.return_value = Path("/fake/chapter/path.md")

        # Mock manifest file not existing
        with patch("pathlib.Path.exists", return_value=False):
            result = self.cache.get_chapter_embeddings(self.test_document, self.test_chapter)

        assert result == {}

    @patch("document_mcp.utils.embedding_cache._get_chapter_embeddings_path")
    @patch("document_mcp.utils.embedding_cache._get_chapter_path")
    def test_get_chapter_embeddings_invalid_cache(self, mock_chapter_path, mock_embeddings_path):
        """Test getting embeddings when cache is older than source file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup paths
            embeddings_dir = Path(temp_dir) / "embeddings"
            embeddings_dir.mkdir()
            chapter_dir = embeddings_dir / self.test_chapter
            chapter_dir.mkdir()

            mock_embeddings_path.return_value = chapter_dir

            # Create a source file that's newer
            source_file = Path(temp_dir) / "source.md"
            source_file.write_text("content")
            mock_chapter_path.return_value = source_file

            # Create manifest file that's older
            manifest_path = chapter_dir / "manifest.json"
            old_time = datetime.datetime.now() - datetime.timedelta(hours=1)
            manifest = ChapterEmbeddingManifest(
                chapter_name=self.test_chapter,
                total_paragraphs=1,
                cache_entries=[],
                last_updated=old_time,
            )
            manifest_path.write_text(manifest.model_dump_json())

            result = self.cache.get_chapter_embeddings(self.test_document, self.test_chapter)
            assert result == {}

    @patch("document_mcp.utils.embedding_cache._get_chapter_embeddings_path")
    @patch("document_mcp.utils.embedding_cache._get_chapter_path")
    def test_get_chapter_embeddings_valid_cache(self, mock_chapter_path, mock_embeddings_path):
        """Test getting embeddings when cache is valid and newer than source."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup paths
            embeddings_dir = Path(temp_dir) / "embeddings"
            embeddings_dir.mkdir()
            chapter_dir = embeddings_dir / self.test_chapter
            chapter_dir.mkdir()

            mock_embeddings_path.return_value = chapter_dir

            # Create a source file
            source_file = Path(temp_dir) / "source.md"
            source_file.write_text("content")
            mock_chapter_path.return_value = source_file

            # Wait a moment to ensure proper timing
            import time

            time.sleep(0.01)

            # Create embedding files (np.save adds .npy extension automatically)
            for idx, embedding in self.test_embeddings.items():
                embedding_file = chapter_dir / f"paragraph_{idx}"
                np.save(str(embedding_file), embedding)

            # Create manifest file that's newer than source - use current time + a bit
            manifest_path = chapter_dir / "manifest.json"
            new_time = datetime.datetime.now() + datetime.timedelta(seconds=1)
            cache_entries = [
                EmbeddingCacheEntry(
                    content_hash="hash123",
                    paragraph_index=idx,
                    model_version="models/text-embedding-004",
                    created_at=new_time,
                    file_modified_time=new_time,
                )
                for idx in self.test_embeddings
            ]
            manifest = ChapterEmbeddingManifest(
                chapter_name=self.test_chapter,
                total_paragraphs=len(self.test_embeddings),
                cache_entries=cache_entries,
                last_updated=new_time,
            )
            manifest_path.write_text(manifest.model_dump_json())

            result = self.cache.get_chapter_embeddings(self.test_document, self.test_chapter)

            # Verify all embeddings were loaded
            assert len(result) == len(self.test_embeddings)
            for idx, expected_embedding in self.test_embeddings.items():
                assert idx in result
                np.testing.assert_array_equal(result[idx], expected_embedding)

    @patch("document_mcp.utils.embedding_cache._get_chapter_embeddings_path")
    @patch("document_mcp.utils.embedding_cache._get_chapter_path")
    def test_store_chapter_embeddings(self, mock_chapter_path, mock_embeddings_path):
        """Test storing embeddings to cache."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup paths - this will be the chapter-specific directory
            embeddings_root = Path(temp_dir) / "embeddings"
            chapter_dir = embeddings_root / self.test_chapter
            mock_embeddings_path.return_value = chapter_dir

            # Mock source file
            source_file = Path(temp_dir) / "source.md"
            source_file.write_text("content")
            mock_chapter_path.return_value = source_file

            # Store embeddings
            self.cache.store_chapter_embeddings(
                self.test_document,
                self.test_chapter,
                self.test_embeddings,
                self.test_contents,
            )

            # Verify directory was created
            assert chapter_dir.exists()

            # Verify embedding files were created (.npy extension added by np.save)
            for idx in self.test_embeddings:
                embedding_file = chapter_dir / f"paragraph_{idx}.npy"
                assert embedding_file.exists()

                # Load and verify embedding
                loaded_embedding = np.load(str(embedding_file))
                np.testing.assert_array_equal(loaded_embedding, self.test_embeddings[idx])

            # Verify manifest was created
            manifest_file = chapter_dir / "manifest.json"
            assert manifest_file.exists()

            # Load and verify manifest
            manifest_data = manifest_file.read_text()
            manifest = ChapterEmbeddingManifest.model_validate_json(manifest_data)

            assert manifest.chapter_name == self.test_chapter
            assert manifest.total_paragraphs == len(self.test_embeddings)
            assert len(manifest.cache_entries) == len(self.test_embeddings)

            for entry in manifest.cache_entries:
                assert entry.model_version == "models/text-embedding-004"
                assert entry.paragraph_index in self.test_embeddings

    @patch("document_mcp.utils.embedding_cache._get_chapter_embeddings_path")
    def test_invalidate_chapter_cache(self, mock_embeddings_path):
        """Test invalidating chapter cache."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create cache directory with files
            chapter_dir = Path(temp_dir) / "embeddings"
            chapter_dir.mkdir(parents=True)
            mock_embeddings_path.return_value = chapter_dir

            # Create some cache files
            (chapter_dir / "manifest.json").write_text("{}")
            (chapter_dir / "paragraph_0.emb").write_text("fake embedding")
            (chapter_dir / "paragraph_1.emb").write_text("fake embedding")

            assert chapter_dir.exists()
            assert len(list(chapter_dir.iterdir())) == 3

            # Invalidate cache
            self.cache.invalidate_chapter_cache(self.test_document, self.test_chapter)

            # Verify directory and files were removed
            assert not chapter_dir.exists()

    @patch("document_mcp.utils.embedding_cache._get_embeddings_path")
    def test_invalidate_document_cache(self, mock_embeddings_path):
        """Test invalidating entire document cache."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create cache directory structure
            embeddings_dir = Path(temp_dir) / "embeddings"
            embeddings_dir.mkdir(parents=True)
            mock_embeddings_path.return_value = embeddings_dir

            # Create chapter directories with files
            chapter1_dir = embeddings_dir / "chapter1.md"
            chapter1_dir.mkdir()
            (chapter1_dir / "manifest.json").write_text("{}")
            (chapter1_dir / "paragraph_0.emb").write_text("fake")

            chapter2_dir = embeddings_dir / "chapter2.md"
            chapter2_dir.mkdir()
            (chapter2_dir / "manifest.json").write_text("{}")
            (chapter2_dir / "paragraph_0.emb").write_text("fake")

            assert embeddings_dir.exists()
            assert len(list(embeddings_dir.iterdir())) == 2

            # Invalidate document cache
            self.cache.invalidate_document_cache(self.test_document)

            # Verify all chapter directories were removed
            assert not embeddings_dir.exists()

    @patch("document_mcp.utils.embedding_cache._get_chapter_embeddings_path")
    @patch("document_mcp.utils.embedding_cache._get_chapter_path")
    def test_is_cache_valid_logic(self, mock_chapter_path, mock_embeddings_path):
        """Test cache validity logic directly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup paths
            chapter_dir = Path(temp_dir) / "embeddings"
            chapter_dir.mkdir(parents=True)
            mock_embeddings_path.return_value = chapter_dir

            # Create source file
            source_file = Path(temp_dir) / "source.md"
            source_file.write_text("content")
            mock_chapter_path.return_value = source_file

            # Test 1: No manifest file
            assert not self.cache._is_cache_valid(self.test_document, self.test_chapter)

            # Test 2: Manifest newer than source (valid)
            manifest_path = chapter_dir / "manifest.json"
            future_time = datetime.datetime.now() + datetime.timedelta(minutes=1)
            manifest = ChapterEmbeddingManifest(
                chapter_name=self.test_chapter,
                total_paragraphs=1,
                cache_entries=[],
                last_updated=future_time,
            )
            manifest_path.write_text(manifest.model_dump_json())

            assert self.cache._is_cache_valid(self.test_document, self.test_chapter)

            # Test 3: Manifest older than source (invalid)
            past_time = datetime.datetime.now() - datetime.timedelta(hours=1)
            manifest.last_updated = past_time
            manifest_path.write_text(manifest.model_dump_json())

            assert not self.cache._is_cache_valid(self.test_document, self.test_chapter)

    def test_model_version_compatibility(self):
        """Test that cache respects model version compatibility."""
        cache_v1 = EmbeddingCache("models/text-embedding-003")
        cache_v2 = EmbeddingCache("models/text-embedding-004")

        assert cache_v1.model_version == "models/text-embedding-003"
        assert cache_v2.model_version == "models/text-embedding-004"

        # Different model versions should be treated as different caches
        assert cache_v1.model_version != cache_v2.model_version
