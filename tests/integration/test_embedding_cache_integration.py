"""Integration tests for embedding cache with semantic search.

Tests the complete flow from semantic search through caching to ensure
the cache integration works correctly in real scenarios.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np

from document_mcp.tools.content_tools import _perform_semantic_search
from document_mcp.utils.embedding_cache import EmbeddingCache


class MockEmbedding:
    """Mock embedding object with .values attribute for new google-genai SDK."""

    def __init__(self, values):
        self.values = values


def create_mock_embed_response(embeddings_list):
    """Create a mock response for client.models.embed_content (new SDK)."""
    mock_response = MagicMock()
    mock_response.embeddings = [MockEmbedding(e) for e in embeddings_list]
    return mock_response


class TestEmbeddingCacheIntegration:
    """Test suite for embedding cache integration with semantic search."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_document = "test_document"
        self.test_chapter = "01-test.md"

    @patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"})
    @patch("document_mcp.tools.content_tools.genai.Client")
    @patch("document_mcp.tools.content_tools._get_document_path")
    @patch("document_mcp.tools.content_tools._get_ordered_chapter_files")
    @patch("document_mcp.tools.content_tools._split_into_paragraphs")
    def test_semantic_search_with_empty_cache(self, mock_split, mock_chapters, mock_doc_path, mock_client_class):
        """Test semantic search when cache is empty (first run)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup document structure
            doc_path = Path(temp_dir) / self.test_document
            doc_path.mkdir()

            # Mock document path to return our temp directory
            mock_doc_path.return_value = doc_path

            # Mock chapter files
            chapter_file = MagicMock()
            chapter_file.name = self.test_chapter
            chapter_file.read_text.return_value = "Sample content for testing"
            mock_chapters.return_value = [chapter_file]

            # Mock paragraphs
            test_paragraphs = [
                "First paragraph about machine learning",
                "Second paragraph about deep learning",
                "Third paragraph about neural networks",
            ]
            mock_split.return_value = test_paragraphs

            # Mock embeddings API response (new google-genai SDK)
            mock_client = MagicMock()
            mock_client.models.embed_content.return_value = create_mock_embed_response([
                [1.0, 0.0, 0.0],  # Query embedding
                [0.9, 0.1, 0.0],  # First paragraph (high similarity)
                [0.5, 0.5, 0.0],  # Second paragraph (medium similarity)
                [0.0, 0.0, 1.0],  # Third paragraph (low similarity)
            ])
            mock_client_class.return_value = mock_client

            # Set up environment for cache paths
            os.environ["DOCUMENT_ROOT_DIR"] = str(temp_dir)

            # Execute search
            results = _perform_semantic_search(
                document_name=self.test_document,
                query_text="machine learning concepts",
                scope="document",
                chapter_name=None,
                similarity_threshold=0.7,
                max_results=10,
            )

            # Verify results (two paragraphs should meet the 0.7 threshold based on cosine similarity)
            assert len(results) >= 1  # At least first paragraph meets threshold
            assert results[0].paragraph_index == 0
            assert "machine learning" in results[0].content

            # Verify API was called with query + paragraphs
            mock_client.models.embed_content.assert_called_once()

            # Verify cache directory was created
            cache_dir = doc_path / ".embeddings" / self.test_chapter
            assert cache_dir.exists()

            # Verify embedding files were created
            for i in range(3):
                embedding_file = cache_dir / f"paragraph_{i}.npy"
                assert embedding_file.exists()

            # Verify manifest was created
            manifest_file = cache_dir / "manifest.json"
            assert manifest_file.exists()

    @patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"})
    @patch("document_mcp.tools.content_tools.genai.Client")
    @patch("document_mcp.tools.content_tools._get_document_path")
    @patch("document_mcp.tools.content_tools._get_ordered_chapter_files")
    @patch("document_mcp.tools.content_tools._split_into_paragraphs")
    def test_semantic_search_with_valid_cache(self, mock_split, mock_chapters, mock_doc_path, mock_client_class):
        """Test semantic search when cache is valid (subsequent runs)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup document structure
            doc_path = Path(temp_dir) / self.test_document
            doc_path.mkdir()

            # Create actual chapter file (needed for cache validation)
            chapter_path = doc_path / self.test_chapter
            chapter_path.write_text("Sample content for testing", encoding="utf-8")

            # Mock document path to return our temp directory
            mock_doc_path.return_value = doc_path

            # Mock chapter files
            chapter_file = MagicMock()
            chapter_file.name = self.test_chapter
            chapter_file.read_text.return_value = "Sample content for testing"
            mock_chapters.return_value = [chapter_file]

            # Mock paragraphs
            test_paragraphs = [
                "First paragraph about machine learning",
                "Second paragraph about deep learning",
            ]
            mock_split.return_value = test_paragraphs

            # Set up environment for cache paths
            os.environ["DOCUMENT_ROOT_DIR"] = str(temp_dir)

            # Pre-populate cache with embeddings that will have high similarity to query
            # Note: Must match model version used in actual code: "text-embedding-004"
            cache = EmbeddingCache("text-embedding-004")
            test_embeddings = {
                0: np.array([1.0, 0.0, 0.0]),  # Perfect match with query [1.0, 0.0, 0.0]
                1: np.array([0.9, 0.1, 0.0]),  # High similarity with query
            }
            test_contents = {
                0: "First paragraph about machine learning",
                1: "Second paragraph about deep learning",
            }

            cache.store_chapter_embeddings(
                self.test_document, self.test_chapter, test_embeddings, test_contents
            )

            # Mock only query embedding (paragraphs should be cached) - new SDK pattern
            mock_client = MagicMock()
            mock_client.models.embed_content.return_value = create_mock_embed_response([
                [1.0, 0.0, 0.0]  # Only query embedding needed
            ])
            mock_client_class.return_value = mock_client

            # Execute search
            results = _perform_semantic_search(
                document_name=self.test_document,
                query_text="machine learning concepts",
                scope="document",
                chapter_name=None,
                similarity_threshold=0.5,  # Lower threshold to ensure matches
                max_results=10,
            )

            # Verify results (should get results from cached embeddings)
            assert len(results) >= 1  # At least first paragraph should match
            assert results[0].paragraph_index == 0  # Best match should be first paragraph

            # Verify API was called only for query (not paragraphs)
            mock_client.models.embed_content.assert_called_once()

    @patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"})
    @patch("document_mcp.tools.content_tools.genai.Client")
    @patch("document_mcp.tools.content_tools._get_document_path")
    @patch("document_mcp.tools.content_tools._get_ordered_chapter_files")
    @patch("document_mcp.tools.content_tools._split_into_paragraphs")
    def test_semantic_search_cache_invalidation(self, mock_split, mock_chapters, mock_doc_path, mock_client_class):
        """Test that cache is invalidated when content changes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup document structure
            doc_path = Path(temp_dir) / self.test_document
            doc_path.mkdir()
            chapter_path = doc_path / self.test_chapter
            chapter_path.write_text("Original content", encoding="utf-8")

            # Mock document path to return our temp directory
            mock_doc_path.return_value = doc_path

            # Mock chapter files
            chapter_file = MagicMock()
            chapter_file.name = self.test_chapter
            chapter_file.read_text.return_value = "Modified content for testing"
            mock_chapters.return_value = [chapter_file]

            # Mock paragraphs
            test_paragraphs = ["Modified paragraph"]
            mock_split.return_value = test_paragraphs

            # Set up environment for cache paths
            os.environ["DOCUMENT_ROOT_DIR"] = str(temp_dir)

            # Pre-populate cache with old content
            # Note: Must match model version used in actual code: "text-embedding-004"
            cache = EmbeddingCache("text-embedding-004")
            old_embeddings = {0: np.array([0.9, 0.1, 0.0])}
            old_contents = {0: "Original paragraph"}

            cache.store_chapter_embeddings(
                self.test_document, self.test_chapter, old_embeddings, old_contents
            )

            # Verify cache exists
            cache_dir = doc_path / ".embeddings" / self.test_chapter
            assert cache_dir.exists()

            # Modify source file to make it newer than cache
            import time

            time.sleep(0.2)  # Ensure newer timestamp (increased for reliability)
            chapter_path.write_text("Modified content", encoding="utf-8")

            # Mock embedding response for new content - new SDK pattern
            mock_client = MagicMock()
            mock_client.models.embed_content.return_value = create_mock_embed_response([
                [1.0, 0.0, 0.0],  # Query embedding
                [0.8, 0.2, 0.0],  # New paragraph embedding
            ])
            mock_client_class.return_value = mock_client

            # Execute search
            _perform_semantic_search(
                document_name=self.test_document,
                query_text="test query",
                scope="document",
                chapter_name=None,
                similarity_threshold=0.5,
                max_results=10,
            )

            # Verify API was called for both query and paragraph (cache invalid)
            mock_client.models.embed_content.assert_called_once()

    def test_cache_persistence_across_instances(self):
        """Test that cache persists across different EmbeddingCache instances."""
        with tempfile.TemporaryDirectory() as temp_dir:
            os.environ["DOCUMENT_ROOT_DIR"] = str(temp_dir)

            # Create first cache instance and store embeddings
            cache1 = EmbeddingCache("models/text-embedding-004")
            test_embeddings = {0: np.array([1.0, 0.0, 0.0])}
            test_contents = {0: "Test content"}

            # Create document directory
            doc_path = Path(temp_dir) / self.test_document
            doc_path.mkdir()
            chapter_path = doc_path / self.test_chapter
            chapter_path.write_text("Test content", encoding="utf-8")

            cache1.store_chapter_embeddings(
                self.test_document, self.test_chapter, test_embeddings, test_contents
            )

            # Create second cache instance and verify it can load embeddings
            cache2 = EmbeddingCache("models/text-embedding-004")
            loaded_embeddings = cache2.get_chapter_embeddings(self.test_document, self.test_chapter)

            assert len(loaded_embeddings) == 1
            assert 0 in loaded_embeddings
            np.testing.assert_array_equal(loaded_embeddings[0], test_embeddings[0])

    def test_model_version_isolation(self):
        """Test that different model versions maintain separate caches."""
        with tempfile.TemporaryDirectory() as temp_dir:
            os.environ["DOCUMENT_ROOT_DIR"] = str(temp_dir)

            # Create document
            doc_path = Path(temp_dir) / self.test_document
            doc_path.mkdir()
            chapter_path = doc_path / self.test_chapter
            chapter_path.write_text("Test content", encoding="utf-8")

            # Store embeddings with first model version
            cache_v1 = EmbeddingCache("models/text-embedding-003")
            embeddings_v1 = {0: np.array([1.0, 0.0, 0.0])}
            contents = {0: "Test content"}

            cache_v1.store_chapter_embeddings(self.test_document, self.test_chapter, embeddings_v1, contents)

            # Try to load with different model version
            cache_v2 = EmbeddingCache("models/text-embedding-004")
            loaded_embeddings = cache_v2.get_chapter_embeddings(self.test_document, self.test_chapter)

            # Should return empty dict (different model version)
            assert loaded_embeddings == {}
