"""Unit tests for semantic search functionality.

These tests focus on the core logic of the semantic search implementation,
with mocked external dependencies (API calls).
"""

from unittest.mock import MagicMock
from unittest.mock import patch

from document_mcp.tools.content_tools import _generate_context_snippet
from document_mcp.tools.content_tools import _perform_semantic_search


class MockEmbedding:
    """Mock embedding object with .values attribute."""

    def __init__(self, values):
        self.values = values


def create_mock_embed_response(embeddings_list):
    """Create a mock response for client.models.embed_content."""
    mock_response = MagicMock()
    mock_response.embeddings = [MockEmbedding(e) for e in embeddings_list]
    return mock_response


class TestSemanticSearchCore:
    """Test suite for semantic search core functionality."""

    @patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"})
    @patch("document_mcp.tools.content_tools.genai.Client")
    @patch("document_mcp.tools.content_tools._get_document_path")
    @patch("document_mcp.tools.content_tools._get_ordered_chapter_files")
    @patch("document_mcp.tools.content_tools._split_into_paragraphs")
    def test_perform_semantic_search_document_scope(
        self, mock_split, mock_chapters, mock_doc_path, mock_client_class
    ):
        """Test semantic search with document scope."""
        # Setup mocks
        mock_doc_path.return_value.exists.return_value = True

        # Mock chapter files
        mock_chapter = MagicMock()
        mock_chapter.name = "01-test.md"
        mock_chapter.read_text.return_value = "Sample content"
        mock_chapters.return_value = [mock_chapter]

        # Mock paragraphs
        mock_split.return_value = ["First paragraph", "Second paragraph"]

        # Mock embeddings API response with new SDK structure
        mock_client = MagicMock()
        mock_client.models.embed_content.return_value = create_mock_embed_response(
            [
                [1.0, 0.0, 0.0],  # Query embedding
                [0.9, 0.1, 0.0],  # First paragraph (high similarity)
                [0.0, 0.0, 1.0],  # Second paragraph (low similarity)
            ]
        )
        mock_client_class.return_value = mock_client

        # Execute search
        results = _perform_semantic_search(
            document_name="test_doc",
            query_text="test query",
            scope="document",
            chapter_name=None,
            similarity_threshold=0.8,
            max_results=10,
        )

        # Verify results
        assert len(results) == 1  # Only one paragraph above threshold
        assert results[0].document_name == "test_doc"
        assert results[0].chapter_name == "01-test.md"
        assert results[0].content == "First paragraph"
        assert results[0].similarity_score > 0.8

    @patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"})
    @patch("document_mcp.tools.content_tools.genai.Client")
    @patch("document_mcp.tools.content_tools._get_chapter_path")
    @patch("document_mcp.tools.content_tools._split_into_paragraphs")
    def test_perform_semantic_search_chapter_scope(self, mock_split, mock_chapter_path, mock_client_class):
        """Test semantic search with chapter scope."""
        # Setup mocks
        mock_chapter_path.return_value.exists.return_value = True
        mock_chapter_path.return_value.read_text.return_value = "Chapter content"

        # Mock paragraphs
        mock_split.return_value = ["Chapter paragraph"]

        # Mock embeddings API response with high similarity
        mock_client = MagicMock()
        mock_client.models.embed_content.return_value = create_mock_embed_response(
            [
                [1.0, 0.0, 0.0],  # Query embedding
                [0.9, 0.1, 0.0],  # Paragraph embedding (high similarity)
            ]
        )
        mock_client_class.return_value = mock_client

        # Execute search
        results = _perform_semantic_search(
            document_name="test_doc",
            query_text="test query",
            scope="chapter",
            chapter_name="01-test.md",
            similarity_threshold=0.7,
            max_results=10,
        )

        # Verify results
        assert len(results) == 1
        assert results[0].chapter_name == "01-test.md"
        assert results[0].content == "Chapter paragraph"

    def test_perform_semantic_search_no_api_key(self):
        """Test semantic search fails gracefully without API key."""
        with patch.dict("os.environ", {}, clear=True):
            results = _perform_semantic_search(
                document_name="test_doc",
                query_text="test query",
                scope="document",
                chapter_name=None,
                similarity_threshold=0.7,
                max_results=10,
            )
            assert results == []

    @patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"})
    @patch("document_mcp.tools.content_tools._get_document_path")
    def test_perform_semantic_search_document_not_found(self, mock_doc_path):
        """Test semantic search with non-existent document."""
        mock_doc_path.return_value.exists.return_value = False

        results = _perform_semantic_search(
            document_name="nonexistent_doc",
            query_text="test query",
            scope="document",
            chapter_name=None,
            similarity_threshold=0.7,
            max_results=10,
        )

        assert results == []

    @patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"})
    @patch("document_mcp.tools.content_tools.genai.Client")
    @patch("document_mcp.tools.content_tools._get_document_path")
    @patch("document_mcp.tools.content_tools._get_ordered_chapter_files")
    @patch("document_mcp.tools.content_tools._split_into_paragraphs")
    def test_perform_semantic_search_similarity_threshold_filtering(
        self, mock_split, mock_chapters, mock_doc_path, mock_client_class
    ):
        """Test that similarity threshold properly filters results."""
        # Setup mocks
        mock_doc_path.return_value.exists.return_value = True

        mock_chapter = MagicMock()
        mock_chapter.name = "01-test.md"
        mock_chapter.read_text.return_value = "Sample content"
        mock_chapters.return_value = [mock_chapter]

        mock_split.return_value = ["Para 1", "Para 2", "Para 3"]

        # Mock embeddings with varying similarities
        mock_client = MagicMock()
        mock_client.models.embed_content.return_value = create_mock_embed_response(
            [
                [1.0, 0.0, 0.0],  # Query
                [0.9, 0.1, 0.0],  # Para 1: high similarity (~0.95)
                [0.5, 0.5, 0.0],  # Para 2: medium similarity (~0.71)
                [0.0, 1.0, 0.0],  # Para 3: low similarity (0.0)
            ]
        )
        mock_client_class.return_value = mock_client

        # Test with high threshold
        results = _perform_semantic_search(
            document_name="test_doc",
            query_text="test query",
            scope="document",
            chapter_name=None,
            similarity_threshold=0.9,
            max_results=10,
        )

        # Should only return the highest similarity paragraph
        assert len(results) == 1
        assert results[0].content == "Para 1"

    @patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"})
    @patch("document_mcp.tools.content_tools.genai.Client")
    @patch("document_mcp.tools.content_tools._get_document_path")
    @patch("document_mcp.tools.content_tools._get_ordered_chapter_files")
    @patch("document_mcp.tools.content_tools._split_into_paragraphs")
    def test_perform_semantic_search_max_results_limiting(
        self, mock_split, mock_chapters, mock_doc_path, mock_client_class
    ):
        """Test that max_results properly limits output."""
        # Setup mocks
        mock_doc_path.return_value.exists.return_value = True

        mock_chapter = MagicMock()
        mock_chapter.name = "01-test.md"
        mock_chapter.read_text.return_value = "Sample content"
        mock_chapters.return_value = [mock_chapter]

        mock_split.return_value = ["Para 1", "Para 2", "Para 3", "Para 4"]

        # Mock embeddings - all with high similarity
        mock_client = MagicMock()
        mock_client.models.embed_content.return_value = create_mock_embed_response(
            [
                [1.0, 0.0, 0.0],  # Query
                [0.95, 0.05, 0.0],  # Para 1
                [0.9, 0.1, 0.0],  # Para 2
                [0.85, 0.15, 0.0],  # Para 3
                [0.8, 0.2, 0.0],  # Para 4
            ]
        )
        mock_client_class.return_value = mock_client

        # Test with max_results=2
        results = _perform_semantic_search(
            document_name="test_doc",
            query_text="test query",
            scope="document",
            chapter_name=None,
            similarity_threshold=0.7,
            max_results=2,
        )

        # Should only return top 2 results
        assert len(results) == 2


class TestContextSnippetGeneration:
    """Test suite for context snippet generation."""

    def test_generate_context_snippet_with_surrounding(self):
        """Test snippet generation with surrounding paragraphs."""
        all_paragraphs = [
            {"paragraph_index": 0, "content": "Previous paragraph.", "chapter_name": "ch1.md"},
            {"paragraph_index": 1, "content": "Target paragraph.", "chapter_name": "ch1.md"},
            {"paragraph_index": 2, "content": "Next paragraph.", "chapter_name": "ch1.md"},
        ]
        snippet = _generate_context_snippet("Target paragraph.", all_paragraphs, 1)
        # Should include context from surrounding paragraphs
        assert snippet is not None
        assert "[MATCH]" in snippet
        assert "Previous" in snippet
        assert "Next" in snippet

    def test_generate_context_snippet_at_start(self):
        """Test snippet generation at the start of document."""
        all_paragraphs = [
            {"paragraph_index": 0, "content": "First paragraph.", "chapter_name": "ch1.md"},
            {"paragraph_index": 1, "content": "Second paragraph.", "chapter_name": "ch1.md"},
        ]
        snippet = _generate_context_snippet("First paragraph.", all_paragraphs, 0)
        assert snippet is not None
        assert "[MATCH]" in snippet

    def test_generate_context_snippet_at_end(self):
        """Test snippet generation at the end of document."""
        all_paragraphs = [
            {"paragraph_index": 0, "content": "First paragraph.", "chapter_name": "ch1.md"},
            {"paragraph_index": 1, "content": "Last paragraph.", "chapter_name": "ch1.md"},
        ]
        snippet = _generate_context_snippet("Last paragraph.", all_paragraphs, 1)
        assert snippet is not None
        assert "[MATCH]" in snippet

    def test_generate_context_snippet_single_paragraph(self):
        """Test snippet generation with single paragraph."""
        all_paragraphs = [
            {"paragraph_index": 0, "content": "Only paragraph.", "chapter_name": "ch1.md"},
        ]
        snippet = _generate_context_snippet("Only paragraph.", all_paragraphs, 0)
        assert snippet is not None
        assert "[MATCH]" in snippet
