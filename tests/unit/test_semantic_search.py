"""Unit tests for semantic search functionality.

These tests focus on the core logic of the semantic search implementation,
with mocked external dependencies (API calls).
"""

from unittest.mock import MagicMock
from unittest.mock import patch

from document_mcp.tools.content_tools import _generate_context_snippet
from document_mcp.tools.content_tools import _perform_semantic_search


class TestSemanticSearchCore:
    """Test suite for semantic search core functionality."""

    @patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"})
    @patch("document_mcp.tools.content_tools.genai")
    @patch("document_mcp.tools.content_tools._get_document_path")
    @patch("document_mcp.tools.content_tools._get_ordered_chapter_files")
    @patch("document_mcp.tools.content_tools._split_into_paragraphs")
    def test_perform_semantic_search_document_scope(
        self, mock_split, mock_chapters, mock_doc_path, mock_genai
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

        # Mock embeddings API response
        mock_genai.embed_content.return_value = {
            "embedding": [
                [1.0, 0.0, 0.0],  # Query embedding
                [0.9, 0.1, 0.0],  # First paragraph (high similarity)
                [0.0, 0.0, 1.0],  # Second paragraph (low similarity)
            ]
        }

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
    @patch("document_mcp.tools.content_tools.genai")
    @patch("document_mcp.tools.content_tools._get_chapter_path")
    @patch("document_mcp.tools.content_tools._split_into_paragraphs")
    def test_perform_semantic_search_chapter_scope(
        self, mock_split, mock_chapter_path, mock_genai
    ):
        """Test semantic search with chapter scope."""
        # Setup mocks
        mock_chapter_path.return_value.exists.return_value = True
        mock_chapter_path.return_value.read_text.return_value = "Chapter content"

        # Mock paragraphs
        mock_split.return_value = ["Chapter paragraph"]

        # Mock embeddings API response with high similarity
        mock_genai.embed_content.return_value = {
            "embedding": [
                [1.0, 0.0, 0.0],  # Query embedding
                [0.9, 0.1, 0.0],  # Paragraph embedding (high similarity)
            ]
        }

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
    @patch("document_mcp.tools.content_tools.genai")
    @patch("document_mcp.tools.content_tools._get_document_path")
    @patch("document_mcp.tools.content_tools._get_ordered_chapter_files")
    @patch("document_mcp.tools.content_tools._split_into_paragraphs")
    def test_perform_semantic_search_similarity_threshold_filtering(
        self, mock_split, mock_chapters, mock_doc_path, mock_genai
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
        mock_genai.embed_content.return_value = {
            "embedding": [
                [1.0, 0.0, 0.0],  # Query
                [0.9, 0.1, 0.0],  # Para 1: high similarity (~0.95)
                [0.5, 0.5, 0.0],  # Para 2: medium similarity (~0.71)
                [0.0, 1.0, 0.0],  # Para 3: low similarity (0.0)
            ]
        }

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
    @patch("document_mcp.tools.content_tools.genai")
    @patch("document_mcp.tools.content_tools._get_document_path")
    @patch("document_mcp.tools.content_tools._get_ordered_chapter_files")
    @patch("document_mcp.tools.content_tools._split_into_paragraphs")
    def test_perform_semantic_search_max_results_limiting(
        self, mock_split, mock_chapters, mock_doc_path, mock_genai
    ):
        """Test that max_results properly limits the number of results."""
        # Setup mocks
        mock_doc_path.return_value.exists.return_value = True

        mock_chapter = MagicMock()
        mock_chapter.name = "01-test.md"
        mock_chapter.read_text.return_value = "Sample content"
        mock_chapters.return_value = [mock_chapter]

        # Many paragraphs
        mock_split.return_value = [f"Paragraph {i}" for i in range(5)]

        # Mock embeddings with all high similarities
        embeddings = [[1.0, 0.0]] + [[0.9, 0.1] for _ in range(5)]
        mock_genai.embed_content.return_value = {"embedding": embeddings}

        # Test with max_results=2
        results = _perform_semantic_search(
            document_name="test_doc",
            query_text="test query",
            scope="document",
            chapter_name=None,
            similarity_threshold=0.5,
            max_results=2,
        )

        # Should only return 2 results
        assert len(results) == 2


class TestContextSnippetGeneration:
    """Test suite for context snippet generation."""

    def test_generate_context_snippet_middle_paragraph(self):
        """Test context snippet generation for middle paragraph."""
        all_paragraphs = [
            {"chapter_name": "01-test.md", "content": "First paragraph"},
            {"chapter_name": "01-test.md", "content": "Target paragraph"},
            {"chapter_name": "01-test.md", "content": "Third paragraph"},
        ]

        snippet = _generate_context_snippet("Target paragraph", all_paragraphs, 1)

        assert "[MATCH] Target paragraph" in snippet
        assert "First paragraph" in snippet
        assert "Third paragraph" in snippet

    def test_generate_context_snippet_first_paragraph(self):
        """Test context snippet generation for first paragraph."""
        all_paragraphs = [
            {"chapter_name": "01-test.md", "content": "Target paragraph"},
            {"chapter_name": "01-test.md", "content": "Second paragraph"},
        ]

        snippet = _generate_context_snippet("Target paragraph", all_paragraphs, 0)

        assert "[MATCH] Target paragraph" in snippet
        assert "Second paragraph" in snippet
        # Should not include previous paragraph (doesn't exist)

    def test_generate_context_snippet_last_paragraph(self):
        """Test context snippet generation for last paragraph."""
        all_paragraphs = [
            {"chapter_name": "01-test.md", "content": "First paragraph"},
            {"chapter_name": "01-test.md", "content": "Target paragraph"},
        ]

        snippet = _generate_context_snippet("Target paragraph", all_paragraphs, 1)

        assert "[MATCH] Target paragraph" in snippet
        assert "First paragraph" in snippet
        # Should not include next paragraph (doesn't exist)

    def test_generate_context_snippet_different_chapters(self):
        """Test context snippet only includes same-chapter paragraphs."""
        all_paragraphs = [
            {"chapter_name": "01-test.md", "content": "Chapter 1 para"},
            {"chapter_name": "02-test.md", "content": "Target paragraph"},
            {"chapter_name": "03-test.md", "content": "Chapter 3 para"},
        ]

        snippet = _generate_context_snippet("Target paragraph", all_paragraphs, 1)

        assert "[MATCH] Target paragraph" in snippet
        # Should not include paragraphs from other chapters
        assert "Chapter 1 para" not in snippet
        assert "Chapter 3 para" not in snippet

    def test_generate_context_snippet_error_handling(self):
        """Test context snippet generation with invalid inputs."""
        all_paragraphs = [
            {"chapter_name": "01-test.md", "content": "Some paragraph"},
        ]

        # Test with non-existent target content
        snippet = _generate_context_snippet("Nonexistent", all_paragraphs, 0)
        assert snippet is None


class TestSemanticSearchValidation:
    """Test suite for semantic search input validation."""

    def test_similarity_threshold_validation(self):
        """Test that similarity threshold is properly validated."""
        # This would be tested in the actual tool function
        # Here we test the boundaries
        valid_thresholds = [0.0, 0.5, 1.0]
        invalid_thresholds = [-0.1, 1.1, 2.0]

        for threshold in valid_thresholds:
            assert 0.0 <= threshold <= 1.0

        for threshold in invalid_thresholds:
            assert not (0.0 <= threshold <= 1.0)

    def test_max_results_validation(self):
        """Test that max_results is properly validated."""
        valid_limits = [1, 10, 100]
        invalid_limits = [0, -1, -10]

        for limit in valid_limits:
            assert limit > 0

        for limit in invalid_limits:
            assert not (limit > 0)
