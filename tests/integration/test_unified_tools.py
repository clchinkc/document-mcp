"""Integration tests for unified tools in the Document MCP tool server."""

from document_mcp.mcp_client import find_similar_text
from document_mcp.mcp_client import find_text
from document_mcp.mcp_client import get_statistics
from document_mcp.mcp_client import read_content
from document_mcp.mcp_client import replace_text


class TestUnifiedReadContent:
    """Integration tests for the unified read_content tool."""

    def test_read_content_document_scope(self, document_factory):
        """Test reading full document using unified read_content tool."""
        doc_name = "test_unified_doc"
        chapters = {
            "01-intro.md": "# Introduction\n\nWelcome to the document.",
            "02-content.md": "# Content\n\nThis is the main content.",
        }
        document_factory(doc_name, chapters)

        result = read_content(doc_name, scope="document")

        assert result is not None
        assert result.document_name == doc_name
        assert result.scope == "document"
        assert result.content is not None
        # Content should include both chapters (since total is small)
        assert "Introduction" in result.content
        assert "main content" in result.content
        # Check pagination metadata
        assert result.pagination.page == 1
        assert result.pagination.total_characters > 0

    def test_read_content_chapter_scope(self, document_factory):
        """Test reading specific chapter using unified read_content tool."""
        doc_name = "test_unified_chapter"
        chapters = {
            "chapter1.md": "# Chapter 1\n\nFirst chapter content.",
            "chapter2.md": "# Chapter 2\n\nSecond chapter content.",
        }
        document_factory(doc_name, chapters)

        result = read_content(doc_name, scope="chapter", chapter_name="chapter1.md")

        assert result is not None
        assert result.document_name == doc_name
        assert result.chapter_name == "chapter1.md"
        assert "First chapter content" in result.content
        assert result.word_count > 0

    def test_read_content_paragraph_scope(self, document_factory):
        """Test reading specific paragraph using unified read_content tool."""
        doc_name = "test_unified_paragraph"
        chapters = {
            "test.md": "# Title\n\nFirst paragraph.\n\nSecond paragraph.\n\nThird paragraph.",
        }
        document_factory(doc_name, chapters)

        result = read_content(doc_name, scope="paragraph", chapter_name="test.md", paragraph_index=1)

        assert result is not None
        assert result.document_name == doc_name
        assert result.chapter_name == "test.md"
        assert result.paragraph_index_in_chapter == 1
        assert result.content == "First paragraph."

    def test_read_content_invalid_scope(self, document_factory):
        """Test read_content with invalid scope."""
        doc_name = "test_invalid_scope"
        document_factory(doc_name, {"test.md": "Content"})

        result = read_content(doc_name, scope="invalid_scope")
        assert result is None

    def test_read_content_missing_chapter_name_for_chapter_scope(self, document_factory):
        """Test read_content chapter scope without chapter_name."""
        doc_name = "test_missing_chapter"
        document_factory(doc_name, {"test.md": "Content"})

        result = read_content(doc_name, scope="chapter")
        assert result is None

    def test_read_content_missing_parameters_for_paragraph_scope(self, document_factory):
        """Test read_content paragraph scope without required parameters."""
        doc_name = "test_missing_params"
        document_factory(doc_name, {"test.md": "Content"})

        result = read_content(doc_name, scope="paragraph")
        assert result is None

        result = read_content(doc_name, scope="paragraph", chapter_name="test.md")
        assert result is None


class TestUnifiedTools:
    """Integration tests for the new unified tools."""

    def test_find_text_document_scope(self, document_factory):
        """Test unified find_text with document scope."""
        doc_name = "test_find_doc"
        chapters = {
            "chapter1.md": "This is the first chapter with important content.",
            "chapter2.md": "Second chapter also has important information.",
            "chapter3.md": "Third chapter with different content.",
        }
        document_factory(doc_name, chapters)

        results = find_text(doc_name, "important", scope="document", case_sensitive=False)

        assert results is not None
        assert len(results) == 2
        chapter_names = [r.chapter_name for r in results]
        assert "chapter1.md" in chapter_names
        assert "chapter2.md" in chapter_names

    def test_find_text_chapter_scope(self, document_factory):
        """Test unified find_text with chapter scope."""
        doc_name = "test_find_chapter"
        chapters = {
            "test.md": "This chapter contains multiple instances of the word test. Test again.",
        }
        document_factory(doc_name, chapters)

        results = find_text(
            doc_name,
            "test",
            scope="chapter",
            chapter_name="test.md",
            case_sensitive=False,
        )

        assert results is not None
        assert len(results) >= 1

    def test_find_text_invalid_scope(self, document_factory):
        """Test unified find_text with invalid scope."""
        doc_name = "test_find_invalid"
        document_factory(doc_name, {"test.md": "Content"})

        result = find_text(doc_name, "content", scope="invalid")
        assert result is None

    def test_replace_text_document_scope(self, document_factory, temp_docs_root):
        """Test unified replace_text with document scope."""
        doc_name = "test_replace_doc"
        chapters = {
            "chapter1.md": "Replace this old text in chapter 1.",
            "chapter2.md": "Also replace old text in chapter 2.",
        }
        document_factory(doc_name, chapters)

        result = replace_text(doc_name, "old text", "new content", scope="document")

        assert result is not None
        assert result.success is True
        assert result.details["total_occurrences_replaced"] == 2

        ch1_content = (temp_docs_root / doc_name / "chapter1.md").read_text()
        ch2_content = (temp_docs_root / doc_name / "chapter2.md").read_text()
        assert "new content" in ch1_content
        assert "new content" in ch2_content
        assert "old text" not in ch1_content
        assert "old text" not in ch2_content

    def test_replace_text_chapter_scope(self, document_factory, temp_docs_root):
        """Test unified replace_text with chapter scope."""
        doc_name = "test_replace_chapter"
        chapters = {
            "target.md": "Replace this specific text only in this chapter.",
            "other.md": "Do not replace this specific text in this chapter.",
        }
        document_factory(doc_name, chapters)

        result = replace_text(
            doc_name,
            "specific text",
            "modified text",
            scope="chapter",
            chapter_name="target.md",
        )

        assert result is not None
        assert result.success is True
        assert result.details["occurrences_replaced"] == 1

        target_content = (temp_docs_root / doc_name / "target.md").read_text()
        other_content = (temp_docs_root / doc_name / "other.md").read_text()
        assert "modified text" in target_content
        assert "specific text" not in target_content
        assert "specific text" in other_content

    def test_get_statistics_document_scope(self, document_factory):
        """Test unified get_statistics with document scope."""
        doc_name = "test_stats_doc"
        chapters = {
            "ch1.md": "Chapter one has four words.",
            "ch2.md": "Chapter two has five total words.",
        }
        document_factory(doc_name, chapters)

        result = get_statistics(doc_name, scope="document")

        assert result is not None
        assert result.scope.startswith("document:")
        assert result.chapter_count == 2
        assert result.word_count == 11
        assert result.paragraph_count == 2

    def test_get_statistics_chapter_scope(self, document_factory):
        """Test unified get_statistics with chapter scope."""
        doc_name = "test_stats_chapter"
        chapters = {
            "target.md": "This chapter has exactly five words.",
            "other.md": "This other chapter has more than five words total.",
        }
        document_factory(doc_name, chapters)

        result = get_statistics(doc_name, scope="chapter", chapter_name="target.md")

        assert result is not None
        assert result.scope.endswith("target.md")
        assert result.word_count == 6
        assert result.paragraph_count == 1
        assert not hasattr(result, "chapter_count") or result.chapter_count is None

    def test_unified_tools_with_missing_parameters(self, document_factory):
        """Test unified tools with missing required parameters."""
        doc_name = "test_missing_params"
        document_factory(doc_name, {"test.md": "Content"})

        result = find_text(doc_name, "content", scope="chapter")
        assert result is None

        result = replace_text(doc_name, "old", "new", scope="chapter")
        assert result is None

        result = get_statistics(doc_name, scope="chapter")
        assert result is None


class TestSemanticSearchIntegration:
    """Integration tests for semantic search functionality."""

    def test_find_similar_text_no_api_key(self, document_factory, monkeypatch):
        """Test semantic search fails gracefully without API key."""
        # Remove API key from environment
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

        doc_name = "test_no_api_key"
        chapters = {"ch1.md": "Some content"}
        document_factory(doc_name, chapters)

        result = find_similar_text(doc_name, "query text")

        assert result is not None
        assert result.total_results == 0
        assert len(result.results) == 0

    def test_find_similar_text_missing_parameters(self, document_factory):
        """Test semantic search with missing required parameters."""
        doc_name = "test_missing_params"
        document_factory(doc_name, {"test.md": "Content"})

        # Missing chapter_name for chapter scope
        result = find_similar_text(doc_name, "test query", scope="chapter")
        assert result is None

    def test_find_similar_text_invalid_threshold(self, document_factory):
        """Test semantic search with invalid similarity threshold."""
        doc_name = "test_invalid_threshold"
        document_factory(doc_name, {"test.md": "Content"})

        # Invalid threshold > 1.0
        result = find_similar_text(doc_name, "test query", similarity_threshold=1.5)
        assert result is None

        # Invalid threshold < 0.0
        result = find_similar_text(doc_name, "test query", similarity_threshold=-0.1)
        assert result is None

    def test_find_similar_text_invalid_max_results(self, document_factory):
        """Test semantic search with invalid max_results."""
        doc_name = "test_invalid_max"
        document_factory(doc_name, {"test.md": "Content"})

        # Invalid max_results <= 0
        result = find_similar_text(doc_name, "test query", max_results=0)
        assert result is None

        result = find_similar_text(doc_name, "test query", max_results=-1)
        assert result is None
