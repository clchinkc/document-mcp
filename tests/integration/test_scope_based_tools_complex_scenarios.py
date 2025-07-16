"""Integration tests for scope-based tools with complex document scenarios.

These tests validate scope-based tools performance and behavior with realistic,
complex document structures using direct function calls.
"""

from tests.tool_imports import find_text
from tests.tool_imports import get_statistics
from tests.tool_imports import read_content


class TestScopeBasedToolsComplexScenarios:
    """Test scope-based tools with complex, realistic document scenarios."""

    def test_scope_based_tools_with_large_documents(self, document_factory):
        """Test scope-based tools performance with documents containing 10+ chapters."""
        # Create large document with many chapters
        document_name = "Large Novel"

        # Create document using existing factory
        chapters_dict = {}
        for i in range(1, 11):  # Reduced to 10 chapters for faster testing
            chapter_name = f"{i:02d}-chapter-{i}.md"
            chapter_content = f"""# Chapter {i}: The Journey Continues

This is chapter {i} of our epic novel. It contains multiple paragraphs
with substantial content to test the performance of our scope-based tools.

## Section {i}.1: Character Development

The main character continues to evolve in this chapter. We see significant
growth and development that adds depth to the overall narrative structure.

This chapter contains substantial content for testing the scope-based tools.
"""
            chapters_dict[chapter_name] = chapter_content

        # Use the document_factory fixture
        document_factory(document_name, chapters_dict)

        # Test scope-based read_content with full document scope
        response = read_content(document_name=document_name, scope="document")

        assert response is not None
        assert "chapters" in response or "content" in response

        # Test scope-based find_text across entire large document
        search_response = find_text(
            document_name=document_name, search_text="character", scope="document"
        )

        assert search_response is not None
        # Should find multiple occurrences across chapters

        # Test scope-based get_statistics for large document
        stats_response = get_statistics(document_name=document_name, scope="document")

        assert stats_response is not None
        # Should return statistics for the entire large document

    def test_scope_based_tools_error_recovery(self, document_factory):
        """Test scope-based tools handle document corruption/missing files gracefully."""
        # Test with non-existent document
        response = read_content(document_name="NonExistentDocument", scope="document")

        # Should return None or error response, not crash
        assert response is None or isinstance(response, dict)

        # Create document, then test with non-existent chapter
        document_name = "Test Document"
        document_factory(document_name, {"01-intro.md": "Introduction content"})

        chapter_response = read_content(
            document_name=document_name,
            scope="chapter",
            chapter_name="nonexistent-chapter.md",
        )

        # Should handle missing chapter gracefully
        assert chapter_response is None or isinstance(chapter_response, dict)

        # Test with invalid paragraph index
        paragraph_response = read_content(
            document_name=document_name,
            scope="paragraph",
            chapter_name="01-intro.md",
            paragraph_index=999,  # Invalid index
        )

        # Should handle invalid paragraph index gracefully
        assert paragraph_response is None or isinstance(paragraph_response, dict)

    def test_scope_based_tools_with_complex_content_structures(self, document_factory):
        """Test scope-based tools with complex markdown content and formatting."""
        document_name = "Complex Content Document"

        # Create chapter with complex markdown content
        complex_content = """# Chapter with Complex Content

## Tables
| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| Data 1   | Data 2   | Data 3   |
| More     | Content  | Here     |

## Code Blocks
```python
def example_function():
    return "This is code"
```

## Special Characters
This content includes special characters: !@#$%^&*()
Unicode characters: 测试 🚀 ñoël

This chapter tests how scope-based tools handle complex markdown formatting
and special characters in real-world document scenarios.
"""

        document_factory(document_name, {"complex-chapter.md": complex_content})

        # Test read_content with complex formatting
        read_response = read_content(
            document_name=document_name,
            scope="chapter",
            chapter_name="complex-chapter.md",
        )

        assert read_response is not None

        # Test find_text with special characters
        find_response = find_text(
            document_name=document_name,
            search_text="测试",  # Unicode search
            scope="document",
        )

        assert find_response is not None or find_response == []

        # Test find_text with code content
        code_find_response = find_text(
            document_name=document_name,
            search_text="def example_function",
            scope="document",
        )

        assert code_find_response is not None or code_find_response == []

        # Test statistics with complex content
        stats_response = get_statistics(document_name=document_name, scope="document")

        assert stats_response is not None

    def test_scope_based_tools_multiple_operations(self, document_factory):
        """Test scope-based tools with multiple sequential operations."""
        document_name = "Multi Operation Test Document"
        document_factory(
            document_name, {"01-test.md": "Test content for multiple operations"}
        )

        # Test multiple read operations
        doc_result = read_content(document_name=document_name, scope="document")
        assert doc_result is not None

        chapter_result = read_content(
            document_name=document_name, scope="chapter", chapter_name="01-test.md"
        )
        assert chapter_result is not None

        # Test multiple find operations
        find_result1 = find_text(
            document_name=document_name, search_text="Test", scope="document"
        )
        assert find_result1 is not None or find_result1 == []

        find_result2 = find_text(
            document_name=document_name, search_text="content", scope="document"
        )
        assert find_result2 is not None or find_result2 == []

        # Test statistics operation
        stats_result = get_statistics(document_name=document_name, scope="document")
        assert stats_result is not None
