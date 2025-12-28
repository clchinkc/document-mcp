"""Mock factory utilities for consistent test mocking across the Document MCP project.

This module provides standardized mock objects, responses, and factories
to eliminate mock duplication and ensure consistent testing patterns.
"""
from __future__ import annotations


import datetime
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

from document_mcp.models import ChapterContent
from document_mcp.models import ChapterMetadata
from document_mcp.models import DocumentInfo
from document_mcp.models import FullDocumentContent
from document_mcp.models import OperationStatus
from document_mcp.models import SemanticSearchResponse
from document_mcp.models import SemanticSearchResult
from document_mcp.models import StatisticsReport


class MockDataFactory:
    """Factory for creating consistent mock data across tests."""

    @staticmethod
    def create_document_info(
        document_name: str = "test_doc",
        chapter_count: int = 3,
        word_count: int = 1000,
        paragraph_count: int = 50,
    ) -> DocumentInfo:
        """Create a mock DocumentInfo object."""
        chapters = []
        for i in range(chapter_count):
            chapter = MockDataFactory.create_chapter_metadata(
                chapter_name=f"chapter_{i + 1:02d}.md",
                word_count=word_count // chapter_count,
                paragraph_count=paragraph_count // chapter_count,
            )
            chapters.append(chapter)

        return DocumentInfo(
            document_name=document_name,
            total_chapters=chapter_count,
            total_word_count=word_count,
            total_paragraph_count=paragraph_count,
            last_modified=datetime.datetime.now(),
            chapters=chapters,
            has_summary=False,
        )

    @staticmethod
    def create_chapter_metadata(
        chapter_name: str = "chapter_01.md",
        title: str | None = None,
        word_count: int = 300,
        paragraph_count: int = 15,
    ) -> ChapterMetadata:
        """Create a mock ChapterMetadata object."""
        return ChapterMetadata(
            chapter_name=chapter_name,
            title=title or f"Chapter {chapter_name.split('_')[1].split('.')[0]}",
            word_count=word_count,
            paragraph_count=paragraph_count,
            last_modified=datetime.datetime.now(),
        )

    @staticmethod
    def create_chapter_content(
        document_name: str = "test_doc",
        chapter_name: str = "chapter_01.md",
        content: str = "# Test Chapter\n\nThis is test content.",
        word_count: int | None = None,
        paragraph_count: int | None = None,
    ) -> ChapterContent:
        """Create a mock ChapterContent object."""
        if word_count is None:
            word_count = len(content.split())
        if paragraph_count is None:
            paragraph_count = len([p for p in content.split("\n\n") if p.strip()])

        return ChapterContent(
            document_name=document_name,
            chapter_name=chapter_name,
            content=content,
            word_count=word_count,
            paragraph_count=paragraph_count,
            last_modified=datetime.datetime.now(),
        )

    @staticmethod
    def create_full_document_content(
        document_name: str = "test_doc", chapter_contents: list[str] | None = None
    ) -> FullDocumentContent:
        """Create a mock FullDocumentContent object."""
        if chapter_contents is None:
            chapter_contents = [
                "# Chapter 1\n\nFirst chapter content.",
                "# Chapter 2\n\nSecond chapter content.",
                "# Chapter 3\n\nThird chapter content.",
            ]

        chapters = []
        total_words = 0
        total_paragraphs = 0

        for i, content in enumerate(chapter_contents):
            chapter = MockDataFactory.create_chapter_content(
                document_name=document_name, chapter_name=f"chapter_{i + 1:02d}.md", content=content
            )
            chapters.append(chapter)
            total_words += chapter.word_count
            total_paragraphs += chapter.paragraph_count

        return FullDocumentContent(
            document_name=document_name,
            chapters=chapters,
            total_word_count=total_words,
            total_paragraph_count=total_paragraphs,
        )

    @staticmethod
    def create_statistics_report(
        scope: str = "document: test_doc",
        word_count: int = 1000,
        paragraph_count: int = 50,
        chapter_count: int | None = 3,
    ) -> StatisticsReport:
        """Create a mock StatisticsReport object."""
        return StatisticsReport(
            scope=scope, word_count=word_count, paragraph_count=paragraph_count, chapter_count=chapter_count
        )

    @staticmethod
    def create_semantic_search_result(
        document_name: str = "test_doc",
        chapter_name: str = "chapter_01.md",
        paragraph_index: int = 0,
        content: str = "This is matching content.",
        similarity_score: float = 0.85,
        context_snippet: str | None = None,
    ) -> SemanticSearchResult:
        """Create a mock SemanticSearchResult object."""
        return SemanticSearchResult(
            document_name=document_name,
            chapter_name=chapter_name,
            paragraph_index=paragraph_index,
            content=content,
            similarity_score=similarity_score,
            context_snippet=context_snippet or f"...{content}...",
        )

    @staticmethod
    def create_semantic_search_response(
        document_name: str = "test_doc",
        scope: str = "document",
        query_text: str = "test query",
        results: list[SemanticSearchResult] | None = None,
        execution_time_ms: float = 50.0,
    ) -> SemanticSearchResponse:
        """Create a mock SemanticSearchResponse object."""
        if results is None:
            results = [
                MockDataFactory.create_semantic_search_result(
                    document_name=document_name, similarity_score=0.9
                ),
                MockDataFactory.create_semantic_search_result(
                    document_name=document_name, chapter_name="chapter_02.md", similarity_score=0.8
                ),
            ]

        return SemanticSearchResponse(
            document_name=document_name,
            scope=scope,
            query_text=query_text,
            results=results,
            total_results=len(results),
            execution_time_ms=execution_time_ms,
        )

    @staticmethod
    def create_operation_result(
        success: bool = True,
        operation_type: str = "test_operation",
        result_data: dict[str, Any] | None = None,
        error: str | None = None,
        execution_time_ms: float = 10.0,
    ) -> OperationStatus:
        """Create a mock OperationStatus object."""
        return OperationStatus(
            success=success,
            message=f"{operation_type} completed",
            details=result_data or {"message": "Operation completed"},
        )


class MockMCPClientFactory:
    """Factory for creating consistent MCP client mocks."""

    @staticmethod
    def create_mcp_client_mock() -> MagicMock:
        """Create a mock MCP client with standard methods."""
        mock_client = MagicMock()

        # Configure common async methods
        mock_client.call_tool = AsyncMock()
        mock_client.list_tools = AsyncMock()
        mock_client.connect = AsyncMock()
        mock_client.disconnect = AsyncMock()

        return mock_client

    @staticmethod
    def create_mcp_response_mock(
        tool_name: str, success: bool = True, result: dict[str, Any] | None = None, error: str | None = None
    ) -> dict[str, Any]:
        """Create a mock MCP tool response."""
        if result is None:
            result = {"message": f"Mock {tool_name} completed successfully"}

        response = {
            "content": [{"type": "text", "text": str(result) if success else str(error or "Mock error")}]
        }

        if not success:
            response["isError"] = True

        return response


class MockLLMFactory:
    """Factory for creating LLM response mocks."""

    @staticmethod
    def create_agent_result_mock(
        content: str = "Mock agent response", tool_calls: list[dict[str, Any]] | None = None
    ) -> MagicMock:
        """Create a mock agent result object."""
        mock_result = MagicMock()
        mock_result.data = content

        if tool_calls:
            mock_result.all_messages = []
            for tool_call in tool_calls:
                mock_message = MagicMock()
                mock_message.content = [MagicMock()]
                mock_message.content[0].tool_call = MagicMock()
                mock_message.content[0].tool_call.tool_name = tool_call.get("name", "test_tool")
                mock_message.content[0].tool_call.args = tool_call.get("args", {})
                mock_result.all_messages.append(mock_message)
        else:
            mock_result.all_messages = []

        return mock_result

    @staticmethod
    def create_openai_response_mock(
        content: str = "Mock OpenAI response", tool_calls: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        """Create a mock OpenAI API response."""
        message = {"role": "assistant", "content": content}

        if tool_calls:
            message["tool_calls"] = [
                {
                    "id": f"call_{i}",
                    "type": "function",
                    "function": {
                        "name": tool_call.get("name", "test_tool"),
                        "arguments": str(tool_call.get("args", {})),
                    },
                }
                for i, tool_call in enumerate(tool_calls)
            ]

        return {
            "choices": [{"message": message, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        }


def create_mock_llm_response(
    content: str = "Mock LLM response", tool_calls: list[dict[str, Any]] | None = None
) -> MagicMock:
    """Create a mock LLM response for testing.

    Convenience function that creates a mock LLM response using the MockLLMFactory.
    """
    return MockLLMFactory.create_agent_result_mock(content=content, tool_calls=tool_calls)


# Convenience instances for common use cases
mock_data = MockDataFactory()
mock_mcp = MockMCPClientFactory()
mock_llm = MockLLMFactory()
