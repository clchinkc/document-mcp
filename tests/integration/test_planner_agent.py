"""
Integration tests for the planner agent with mocked LLM and real MCP stdio communication.

This module tests planner agent's integration with MCP server using mocked LLM responses
to validate that agent components properly execute plans and handle MCP tool results.
"""

import uuid
from pathlib import Path
from typing import Dict
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic_ai.mcp import MCPServerStdio

from src.agents.planner_agent.main import (
    execute_plan,
    execute_plan_step,
    generate_execution_plan,
)
from src.agents.planner_agent.models import PlanStep


@pytest.fixture
async def mcp_server():
    """Provide a real MCP server for integration testing."""
    server = MCPServerStdio(
        command="python3", args=["-m", "document_mcp.doc_tool_server", "stdio"]
    )
    yield server


@pytest.fixture
def document_factory(temp_docs_root: Path):
    """A factory to create documents with chapters for testing."""

    def _create_document(doc_name: str, chapters: Dict[str, str] = None):
        doc_path = temp_docs_root / doc_name
        doc_path.mkdir(exist_ok=True)
        if chapters:
            for chapter_name, content in chapters.items():
                (doc_path / chapter_name).write_text(content)
        return doc_path

    return _create_document


class TestPlannerAgentMCPIntegration:
    """Integration tests for planner agent with mocked LLM and real MCP server."""

    @pytest.mark.asyncio
    async def test_execute_plan_step_with_real_mcp(self, mcp_server, temp_docs_root):
        """Test executing a single plan step with real MCP server."""
        doc_name = f"planner_test_{uuid.uuid4().hex[:8]}"

        step = PlanStep(
            tool_name="create_document", arguments={"document_name": doc_name}
        )

        async with mcp_server:
            result = await execute_plan_step(mcp_server, step, 0)

            assert result.success is True
            assert result.error is None
            assert result.step_index == 0
            assert result.tool_name == "create_document"
            assert "success" in result.result
            assert result.result["success"] is True

    @pytest.mark.asyncio
    async def test_execute_plan_step_failure(self, mcp_server, temp_docs_root):
        """Test executing a plan step that should fail."""
        # Try to create a chapter in a non-existent document
        step = PlanStep(
            tool_name="create_chapter",
            arguments={
                "document_name": "nonexistent_document",
                "chapter_name": "01-test.md",
                "initial_content": "Test content",
            },
        )

        async with mcp_server:
            result = await execute_plan_step(mcp_server, step, 0)

            # The step should still be marked as successful since the MCP call succeeds,
            # but the result should indicate the operation failed
            assert result.success is True  # MCP call succeeded
            assert result.error is None
            assert "success" in result.result
            assert result.result["success"] is False  # But the operation failed

    @pytest.mark.asyncio
    async def test_execute_multi_step_plan(self, mcp_server, temp_docs_root):
        """Test executing a complete multi-step plan."""
        doc_name = f"planner_multitest_{uuid.uuid4().hex[:8]}"

        plan = [
            PlanStep(
                tool_name="create_document", arguments={"document_name": doc_name}
            ),
            PlanStep(
                tool_name="create_chapter",
                arguments={
                    "document_name": doc_name,
                    "chapter_name": "01-intro.md",
                    "initial_content": "# Introduction\\n\\nThis is the introduction.",
                },
            ),
            PlanStep(
                tool_name="create_chapter",
                arguments={
                    "document_name": doc_name,
                    "chapter_name": "02-main.md",
                    "initial_content": "# Main Content\\n\\nThis is the main content.",
                },
            ),
        ]

        async with mcp_server:
            results = await execute_plan(mcp_server, plan)

            assert len(results) == 3

            # All steps should succeed
            for i, result in enumerate(results):
                assert result.success is True
                assert result.error is None
                assert result.step_index == i
                assert "success" in result.result
                assert result.result["success"] is True

    @pytest.mark.asyncio
    async def test_execute_plan_stops_on_failure(self, mcp_server, temp_docs_root):
        """Test that plan execution stops when a step fails."""
        doc_name = f"planner_failtest_{uuid.uuid4().hex[:8]}"

        plan = [
            PlanStep(
                tool_name="create_document", arguments={"document_name": doc_name}
            ),
            PlanStep(
                tool_name="create_chapter",
                arguments={
                    "document_name": "nonexistent_document",  # This will fail
                    "chapter_name": "01-intro.md",
                    "initial_content": "Test content",
                },
            ),
            PlanStep(
                tool_name="create_chapter",
                arguments={
                    "document_name": doc_name,
                    "chapter_name": "02-main.md",
                    "initial_content": "This should not be executed",
                },
            ),
        ]

        async with mcp_server:
            results = await execute_plan(mcp_server, plan)

            # Should execute first two steps and stop on the failure
            assert len(results) == 2

            # First step should succeed
            assert results[0].success is True
            assert results[0].result["success"] is True

            # Second step should succeed at MCP level but fail at operation level
            assert results[1].success is True
            assert results[1].result["success"] is False

            # Third step should NOT execute since step 2 failed logically

    @pytest.mark.asyncio
    async def test_mocked_plan_execution_with_real_mcp(
        self, mcp_server, temp_docs_root
    ):
        """Test executing a mocked plan with real MCP server integration."""
        doc_name = f"planner_mcp_integration_{uuid.uuid4().hex[:8]}"

        # Create a predefined plan (simulating LLM output)
        mock_plan = [
            PlanStep(
                tool_name="create_document", arguments={"document_name": doc_name}
            ),
            PlanStep(
                tool_name="create_chapter",
                arguments={
                    "document_name": doc_name,
                    "chapter_name": "01-intro.md",
                    "initial_content": "# Introduction\\n\\nTest content for MCP integration.",
                },
            ),
        ]

        # Execute the plan with real MCP server
        async with mcp_server:
            results = await execute_plan(mcp_server, mock_plan)

            # Verify MCP integration worked correctly
            assert len(results) == 2

            # Document creation should succeed
            assert results[0].success is True
            assert results[0].tool_name == "create_document"
            assert results[0].result["success"] is True

            # Chapter creation should succeed
            assert results[1].success is True
            assert results[1].tool_name == "create_chapter"
            assert results[1].result["success"] is True

    @pytest.mark.asyncio
    async def test_plan_generation_logic_with_mocked_llm(self):
        """Test plan generation and parsing logic with mocked LLM responses."""

        # Mock LLM agent
        mock_agent = AsyncMock()

        # Test successful JSON parsing
        mock_result = Mock()
        mock_result.output = """[
            {
                "tool_name": "create_document",
                "arguments": {"document_name": "Test Doc"}
            },
            {
                "tool_name": "create_chapter", 
                "arguments": {
                    "document_name": "Test Doc",
                    "chapter_name": "01-intro.md",
                    "initial_content": "Test content"
                }
            }
        ]"""

        mock_agent.run.return_value = mock_result

        # Test plan generation
        plan, result = await generate_execution_plan(mock_agent, "Create a test document")

        assert len(plan) == 2
        assert plan[0].tool_name == "create_document"
        assert plan[0].arguments == {"document_name": "Test Doc"}
        assert plan[1].tool_name == "create_chapter"
        assert plan[1].arguments["document_name"] == "Test Doc"

    @pytest.mark.asyncio
    async def test_plan_generation_with_markdown_wrapper(self):
        """Test plan generation handles markdown code blocks from LLM."""

        mock_agent = AsyncMock()
        mock_result = Mock()
        mock_result.output = """```json
        [
            {
                "tool_name": "create_document",
                "arguments": {"document_name": "Test"}
            }
        ]
        ```"""

        mock_agent.run.return_value = mock_result

        # Should successfully parse despite markdown wrapper
        plan, result = await generate_execution_plan(mock_agent, "Create a document")

        assert len(plan) == 1
        assert plan[0].tool_name == "create_document"

    @pytest.mark.asyncio
    async def test_plan_generation_error_handling(self):
        """Test plan generation error handling with invalid LLM responses."""

        mock_agent = AsyncMock()

        # Test invalid JSON
        mock_result = Mock()
        mock_result.output = "This is not valid JSON"
        mock_agent.run.return_value = mock_result

        with pytest.raises(ValueError, match="Invalid plan format from LLM"):
            await generate_execution_plan(mock_agent, "Invalid query")

        # Test invalid plan structure
        mock_result.output = '[{"invalid_field": "value"}]'
        mock_agent.run.return_value = mock_result

        with pytest.raises(ValueError, match="Invalid plan format from LLM"):
            await generate_execution_plan(mock_agent, "Invalid plan structure")

    @pytest.mark.asyncio
    async def test_paragraph_operations_integration(self, mcp_server, temp_docs_root):
        """Test planner agent with paragraph-level operations."""
        doc_name = f"planner_paragraph_{uuid.uuid4().hex[:8]}"

        plan = [
            PlanStep(
                tool_name="create_document", arguments={"document_name": doc_name}
            ),
            PlanStep(
                tool_name="create_chapter",
                arguments={
                    "document_name": doc_name,
                    "chapter_name": "01-intro.md",
                    "initial_content": "# Introduction\\n\\nFirst paragraph.",
                },
            ),
            PlanStep(
                tool_name="append_paragraph_to_chapter",
                arguments={
                    "document_name": doc_name,
                    "chapter_name": "01-intro.md",
                    "paragraph_content": "Second paragraph added by planner.",
                },
            ),
            PlanStep(
                tool_name="insert_paragraph_after",
                arguments={
                    "document_name": doc_name,
                    "chapter_name": "01-intro.md",
                    "paragraph_index": 1,
                    "new_content": "Inserted paragraph between first and second.",
                },
            ),
        ]

        async with mcp_server:
            results = await execute_plan(mcp_server, plan)

            assert len(results) == 4

            # All steps should succeed
            for i, result in enumerate(results):
                assert result.success is True
                assert result.error is None
                assert result.step_index == i
                assert "success" in result.result
                assert result.result["success"] is True

    @pytest.mark.asyncio
    async def test_document_statistics_operations(self, mcp_server, temp_docs_root):
        """Test planner agent with document statistics operations."""
        doc_name = f"planner_stats_{uuid.uuid4().hex[:8]}"

        plan = [
            # First create the document and chapter with content
            PlanStep(
                tool_name="create_document", arguments={"document_name": doc_name}
            ),
            PlanStep(
                tool_name="create_chapter",
                arguments={
                    "document_name": doc_name,
                    "chapter_name": "01-intro.md",
                    "initial_content": "# Introduction\\n\\nThis is some content for statistics testing.",
                },
            ),
            # Get document statistics
            PlanStep(
                tool_name="get_document_statistics",
                arguments={"document_name": doc_name},
            ),
        ]

        async with mcp_server:
            results = await execute_plan(mcp_server, plan)

            assert len(results) == 3

            # Document creation should succeed
            assert results[0].success is True
            assert results[0].result["success"] is True

            # Chapter creation should succeed
            assert results[1].success is True
            assert results[1].result["success"] is True

            # Statistics should succeed and return data
            assert results[2].success is True
            assert "word_count" in results[2].result
            assert results[2].result["word_count"] > 0
