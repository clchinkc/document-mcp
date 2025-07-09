"""
Unit tests for the planner agent components.

These tests validate the planner agent's core functionality with mocked dependencies.
"""

from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import ValidationError

from src.agents.planner_agent.main import (
    execute_plan_step,
    generate_execution_plan,
    generate_final_summary,
)
from src.agents.planner_agent.models import PlannerAgentResponse, PlanStep, StepResult


class TestPlanStep:
    """Test the PlanStep model validation."""

    def test_valid_plan_step(self):
        """Test creating a valid PlanStep."""
        step = PlanStep(
            tool_name="create_document", arguments={"document_name": "Test Doc"}
        )
        assert step.tool_name == "create_document"
        assert step.arguments == {"document_name": "Test Doc"}

    def test_plan_step_validation_errors(self):
        """Test PlanStep validation errors."""
        # Missing tool_name
        with pytest.raises(ValidationError):
            PlanStep(arguments={"document_name": "Test"})

        # Missing arguments
        with pytest.raises(ValidationError):
            PlanStep(tool_name="create_document")

        # Invalid tool_name type
        with pytest.raises(ValidationError):
            PlanStep(tool_name=123, arguments={})

        # Invalid arguments type
        with pytest.raises(ValidationError):
            PlanStep(tool_name="create_document", arguments="invalid")


class TestStepResult:
    """Test the StepResult model validation."""

    def test_successful_step_result(self):
        """Test creating a successful StepResult."""
        result = StepResult(
            step_index=0,
            tool_name="create_document",
            arguments={"document_name": "Test"},
            success=True,
            result={"status": "success"},
            error=None,
        )
        assert result.success is True
        assert result.error is None
        assert result.result == {"status": "success"}

    def test_failed_step_result(self):
        """Test creating a failed StepResult."""
        result = StepResult(
            step_index=1,
            tool_name="create_chapter",
            arguments={"document_name": "Test", "chapter_name": "01-intro.md"},
            success=False,
            result=None,
            error="Document not found",
        )
        assert result.success is False
        assert result.error == "Document not found"
        assert result.result is None


class TestPlannerAgentResponse:
    """Test the PlannerAgentResponse model validation."""

    def test_successful_response(self):
        """Test creating a successful PlannerAgentResponse."""
        step_result = StepResult(
            step_index=0,
            tool_name="create_document",
            arguments={"document_name": "Test"},
            success=True,
            result={"status": "success"},
            error=None,
        )

        response = PlannerAgentResponse(
            query="Create a test document",
            plan_generated=True,
            execution_completed=True,
            steps_executed=[step_result],
            summary="Successfully created document",
            error=None,
        )

        assert response.plan_generated is True
        assert response.execution_completed is True
        assert len(response.steps_executed) == 1
        assert response.error is None

    def test_failed_response(self):
        """Test creating a failed PlannerAgentResponse."""
        response = PlannerAgentResponse(
            query="Invalid query",
            plan_generated=False,
            execution_completed=False,
            steps_executed=[],
            summary="Failed to generate plan",
            error="Invalid JSON format",
        )

        assert response.plan_generated is False
        assert response.execution_completed is False
        assert len(response.steps_executed) == 0
        assert response.error == "Invalid JSON format"


class TestGenerateExecutionPlan:
    """Test the generate_execution_plan function."""

    @pytest.mark.asyncio
    async def test_generate_plan_success(self):
        """Test successful plan generation."""
        # Mock agent result
        mock_result = Mock()
        mock_result.output = (
            '[{"tool_name": "create_document", "arguments": {"document_name": "Test"}}]'
        )

        # Mock agent
        mock_agent = AsyncMock()
        mock_agent.run.return_value = mock_result

        # Test plan generation
        plan, result = await generate_execution_plan(mock_agent, "Create a test document")

        assert len(plan) == 1
        assert plan[0].tool_name == "create_document"
        assert plan[0].arguments == {"document_name": "Test"}

    @pytest.mark.asyncio
    async def test_generate_plan_with_markdown_wrapper(self):
        """Test plan generation with markdown code blocks."""
        # Mock agent result with markdown wrapper
        mock_result = Mock()
        mock_result.output = '```json\n[{"tool_name": "create_document", "arguments": {"document_name": "Test"}}]\n```'

        # Mock agent
        mock_agent = AsyncMock()
        mock_agent.run.return_value = mock_result

        # Test plan generation
        plan, result = await generate_execution_plan(mock_agent, "Create a test document")

        assert len(plan) == 1
        assert plan[0].tool_name == "create_document"
        assert plan[0].arguments == {"document_name": "Test"}

    @pytest.mark.asyncio
    async def test_generate_plan_invalid_json(self):
        """Test plan generation with invalid JSON."""
        # Mock agent result with invalid JSON
        mock_result = Mock()
        mock_result.output = "invalid json"

        # Mock agent
        mock_agent = AsyncMock()
        mock_agent.run.return_value = mock_result

        # Test that invalid JSON raises ValueError
        with pytest.raises(ValueError, match="Invalid plan format from LLM"):
            await generate_execution_plan(mock_agent, "Create a test document")

    @pytest.mark.asyncio
    async def test_generate_plan_invalid_step_format(self):
        """Test plan generation with invalid step format."""
        # Mock agent result with invalid step format
        mock_result = Mock()
        mock_result.output = '[{"invalid_field": "value"}]'

        # Mock agent
        mock_agent = AsyncMock()
        mock_agent.run.return_value = mock_result

        # Test that invalid step format raises ValueError
        with pytest.raises(ValueError, match="Invalid plan format from LLM"):
            await generate_execution_plan(mock_agent, "Create a test document")


class TestExecutePlanStep:
    """Test the execute_plan_step function."""

    @pytest.mark.asyncio
    async def test_execute_step_success(self):
        """Test successful step execution."""
        # Mock MCP server response
        mock_content = Mock()
        mock_content.text = '{"success": true, "message": "Document created"}'

        mock_result = Mock()
        mock_result.content = [mock_content]

        mock_mcp_server = Mock()
        mock_mcp_server._client.call_tool = AsyncMock(return_value=mock_result)

        # Create test step
        step = PlanStep(
            tool_name="create_document", arguments={"document_name": "Test"}
        )

        # Execute step
        result = await execute_plan_step(mock_mcp_server, step, 0)

        assert result.success is True
        assert result.error is None
        assert result.result == {"success": True, "message": "Document created"}
        assert result.step_index == 0
        assert result.tool_name == "create_document"

    @pytest.mark.asyncio
    async def test_execute_step_failure(self):
        """Test step execution failure."""
        # Mock MCP server to raise exception
        mock_mcp_server = Mock()
        mock_mcp_server._client.call_tool = AsyncMock(
            side_effect=Exception("Connection failed")
        )

        # Create test step
        step = PlanStep(
            tool_name="create_document", arguments={"document_name": "Test"}
        )

        # Execute step
        result = await execute_plan_step(mock_mcp_server, step, 0)

        assert result.success is False
        assert result.error == "Connection failed"
        assert result.result is None
        assert result.step_index == 0
        assert result.tool_name == "create_document"


class TestGenerateFinalSummary:
    """Test the generate_final_summary function."""

    @pytest.mark.asyncio
    async def test_summary_all_successful(self):
        """Test summary generation when all steps succeed."""
        # Create successful step results
        results = [
            StepResult(
                step_index=0,
                tool_name="create_document",
                arguments={"document_name": "Test"},
                success=True,
                result={"success": True},
                error=None,
            ),
            StepResult(
                step_index=1,
                tool_name="create_chapter",
                arguments={"document_name": "Test", "chapter_name": "01-intro.md"},
                success=True,
                result={"success": True},
                error=None,
            ),
        ]

        # Mock planning agent (not used in current implementation)
        Mock()

        summary = await generate_final_summary(results)

        assert "Successfully executed 2 step(s)" in summary
        assert "create_document" in summary
        assert "create_chapter" in summary
        assert "Failed" not in summary

    @pytest.mark.asyncio
    async def test_summary_with_failures(self):
        """Test summary generation when some steps fail."""
        # Create mixed results
        results = [
            StepResult(
                step_index=0,
                tool_name="create_document",
                arguments={"document_name": "Test"},
                success=True,
                result={"success": True},
                error=None,
            ),
            StepResult(
                step_index=1,
                tool_name="create_chapter",
                arguments={"document_name": "Test", "chapter_name": "01-intro.md"},
                success=False,
                result=None,
                error="Document not found",
            ),
        ]

        # Mock planning agent
        Mock()

        summary = await generate_final_summary(results)

        assert "Successfully executed 1 step(s)" in summary
        assert "Failed to execute 1 step(s)" in summary
        assert "create_document" in summary
        assert "create_chapter: Document not found" in summary

    @pytest.mark.asyncio
    async def test_summary_no_steps(self):
        """Test summary generation when no steps are executed."""
        # Mock planning agent
        Mock()

        summary = await generate_final_summary([])

        assert summary == "No steps were executed."
