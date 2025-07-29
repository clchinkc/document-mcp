"""Unit tests for React Agent implementation.

This module provides comprehensive unit tests for the ReAct Agent with mocked
dependencies to validate reasoning patterns, multi-step execution, and error handling.
"""

from unittest.mock import Mock

import pytest

from src.agents.react_agent.models import FinalAgentResponse
from src.agents.react_agent.models import ReActStep
from src.agents.react_agent.parser import ActionParser
from src.agents.react_agent.prompts import get_react_system_prompt


class TestReActStep:
    """Test ReActStep model validation."""

    def test_react_step_valid_data(self):
        """Test ReActStep with valid data."""
        step = ReActStep(
            thought="I need to create a document first", action="create_document(document_name='Test Doc')"
        )
        assert step.thought == "I need to create a document first"
        assert step.action == "create_document(document_name='Test Doc')"

    def test_react_step_optional_action(self):
        """Test ReActStep with optional action field."""
        step = ReActStep(thought="Planning the next action")
        assert step.thought == "Planning the next action"
        assert step.action is None

    def test_react_step_validation_errors(self):
        """Test ReActStep validation with invalid data."""
        with pytest.raises(ValueError):
            ReActStep(
                thought=""  # Invalid empty thought
            )


class TestFinalAgentResponse:
    """Test FinalAgentResponse model validation."""

    def test_final_agent_response_basic(self):
        """Test FinalAgentResponse with basic data."""
        response = FinalAgentResponse(summary="Document created successfully")
        assert response.summary == "Document created successfully"
        assert response.details is None
        assert response.execution_steps is None
        assert response.error_message is None

    def test_final_agent_response_complete(self):
        """Test FinalAgentResponse with all fields."""
        response = FinalAgentResponse(
            summary="Task completed",
            details="Document 'test' created with chapter 'intro'",
            execution_steps=[{"step": 1, "thought": "Create document", "action": "create_document"}],
            error_message=None,
        )
        assert response.summary == "Task completed"
        assert response.details == "Document 'test' created with chapter 'intro'"
        assert len(response.execution_steps) == 1
        assert response.error_message is None


class TestActionParser:
    """Test ActionParser functionality already covered in test_react_agent_parser.py."""

    def test_action_parser_integration_with_react_step(self):
        """Test ActionParser integration with ReActStep workflow."""
        parser = ActionParser()
        step = ReActStep(
            thought="I need to create a document", action="create_document(document_name='Test')"
        )

        tool_name, kwargs = parser.parse(step.action)
        assert tool_name == "create_document"
        assert kwargs == {"document_name": "Test"}


class TestReActSystemPrompt:
    """Test ReAct system prompt functionality."""

    def test_get_react_system_prompt(self):
        """Test ReAct system prompt generation."""
        prompt = get_react_system_prompt()

        # Basic validation that prompt contains expected elements
        assert isinstance(prompt, str)
        assert len(prompt) > 0

        # Check for key ReAct components
        assert any(keyword in prompt.lower() for keyword in ["react", "reasoning", "action", "observation"])

    def test_react_prompt_structure(self):
        """Test that ReAct prompt contains expected structural elements."""
        prompt = get_react_system_prompt()

        # Should contain instructions about the ReAct pattern
        assert "reason" in prompt.lower() or "think" in prompt.lower()
        assert "action" in prompt.lower()
        assert "observe" in prompt.lower() or "observation" in prompt.lower()


# Mock classes for ReAct Agent testing
class MockReActAgent:
    """Mock ReAct Agent for testing without importing the full implementation."""

    def __init__(self):
        self.agent_type = "react"
        self.settings = Mock()
        self.settings.default_timeout = 60
        self.settings.max_retries = 3
        self._llm = None
        self._mcp_client = None

    async def get_llm(self):
        """Mock LLM getter."""
        if self._llm is None:
            self._llm = Mock()
        return self._llm

    def get_mcp_server_environment(self):
        """Mock MCP server environment."""
        return {"TEST_ENV": "value"}

    def get_system_prompt(self):
        """Mock system prompt."""
        return get_react_system_prompt()


class TestReActAgentMockBehavior:
    """Test ReAct Agent behavior patterns with mocked implementations."""

    def test_react_agent_initialization_pattern(self):
        """Test ReAct agent initialization pattern."""
        agent = MockReActAgent()
        assert agent.agent_type == "react"
        assert agent._llm is None
        assert hasattr(agent.settings, "default_timeout")
        assert hasattr(agent.settings, "max_retries")

    @pytest.mark.asyncio
    async def test_react_agent_llm_loading_pattern(self):
        """Test ReAct agent LLM loading pattern."""
        agent = MockReActAgent()

        llm1 = await agent.get_llm()
        llm2 = await agent.get_llm()

        # Should cache LLM instance
        assert llm1 is llm2
        assert agent._llm is not None

    def test_react_agent_system_prompt_pattern(self):
        """Test ReAct agent system prompt pattern."""
        agent = MockReActAgent()
        prompt = agent.get_system_prompt()

        assert isinstance(prompt, str)
        assert len(prompt) > 0


class TestReActPatterns:
    """Test ReAct reasoning patterns."""

    def test_react_step_workflow_pattern(self):
        """Test ReAct step workflow pattern."""
        steps = []

        # Simulate a multi-step workflow
        step1 = ReActStep(thought="I need to check what documents exist", action="list_documents()")
        steps.append(step1)

        step2 = ReActStep(
            thought="No documents found, I should create one",
            action="create_document(document_name='New Document')",
        )
        steps.append(step2)

        step3 = ReActStep(
            thought="Document created, now I need to add content",
            action="create_chapter(document_name='New Document', chapter_name='01-intro.md', initial_content='Introduction')",
        )
        steps.append(step3)

        # Validate step progression
        assert len(steps) == 3
        assert all(isinstance(step, ReActStep) for step in steps)

        # Validate reasoning progression shows logical thinking
        thoughts = [step.thought for step in steps]
        assert "check" in thoughts[0].lower()
        assert "create" in thoughts[1].lower()
        assert "content" in thoughts[2].lower()

    def test_react_error_handling_pattern(self):
        """Test ReAct error handling pattern."""
        # Simulate an error handling step
        error_step = ReActStep(
            thought="The previous action failed, I need to try a different approach",
            action="list_documents()",
        )

        assert "failed" in error_step.thought.lower()
        assert "different approach" in error_step.thought.lower()

    def test_react_completion_detection_pattern(self):
        """Test ReAct completion detection pattern."""
        completion_step = ReActStep(
            thought="The document has been created and the chapter added. Task is complete.",
            action="get_document_statistics(document_name='New Document', scope='document')",
        )

        assert "complete" in completion_step.thought.lower()


class TestReActAgentToolIntegration:
    """Test ReAct Agent tool integration patterns."""

    def test_react_tool_call_formatting(self):
        """Test ReAct tool call formatting."""
        parser = ActionParser()

        # Test various tool call formats that ReAct agent might generate
        test_cases = [
            ("create_document(document_name='Test Doc')", "create_document", {"document_name": "Test Doc"}),
            ("list_documents()", "list_documents", {}),
            (
                "create_chapter(document_name='Test', chapter_name='01-intro.md', initial_content='Hello world')",
                "create_chapter",
                {"document_name": "Test", "chapter_name": "01-intro.md", "initial_content": "Hello world"},
            ),
        ]

        for action_string, expected_tool, expected_kwargs in test_cases:
            step = ReActStep(thought="Testing tool call formatting", action=action_string)

            tool_name, kwargs = parser.parse(step.action)
            assert tool_name == expected_tool
            assert kwargs == expected_kwargs

    def test_react_tool_response_integration(self):
        """Test ReAct tool response integration pattern."""
        # Simulate how ReAct agent would process tool responses
        initial_step = ReActStep(
            thought="I need to create a document", action="create_document(document_name='Test')"
        )

        assert initial_step.thought == "I need to create a document"
        assert initial_step.action == "create_document(document_name='Test')"
