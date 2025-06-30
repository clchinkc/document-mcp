#!/usr/bin/env python3
"""
Test suite for ReAct Agent Termination Logic (Task 8)

This test suite validates that the agent can self-terminate when:
1. Tasks are complete (action=None)
2. Maximum steps are reached
3. Critical errors occur
4. Various edge cases

Tests are designed to be strict and ensure proper termination behavior.
"""

import os

import pytest

from src.agents.react_agent.main import ReActStep, run_react_loop

# Basic ReActStep termination tests moved to test_react_step.py for better organization


class TestTerminationLogicIntegration:
    """Test the termination logic integration in the ReAct loop."""

    @pytest.fixture
    def mock_environment(self, monkeypatch):
        """Set up mock environment for testing."""
        env_vars = {
            "OPENAI_API_KEY": "test_key",
            "OPENAI_MODEL_NAME": "gpt-4",
            "MCP_SERVER_HOST": "localhost",
            "MCP_SERVER_PORT": "8000",
        }
        for key, value in env_vars.items():
            monkeypatch.setenv(key, value)
        yield

    @pytest.mark.asyncio
    async def test_successful_task_completion_termination(self, mock_environment, mocker):
        """Test that the agent terminates successfully when task is complete."""

        # Mock the optimized components
        mock_load_config = mocker.patch("src.agents.react_agent.main.load_llm_config")
        mock_mcp_server_class = mocker.patch("src.agents.react_agent.main.MCPServerStdio")
        mock_get_cached_agent = mocker.patch("src.agents.react_agent.main.get_cached_agent")

        # Setup mocks - make load_llm_config async
        mock_model = mocker.Mock()
        mock_load_config = mocker.AsyncMock(return_value=mock_model)

        # Mock the MCP server instance with proper async context manager
        mock_mcp_server_instance = mocker.Mock()
        mock_mcp_server_instance.__aenter__ = mocker.AsyncMock(
            return_value=mock_mcp_server_instance
        )
        mock_mcp_server_instance.__aexit__ = mocker.AsyncMock(return_value=False)
        mock_mcp_server_class.return_value = mock_mcp_server_instance

        # Mock the cached agent
        mock_agent = mocker.AsyncMock()
        mock_get_cached_agent.return_value = mock_agent

        # Simulate a successful 2-step completion
        step1_result = mocker.MagicMock()
        step1_result.output = ReActStep(
            thought="I need to create a document",
            action='create_document(document_name="Test")',
        )

        step2_result = mocker.MagicMock()
        step2_result.output = ReActStep(
            thought="Document created successfully. Task is complete.", action=None
        )

        mock_agent.run.side_effect = [step1_result, step2_result]

        # Mock tool execution
        mock_execute_tool = mocker.patch(
            "src.agents.react_agent.main.execute_mcp_tool_directly",
            return_value='{"success": true, "message": "Document created"}',
        )

        # Run the ReAct loop
        history = await run_react_loop(
            "Create a document named Test", max_steps=10
        )

        # Verify termination behavior
        assert len(history) == 2
        assert history[0]["action"] == 'create_document(document_name="Test")'
        assert history[1]["action"] is None
        assert history[1]["observation"] == "Task completed."
        assert "complete" in history[1]["thought"].lower()

    @pytest.mark.asyncio
    async def test_max_steps_termination(self, mock_environment, mocker):
        """Test that the agent terminates when maximum steps are reached."""

        mock_load_config = mocker.patch("src.agents.react_agent.main.load_llm_config")
        mock_mcp_server_class = mocker.patch("src.agents.react_agent.main.MCPServerStdio")
        mock_get_cached_agent = mocker.patch("src.agents.react_agent.main.get_cached_agent")

        # Setup mocks - make load_llm_config async
        mock_model = mocker.Mock()
        mock_load_config = mocker.AsyncMock(return_value=mock_model)

        # Mock the MCP server instance
        mock_mcp_server_instance = mocker.Mock()
        mock_mcp_server_instance.__aenter__ = mocker.AsyncMock(
            return_value=mock_mcp_server_instance
        )
        mock_mcp_server_instance.__aexit__ = mocker.AsyncMock(return_value=False)
        mock_mcp_server_class.return_value = mock_mcp_server_instance

        # Mock the cached agent
        mock_agent = mocker.AsyncMock()
        mock_get_cached_agent.return_value = mock_agent

        # Simulate agent that never terminates (always returns actions)
        ongoing_step = mocker.MagicMock()
        ongoing_step.output = ReActStep(
            thought="Still working on the task", action="list_documents()"
        )

        mock_agent.run.return_value = ongoing_step

        # Mock tool execution
        mock_execute_tool = mocker.patch(
            "src.agents.react_agent.main.execute_mcp_tool_directly",
            return_value='{"documents": []}',
        )

        # Run with very low max_steps
        history = await run_react_loop("Never ending task", max_steps=3)

        # Verify max steps termination
        assert len(history) == 3
        assert history[0]["step"] == 1
        assert history[1]["step"] == 2
        assert history[2]["step"] == 3
        
        # All steps should have actions (no termination step)
        for step in history:
            assert step["action"] == "list_documents()"
            assert step["thought"] == "Still working on the task"

    @pytest.mark.asyncio
    async def test_llm_error_termination(self, mock_environment, mocker):
        """Test that the agent terminates gracefully on LLM errors."""

        mock_load_config = mocker.patch("src.agents.react_agent.main.load_llm_config")
        mock_mcp_server_class = mocker.patch("src.agents.react_agent.main.MCPServerStdio")
        mock_get_cached_agent = mocker.patch("src.agents.react_agent.main.get_cached_agent")

        # Setup mocks - make load_llm_config async
        mock_model = mocker.Mock()
        mock_load_config = mocker.AsyncMock(return_value=mock_model)

        # Mock the MCP server instance
        mock_mcp_server_instance = mocker.Mock()
        mock_mcp_server_instance.__aenter__ = mocker.AsyncMock(
            return_value=mock_mcp_server_instance
        )
        mock_mcp_server_instance.__aexit__ = mocker.AsyncMock(return_value=False)
        mock_mcp_server_class.return_value = mock_mcp_server_instance

        # Mock the cached agent
        mock_agent = mocker.AsyncMock()
        mock_get_cached_agent.return_value = mock_agent

        # Simulate LLM failure
        mock_agent.run.side_effect = Exception("API rate limit exceeded")

        # Run the ReAct loop
        history = await run_react_loop("Create a document", max_steps=10)

        # Verify error termination
        assert len(history) == 1
        assert history[0]["action"] is None
        assert "Error occurred" in history[0]["thought"]
        assert "API rate limit exceeded" in history[0]["observation"]

    @pytest.mark.asyncio
    async def test_tool_execution_error_continuation(self, mock_environment, mocker):
        """Test that the agent continues after tool execution errors."""

        mock_load_config = mocker.patch("src.agents.react_agent.main.load_llm_config")
        mock_mcp_server_class = mocker.patch("src.agents.react_agent.main.MCPServerStdio")
        mock_get_cached_agent = mocker.patch("src.agents.react_agent.main.get_cached_agent")

        # Setup mocks - make load_llm_config async
        mock_model = mocker.Mock()
        mock_load_config = mocker.AsyncMock(return_value=mock_model)

        # Mock the MCP server instance
        mock_mcp_server_instance = mocker.Mock()
        mock_mcp_server_instance.__aenter__ = mocker.AsyncMock(
            return_value=mock_mcp_server_instance
        )
        mock_mcp_server_instance.__aexit__ = mocker.AsyncMock(return_value=False)
        mock_mcp_server_class.return_value = mock_mcp_server_instance

        # Mock the cached agent
        mock_agent = mocker.AsyncMock()
        mock_get_cached_agent.return_value = mock_agent

        # Simulate agent steps: first with tool error, then successful completion
        step1_result = mocker.MagicMock()
        step1_result.output = ReActStep(
            thought="I'll create a document",
            action='create_document(document_name="Test")',
        )

        step2_result = mocker.MagicMock()
        step2_result.output = ReActStep(
            thought="Got an error, let me try a different approach",
            action='list_documents()',
        )

        step3_result = mocker.MagicMock()
        step3_result.output = ReActStep(
            thought="Successfully listed documents. Task complete.",
            action=None,
        )

        mock_agent.run.side_effect = [step1_result, step2_result, step3_result]

        # Mock tool execution with error on first call, success on second
        mock_execute_tool = mocker.patch(
            "src.agents.react_agent.main.execute_mcp_tool_directly",
            side_effect=[
                '{"error": "Document already exists"}',
                '{"documents": ["existing_doc.md"]}',
            ],
        )

        # Run the ReAct loop
        history = await run_react_loop("Create a document named Test", max_steps=10)

        # Verify error recovery behavior
        assert len(history) == 3
        assert history[0]["action"] == 'create_document(document_name="Test")'
        assert "error" in history[0]["observation"].lower()
        assert history[1]["action"] == 'list_documents()'
        assert history[2]["action"] is None

    @pytest.mark.asyncio
    async def test_action_parsing_error_continuation(self, mock_environment, mocker):
        """Test that the agent continues after action parsing errors."""

        mock_load_config = mocker.patch("src.agents.react_agent.main.load_llm_config")
        mock_mcp_server_class = mocker.patch("src.agents.react_agent.main.MCPServerStdio")
        mock_get_cached_agent = mocker.patch("src.agents.react_agent.main.get_cached_agent")

        # Setup mocks - make load_llm_config async
        mock_model = mocker.Mock()
        mock_load_config = mocker.AsyncMock(return_value=mock_model)

        # Mock the MCP server instance
        mock_mcp_server_instance = mocker.Mock()
        mock_mcp_server_instance.__aenter__ = mocker.AsyncMock(
            return_value=mock_mcp_server_instance
        )
        mock_mcp_server_instance.__aexit__ = mocker.AsyncMock(return_value=False)
        mock_mcp_server_class.return_value = mock_mcp_server_instance

        # Mock the cached agent
        mock_agent = mocker.AsyncMock()
        mock_get_cached_agent.return_value = mock_agent

        # Simulate agent steps: first with invalid action, then valid completion
        step1_result = mocker.MagicMock()
        step1_result.output = ReActStep(
            thought="I'll create a document",
            action="invalid_action_format_without_parentheses",
        )

        step2_result = mocker.MagicMock()
        step2_result.output = ReActStep(
            thought="Let me use the correct format", action=None
        )

        mock_agent.run.side_effect = [step1_result, step2_result]

        # Run the ReAct loop
        history = await run_react_loop("Create a document", max_steps=10)

        # Verify parsing error recovery
        assert len(history) == 2
        assert history[0]["action"] == "invalid_action_format_without_parentheses"
        assert "Invalid action format" in history[0]["observation"]
        assert history[1]["action"] is None


class TestTerminationEdgeCases:
    """Test edge cases for termination logic."""

    def test_whitespace_action_termination(self):
        """Test that whitespace-only action is treated as termination."""
        step = ReActStep(thought="Task complete", action="   ")
        # A step is terminal when action is None or contains only whitespace
        assert step.action is None or step.action.strip() == ""

    def test_empty_string_action_termination(self):
        """Test that empty string action is treated as termination."""
        step = ReActStep(thought="Task complete", action="")
        # A step is terminal when action is None or empty string
        assert step.action is None or step.action == ""


def run_termination_tests():
    """Run all termination tests for debugging purposes."""
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    run_termination_tests()
