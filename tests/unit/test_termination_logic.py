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
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from src.agents.react_agent.main import ReActStep, run_react_loop

# Basic ReActStep termination tests moved to test_react_step.py for better organization


class TestTerminationLogicIntegration:
    """Test the termination logic integration in the ReAct loop."""

    @pytest.fixture
    def mock_environment(self):
        """Set up mock environment for testing."""
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "test_key",
                "OPENAI_MODEL_NAME": "gpt-4",
                "MCP_SERVER_HOST": "localhost",
                "MCP_SERVER_PORT": "8000",
            },
        ):
            yield

    @pytest.mark.asyncio
    async def test_successful_task_completion_termination(self, mock_environment):
        """Test that the agent terminates successfully when task is complete."""

        # Mock the optimized components
        with patch(
            "src.agents.react_agent.main.load_llm_config"
        ) as mock_load_config, patch(
            "src.agents.react_agent.main.MCPServerSSE"
        ) as mock_mcp_server_class, patch(
            "src.agents.react_agent.main.get_cached_agent"
        ) as mock_get_cached_agent:

            # Setup mocks - make load_llm_config async
            mock_model = Mock()
            mock_load_config = AsyncMock(return_value=mock_model)

            # Mock the MCP server instance with proper async context manager
            mock_mcp_server_instance = Mock()
            mock_mcp_server_instance.__aenter__ = AsyncMock(
                return_value=mock_mcp_server_instance
            )
            mock_mcp_server_instance.__aexit__ = AsyncMock(return_value=False)
            mock_mcp_server_class.return_value = mock_mcp_server_instance

            # Mock the cached agent
            mock_agent = AsyncMock()
            mock_get_cached_agent.return_value = mock_agent

            # Simulate a successful 2-step completion
            step1_result = MagicMock()
            step1_result.output = ReActStep(
                thought="I need to create a document",
                action='create_document(document_name="Test")',
            )

            step2_result = MagicMock()
            step2_result.output = ReActStep(
                thought="Document created successfully. Task is complete.", action=None
            )

            mock_agent.run.side_effect = [step1_result, step2_result]

            # Mock tool execution
            with patch(
                "src.agents.react_agent.main.execute_mcp_tool_directly",
                return_value='{"success": true, "message": "Document created"}',
            ):

                # Run the ReAct loop
                history = await run_react_loop(
                    "Create a document named Test", max_steps=10
                )

                # Verify termination behavior
                assert len(history) == 2
                assert history[0]["action"] == 'create_document(document_name="Test")'
                assert history[1]["action"] is None
                assert history[1]["observation"] == "Task completed successfully"

    @pytest.mark.asyncio
    async def test_max_steps_termination(self, mock_environment):
        """Test that the agent terminates when maximum steps are reached."""

        with patch(
            "src.agents.react_agent.main.load_llm_config"
        ) as mock_load_config, patch(
            "src.agents.react_agent.main.MCPServerSSE"
        ) as mock_mcp_server_class, patch(
            "src.agents.react_agent.main.get_cached_agent"
        ) as mock_get_cached_agent:

            # Setup mocks - make load_llm_config async
            mock_model = Mock()
            mock_load_config = AsyncMock(return_value=mock_model)

            # Mock the MCP server instance
            mock_mcp_server_instance = Mock()
            mock_mcp_server_instance.__aenter__ = AsyncMock(
                return_value=mock_mcp_server_instance
            )
            mock_mcp_server_instance.__aexit__ = AsyncMock(return_value=False)
            mock_mcp_server_class.return_value = mock_mcp_server_instance

            # Mock the cached agent
            mock_agent = AsyncMock()
            mock_get_cached_agent.return_value = mock_agent

            # Simulate agent that never terminates (always returns actions)
            ongoing_step = MagicMock()
            ongoing_step.output = ReActStep(
                thought="Still working on the task", action="list_documents()"
            )

            mock_agent.run.return_value = ongoing_step

            # Mock tool execution
            with patch(
                "src.agents.react_agent.main.execute_mcp_tool_directly",
                return_value='{"documents": []}',
            ):

                # Run with very low max_steps
                history = await run_react_loop("Never ending task", max_steps=3)

                # Verify max steps termination
                assert len(history) == 4  # 3 steps + 1 timeout step
                assert history[-1]["step"] == 4  # max_steps + 1
                assert history[-1]["action"] is None
                assert "Maximum steps" in history[-1]["thought"]
                assert "Task incomplete" in history[-1]["observation"]

    @pytest.mark.asyncio
    async def test_llm_error_termination(self, mock_environment):
        """Test that the agent terminates gracefully on LLM errors."""

        with patch(
            "src.agents.react_agent.main.load_llm_config"
        ) as mock_load_config, patch(
            "src.agents.react_agent.main.MCPServerSSE"
        ) as mock_mcp_server_class, patch(
            "src.agents.react_agent.main.get_cached_agent"
        ) as mock_get_cached_agent:

            # Setup mocks - make load_llm_config async
            mock_model = Mock()
            mock_load_config = AsyncMock(return_value=mock_model)

            # Mock the MCP server instance
            mock_mcp_server_instance = Mock()
            mock_mcp_server_instance.__aenter__ = AsyncMock(
                return_value=mock_mcp_server_instance
            )
            mock_mcp_server_instance.__aexit__ = AsyncMock(return_value=False)
            mock_mcp_server_class.return_value = mock_mcp_server_instance

            # Mock the cached agent
            mock_agent = AsyncMock()
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
    async def test_tool_execution_error_continuation(self, mock_environment):
        """Test that the agent continues after tool execution errors."""

        with patch(
            "src.agents.react_agent.main.load_llm_config"
        ) as mock_load_config, patch(
            "src.agents.react_agent.main.MCPServerSSE"
        ) as mock_mcp_server_class, patch(
            "src.agents.react_agent.main.get_cached_agent"
        ) as mock_get_cached_agent:

            # Setup mocks - make load_llm_config async
            mock_model = Mock()
            mock_load_config = AsyncMock(return_value=mock_model)

            # Mock the MCP server instance
            mock_mcp_server_instance = Mock()
            mock_mcp_server_instance.__aenter__ = AsyncMock(
                return_value=mock_mcp_server_instance
            )
            mock_mcp_server_instance.__aexit__ = AsyncMock(return_value=False)
            mock_mcp_server_class.return_value = mock_mcp_server_instance

            # Mock the cached agent
            mock_agent = AsyncMock()
            mock_get_cached_agent.return_value = mock_agent

            # Simulate: error step, then successful completion
            error_step = MagicMock()
            error_step.output = ReActStep(
                thought="I'll try to create a document",
                action='create_document(document_name="Test")',
            )

            completion_step = MagicMock()
            completion_step.output = ReActStep(
                thought="There was an error, but I'll handle it and complete the task",
                action=None,
            )

            mock_agent.run.side_effect = [error_step, completion_step]

            # Mock tool execution failure for first call
            with patch(
                "src.agents.react_agent.main.execute_mcp_tool_directly",
                side_effect=Exception("Tool execution failed"),
            ):

                # Run the ReAct loop
                history = await run_react_loop("Create a document", max_steps=10)

                # Verify error handling and continuation
                assert len(history) == 2
                assert "Tool execution failed" in history[0]["observation"]
                assert (
                    history[1]["action"] is None
                )  # Agent terminated after handling error

    @pytest.mark.asyncio
    async def test_action_parsing_error_continuation(self, mock_environment):
        """Test that the agent continues after action parsing errors."""

        with patch(
            "src.agents.react_agent.main.load_llm_config"
        ) as mock_load_config, patch(
            "src.agents.react_agent.main.MCPServerSSE"
        ) as mock_mcp_server_class, patch(
            "src.agents.react_agent.main.get_cached_agent"
        ) as mock_get_cached_agent:

            # Setup mocks - make load_llm_config async
            mock_model = Mock()
            mock_load_config = AsyncMock(return_value=mock_model)

            # Mock the MCP server instance
            mock_mcp_server_instance = Mock()
            mock_mcp_server_instance.__aenter__ = AsyncMock(
                return_value=mock_mcp_server_instance
            )
            mock_mcp_server_instance.__aexit__ = AsyncMock(return_value=False)
            mock_mcp_server_class.return_value = mock_mcp_server_instance

            # Mock the cached agent
            mock_agent = AsyncMock()
            mock_get_cached_agent.return_value = mock_agent

            # Simulate: invalid action, then termination
            invalid_action_step = MagicMock()
            invalid_action_step.output = ReActStep(
                thought="I'll try an action",
                action="invalid_action_format(",  # Malformed action
            )

            completion_step = MagicMock()
            completion_step.output = ReActStep(
                thought="I had a parsing error, but I'll complete now", action=None
            )

            mock_agent.run.side_effect = [invalid_action_step, completion_step]

            # Run the ReAct loop
            history = await run_react_loop("Test invalid action", max_steps=10)

            # Verify parsing error handling and continuation
            assert len(history) == 2
            assert "Invalid action format" in history[0]["observation"]
            assert history[1]["action"] is None  # Agent terminated after handling error


class TestTerminationEdgeCases:
    """Test edge cases for termination logic."""

    def test_whitespace_action_termination(self):
        """Test that whitespace-only actions are treated as valid actions, not termination."""
        step = ReActStep(thought="Working", action="   ")
        assert step.action is not None
        assert step.action == "   "

    def test_empty_string_action_termination(self):
        """Test that empty string actions are treated as valid actions, not termination."""
        step = ReActStep(thought="Working", action="")
        assert step.action is not None
        assert step.action == ""


def run_termination_tests():
    """Run all termination logic tests."""
    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    print("Running ReAct Agent Termination Logic Tests (Task 8)")
    print("=" * 60)
    run_termination_tests()
