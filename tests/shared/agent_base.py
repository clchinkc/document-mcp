"""
Base classes and mixins for agent testing.

This module provides a structured framework for unit, integration, and e2e
tests for both Simple and React agents, promoting code reuse and consistency.
"""

import asyncio
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import pytest
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio

from tests.shared import (
    assert_agent_response_valid,
    assert_no_error_in_response,
    create_mock_environment,
    generate_unique_name,
)


# --- Abstract Base Classes ---


class AgentTestBase(ABC):
    """
    Abstract base class defining the core interface for agent tests.
    It establishes a contract for running queries and identifying agent types.
    """

    @abstractmethod
    async def run_agent_query(self, query: str, **kwargs) -> Any:
        """Run a query against the agent being tested."""

    @abstractmethod
    def get_agent_type(self) -> str:
        """Get the string identifier for the agent type."""


# --- Test Category Base Classes ---


class UnitTestBase(AgentTestBase):
    """Base class for unit tests, focusing on isolated component testing."""

    # Unit test-specific helpers can be added here.


class IntegrationTestBase(AgentTestBase):
    """
    Base class for integration tests.
    Ensures that the agent environment is properly configured for integration tests.
    """

    @pytest.mark.asyncio
    async def test_agent_environment_setup(self) -> None:
        """Test that agent environment is properly configured."""
        api_keys = ["OPENAI_API_KEY", "GEMINI_API_KEY"]
        has_api_key = any(os.environ.get(key) for key in api_keys)
        assert has_api_key, f"No API key found. Expected one of: {api_keys}"


class E2ETestBase(AgentTestBase):
    """Base class for end-to-end tests, which use real AI services."""

    # E2E test-specific helpers can be added here.


# --- Agent-Specific Mixins ---


class SimpleAgentTestMixin:
    """
    Provides testing functionalities specific to the Simple Agent.
    Includes stateful helpers for initializing the agent and running queries.
    """

    async def initialize_simple_agent_and_mcp_server(self):
        """Initialize the Simple Agent and its MCP server for stateful testing."""
        from src.agents.simple_agent import initialize_agent_and_mcp_server

        return await initialize_agent_and_mcp_server()

    async def run_simple_query_on_agent(
        self, agent, query: str, timeout: float = 60.0
    ):
        """Run a single query on a pre-initialized Simple Agent."""
        from src.agents.simple_agent import (
            FinalAgentResponse,
            process_single_user_query,
        )

        try:
            return await asyncio.wait_for(
                process_single_user_query(agent, query), timeout=timeout
            )
        except Exception as e:
            return FinalAgentResponse(
                summary=f"Error during processing: {str(e)}",
                details=None,
                error_message=str(e),
            )


class ReactAgentTestMixin:
    """
    Provides testing functionalities specific to the React Agent.
    Includes helpers for multi-step workflow validation and stateful execution.
    """

    def assert_multi_step_workflow(
        self, history: List[Dict[str, Any]], min_steps: int = 2
    ) -> None:
        """Assert that a React agent executed a multi-step workflow."""
        assert (
            len(history) >= min_steps
        ), f"Expected at least {min_steps} steps, got {len(history)}"
        action_steps = [
            step for step in history[:-1] if step.get("action") is not None
        ]
        assert len(action_steps) > 0, "Should have at least one step with an action"

    async def initialize_react_agent_and_mcp_server(self):
        """Initialize the React Agent and its MCP server for stateful testing."""
        from src.agents.react_agent.main import (
            REACT_SYSTEM_PROMPT,
            ReActStep,
            load_llm_config,
        )

        mcp_server = MCPServerStdio(
            command="python3", args=["-m", "document_mcp.doc_tool_server", "stdio"]
        )
        model = await load_llm_config()
        agent = Agent(
            model=model,
            mcp_servers=[mcp_server],
            system_prompt=REACT_SYSTEM_PROMPT,
            output_type=ReActStep,
        )
        return agent, mcp_server

    async def run_react_query_on_agent(
        self, agent, user_query: str, max_steps: int = 10
    ) -> List[Dict[str, Any]]:
        """Run a single query on a pre-initialized React agent."""
        from src.agents.react_agent.main import (
            HistoryContextBuilder,
            execute_mcp_tool_directly,
            parse_action_string,
        )

        history = []
        context_builder = HistoryContextBuilder()
        step = 0

        while step < max_steps:
            step += 1
            if step == 1:
                current_context = f"User Query: {user_query}\n\nPlease provide your first thought and action."
            else:
                current_context = f"User Query: {user_query}\n\n{context_builder.get_context()}\n\nPlease provide your next thought and action."

            try:
                result = await agent.run(current_context)
                react_step = result.output
            except Exception as e:
                history.append(
                    {"step": step, "thought": f"Error: {e}", "action": None, "observation": str(e)}
                )
                break

            step_data = {
                "step": step,
                "thought": react_step.thought,
                "action": react_step.action,
                "observation": None,
            }

            if react_step.action and react_step.action.strip():
                try:
                    tool_name, kwargs = parse_action_string(react_step.action)
                    observation = await execute_mcp_tool_directly(agent, tool_name, kwargs)
                    step_data["observation"] = observation
                except Exception as e:
                    step_data["observation"] = f"Error executing action: {e}"
            else:
                step_data["observation"] = "Task completed."
                history.append(step_data)
                break

            history.append(step_data)
            context_builder.add_step(step_data)

        return history


class IntegrationTestBase(AgentTestBase):
    """
    Base class for integration tests.

    This class provides common functionality for integration tests
    that use real MCP servers with mocked AI responses.
    """

    def setup_mock_environment(self, api_key_type: str = "openai") -> Dict[str, str]:
        """
        Set up mock environment for integration testing.

        Args:
            api_key_type: Type of API key to mock

        Returns:
            Environment variables dictionary
        """
        return create_mock_environment(
            api_key_type=api_key_type,
            include_server_config=True,
        )

    async def test_agent_package_imports(self) -> None:
        """Test that all required packages can be imported."""
        try:
            import pydantic_ai

            assert hasattr(
                pydantic_ai, "Agent"
            ), "pydantic_ai should provide Agent class"
        except ImportError:
            pytest.fail("Failed to import pydantic_ai - required for agent")


class UnitTestBase(AgentTestBase):
    """
    Base class for unit tests.

    This class provides common functionality for unit tests
    that mock all external dependencies.
    """

    def setup_complete_mock_environment(self) -> None:
        """Set up completely mocked environment for unit testing."""
        # Mock all external dependencies
        pass  # Implementation depends on specific mocking needs

    def create_mock_response_data(self, **kwargs) -> Dict[str, Any]:
        """
        Create mock response data for testing.

        Args:
            **kwargs: Additional response data

        Returns:
            Mock response data dictionary
        """
        default_data = {
            "summary": "Mock response",
            "details": None,
            "error_message": None,
        }
        default_data.update(kwargs)
        return default_data


class E2ETestBase(AgentTestBase):
    """
    Base class for end-to-end tests.

    This class provides common functionality for E2E tests
    that use real MCP servers and real AI APIs.
    """

    def check_real_api_key_available(self) -> bool:
        """
        Check if a real API key is available for E2E testing.

        Returns:
            True if real API key is available
        """
        api_keys = ["OPENAI_API_KEY", "GEMINI_API_KEY"]
        for key in api_keys:
            value = os.environ.get(key, "").strip()
            if value and value != "test_key" and not value.startswith("sk-test"):
                return True
        return False

    def skip_if_no_real_api_key(self) -> None:
        """Skip test if no real API key is available."""
        if not self.check_real_api_key_available():
            pytest.skip(
                "E2E tests require a real API key "
                "(OPENAI_API_KEY or GEMINI_API_KEY)"
            )


# Utility functions for common test patterns


def create_agent_test_class(agent_type: str, test_type: str = "integration"):
    """
    Factory function to create agent test classes.

    Args:
        agent_type: Type of agent (simple, react)
        test_type: Type of test (unit, integration, e2e)

    Returns:
        Configured test class
    """
    # Base class selection
    if test_type == "unit":
        base_class = UnitTestBase
    elif test_type == "integration":
        base_class = IntegrationTestBase
    elif test_type == "e2e":
        base_class = E2ETestBase
    else:
        raise ValueError(f"Unknown test type: {test_type}")

    # Mixin selection
    if agent_type == "simple":
        mixin_class = SimpleAgentTestMixin
    elif agent_type == "react":
        mixin_class = ReactAgentTestMixin
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    # Create dynamic class
    class_name = f"{agent_type.title()}Agent{test_type.title()}TestBase"

    class DynamicTestClass(mixin_class, base_class):
        def get_agent_type(self) -> str:
            return f"{agent_type}_agent"

        async def run_agent_query(self, query: str, **kwargs) -> Any:
            if agent_type == "simple":
                return await self.run_simple_agent_test(query, **kwargs)
            elif agent_type == "react":
                return await self.run_react_agent_test(query, **kwargs)

    DynamicTestClass.__name__ = class_name
    DynamicTestClass.__qualname__ = class_name

    return DynamicTestClass
