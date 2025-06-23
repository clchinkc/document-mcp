"""
Base classes for agent testing.

This module provides common base classes and utilities for testing
both Simple and React agents, reducing code duplication and ensuring
consistent testing patterns.
"""

import asyncio
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import pytest

from tests.shared import (
    assert_agent_response_valid,
    assert_no_error_in_response,
    create_mock_environment,
    generate_unique_name,
)


class AgentTestBase(ABC):
    """
    Abstract base class for agent testing.

    This class provides common functionality for testing both
    Simple and React agents, including environment setup,
    common test patterns, and assertion helpers.
    """

    @abstractmethod
    async def run_agent_query(self, query: str, **kwargs) -> Any:
        """
        Run a query against the agent being tested.

        Args:
            query: Query string to send to the agent
            **kwargs: Additional arguments for the agent

        Returns:
            Agent response object
        """

    @abstractmethod
    def get_agent_type(self) -> str:
        """
        Get the type of agent being tested.

        Returns:
            String identifier for the agent type
        """

    def assert_valid_response(
        self, response: Any, require_details: bool = False
    ) -> None:
        """
        Assert that an agent response is valid.

        Args:
            response: Response to validate
            require_details: Whether details field is required
        """
        assert_agent_response_valid(
            response,
            response_type=self.get_agent_type(),
            require_details=require_details,
        )

    def assert_successful_response(self, response: Any) -> None:
        """
        Assert that a response indicates success.

        Args:
            response: Response to check
        """
        self.assert_valid_response(response)
        assert_no_error_in_response(response)

    async def test_document_creation(self, test_docs_root) -> str:
        """
        Test basic document creation functionality.

        Args:
            test_docs_root: Test document root directory

        Returns:
            Name of created document
        """
        doc_name = generate_unique_name("test_doc")

        response = await self.run_agent_query(
            f"Create a new document named '{doc_name}'"
        )

        self.assert_successful_response(response)
        return doc_name

    async def test_document_listing(self, test_docs_root) -> List[Any]:
        """
        Test document listing functionality.

        Args:
            test_docs_root: Test document root directory

        Returns:
            List of documents from response
        """
        response = await self.run_agent_query("List all available documents")

        self.assert_valid_response(response)
        assert isinstance(response.details, list), "Document list should be a list"

        return response.details

    async def test_chapter_operations(self, test_docs_root, doc_name: str) -> None:
        """
        Test chapter creation and reading operations.

        Args:
            test_docs_root: Test document root directory
            doc_name: Name of document to work with
        """
        chapter_name = "01-test.md"
        chapter_content = "# Test Chapter\n\nThis is test content."

        # Create chapter
        create_response = await self.run_agent_query(
            f"Create a chapter named '{chapter_name}' in document '{doc_name}' "
            f"with content: {chapter_content}"
        )
        self.assert_successful_response(create_response)

        # Read chapter
        read_response = await self.run_agent_query(
            f"Read chapter '{chapter_name}' from document '{doc_name}'"
        )
        self.assert_valid_response(read_response)

    async def test_document_statistics(self, test_docs_root, doc_name: str) -> None:
        """
        Test document statistics functionality.

        Args:
            test_docs_root: Test document root directory
            doc_name: Name of document to get statistics for
        """
        response = await self.run_agent_query(
            f"Get statistics for document '{doc_name}'"
        )

        self.assert_valid_response(response)

    async def test_search_functionality(self, test_docs_root, doc_name: str) -> None:
        """
        Test text search functionality.

        Args:
            test_docs_root: Test document root directory
            doc_name: Name of document to search in
        """
        search_term = "test"

        response = await self.run_agent_query(
            f"Find the text '{search_term}' in document '{doc_name}'"
        )

        self.assert_valid_response(response)


class SimpleAgentTestMixin:
    """
    Mixin class for Simple Agent specific testing functionality.

    This mixin provides functionality specific to testing the Simple Agent,
    including response format validation and single-step operation patterns.
    """

    def assert_simple_agent_response(self, response) -> None:
        """
        Assert that a response matches Simple Agent format.

        Args:
            response: Response to validate
        """
        # Simple agent responses should have specific structure
        assert hasattr(response, "summary"), "Simple agent response missing summary"
        assert hasattr(response, "details"), "Simple agent response missing details"
        assert hasattr(
            response, "error_message"
        ), "Simple agent response missing error_message"

        # Summary should be a non-empty string
        assert isinstance(response.summary, str), "Summary must be a string"
        assert len(response.summary) > 0, "Summary should not be empty"

    async def run_simple_agent_test(self, query: str, timeout: float = 30.0):
        """
        Run a test query against the Simple Agent.

        Args:
            query: Query to send to the agent
            timeout: Timeout for the operation

        Returns:
            Agent response
        """
        from src.agents.simple_agent import (
            FinalAgentResponse,
            initialize_agent_and_mcp_server,
            process_single_user_query,
        )

        try:
            agent, _ = await initialize_agent_and_mcp_server()
            async with agent.run_mcp_servers():
                result = await asyncio.wait_for(
                    process_single_user_query(agent, query), timeout=timeout
                )
                return result
        except Exception as e:
            return FinalAgentResponse(
                summary=f"Error during processing: {str(e)}",
                details=None,
                error_message=str(e),
            )

    async def test_model_integration(
        self, test_docs_root, model_type: str = "openai"
    ) -> None:
        """
        Test model-specific integration patterns.

        Args:
            test_docs_root: Test document root directory
            model_type: Type of model to test (openai, gemini)
        """
        doc_name = generate_unique_name(f"{model_type}_test_doc")

        # Test document creation with specific model
        create_response = await self.run_simple_agent_test(
            f"Create a new document named '{doc_name}'"
        )

        self.assert_simple_agent_response(create_response)
        assert (
            create_response.error_message is None
        ), f"{model_type} model should not produce error messages for valid requests"

        # Test listing to verify creation
        list_response = await self.run_simple_agent_test(
            "Show me all available documents"
        )
        self.assert_simple_agent_response(list_response)
        assert isinstance(
            list_response.details, list
        ), f"{model_type} model should return list of documents"


class ReactAgentTestMixin:
    """
    Mixin class for React Agent specific testing functionality.

    This mixin provides functionality specific to testing the React Agent,
    including multi-step workflow validation and reasoning pattern checks.
    """

    def assert_react_agent_response(self, history: List[Dict[str, Any]]) -> None:
        """
        Assert that a response matches React Agent format.

        Args:
            history: React agent execution history
        """
        assert isinstance(history, list), "React agent should return execution history"
        assert len(history) > 0, "React agent history should not be empty"

        # Check final step
        final_step = history[-1]
        assert "thought" in final_step, "Final step should have thought"
        assert final_step.get("action") is None, "Final step should have no action"

    def assert_multi_step_workflow(
        self, history: List[Dict[str, Any]], min_steps: int = 2
    ) -> None:
        """
        Assert that a React agent executed a multi-step workflow.

        Args:
            history: Execution history
            min_steps: Minimum number of steps expected
        """
        assert (
            len(history) >= min_steps
        ), f"Expected at least {min_steps} steps, got {len(history)}"

        # Check that there are intermediate steps with actions
        action_steps = [step for step in history[:-1] if step.get("action") is not None]
        assert len(action_steps) > 0, "Should have at least one step with an action"

    async def run_react_agent_test(self, query: str, max_steps: int = 10):
        """
        Run a test query against the React Agent.

        Args:
            query: Query to send to the agent
            max_steps: Maximum number of steps to allow

        Returns:
            Agent execution history
        """
        from src.agents.react_agent.main import run_react_loop

        try:
            history = await run_react_loop(query, max_steps=max_steps)
            return history
        except Exception as e:
            # Return error in history format
            return [
                {
                    "step": 1,
                    "thought": f"Error occurred: {str(e)}",
                    "action": None,
                    "observation": f"Error: {str(e)}",
                }
            ]


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

    async def test_agent_environment_setup(self) -> None:
        """Test that agent environment is properly configured."""
        # Check for any of the supported API keys
        api_keys = ["OPENAI_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"]
        has_api_key = any(os.environ.get(key) for key in api_keys)
        assert has_api_key, f"No API key found. Expected one of: {api_keys}"

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
        api_keys = ["OPENAI_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"]
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
                "(OPENAI_API_KEY, GOOGLE_API_KEY, or GEMINI_API_KEY)"
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
