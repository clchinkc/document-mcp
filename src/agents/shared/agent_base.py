"""Base agent class with common functionality for Document MCP agents.

This module provides a base class that encapsulates common functionality
shared between Simple and ReAct agents, reducing code duplication and
providing consistent behavior across agent implementations.
"""

import abc
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.agent import AgentRunResult
from pydantic_ai.models import Model

from document_mcp.config import get_settings
from document_mcp.error_handler import ErrorContext
from document_mcp.exceptions import AgentConfigurationError

from .config import load_llm_config
from .config import prepare_mcp_server_environment
from .error_handling import create_agent_error_context


class AgentBase(abc.ABC):
    """Base class for Document MCP agents with common functionality."""

    def __init__(self, agent_type: str):
        """Initialize the base agent.

        Args:
            agent_type: Type identifier for this agent (e.g., "simple", "react")
        """
        self.agent_type = agent_type
        self.settings = get_settings()
        self._llm: Model | None = None
        self._agent: Agent | None = None

    async def get_llm(self) -> Model:
        """Get or initialize the LLM model.

        Returns:
            Configured LLM model instance

        Raises:
            AgentConfigurationError: If LLM configuration fails
        """
        if self._llm is None:
            try:
                self._llm = await load_llm_config()
            except Exception as e:
                raise AgentConfigurationError(
                    self.agent_type,
                    f"Failed to load LLM configuration: {str(e)}",
                    details={"original_error": str(e)},
                ) from e
        return self._llm

    def get_mcp_server_environment(self) -> dict[str, str]:
        """Get environment variables for MCP server subprocess.

        Returns:
            Environment dictionary for MCP server
        """
        return prepare_mcp_server_environment()

    def extract_mcp_tool_responses(self, agent_result: AgentRunResult) -> dict[str, Any]:
        """Extract MCP tool responses from agent execution result.

        This method provides a standardized way to extract structured data
        from MCP tool responses across different agent implementations.

        Args:
            agent_result: Result from agent execution

        Returns:
            Dictionary mapping tool names to their response data
        """
        tool_responses = {}

        if hasattr(agent_result, "all_messages"):
            messages = agent_result.all_messages()

            for message in messages:
                if hasattr(message, "parts"):
                    for part in message.parts:
                        # Check for tool returns (ToolReturnPart)
                        if (
                            hasattr(part, "tool_name")
                            and hasattr(part, "content")
                            and type(part).__name__ == "ToolReturnPart"
                        ):
                            tool_name = part.tool_name
                            tool_content = part.content

                            # Store the actual MCP tool response data
                            if isinstance(tool_content, list):
                                tool_responses[tool_name] = {"documents": tool_content}
                            elif isinstance(tool_content, dict):
                                tool_responses[tool_name] = tool_content
                            else:
                                tool_responses[tool_name] = {"content": tool_content}

        return tool_responses

    def create_error_context(self, operation_name: str) -> ErrorContext:
        """Create an error context for agent operations.

        Args:
            operation_name: Name of the operation being performed

        Returns:
            Configured error context for this agent
        """
        return create_agent_error_context(operation_name, self.agent_type)

    def validate_configuration(self) -> None:
        """Validate agent configuration.

        Raises:
            AgentConfigurationError: If configuration is invalid
        """
        if not self.settings.active_provider:
            raise AgentConfigurationError(
                self.agent_type,
                "No LLM provider configured. Please set OPENAI_API_KEY or GEMINI_API_KEY.",
                details={
                    "openai_configured": self.settings.openai_configured,
                    "gemini_configured": self.settings.gemini_configured,
                },
            )

    @abc.abstractmethod
    async def run(self, query: str, **kwargs) -> dict[str, Any]:
        """Execute the agent with the given query.

        Args:
            query: User query to process
            **kwargs: Additional parameters specific to agent implementation

        Returns:
            Agent response dictionary with standardized format

        Raises:
            AgentError: If agent execution fails
        """
        pass

    @abc.abstractmethod
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent.

        Returns:
            System prompt string
        """
        pass

    def create_response(
        self, summary: str, details: dict[str, Any], success: bool = True, **additional_fields
    ) -> dict[str, Any]:
        """Create a standardized agent response.

        Args:
            summary: Human-readable summary of the result
            details: Structured data from MCP tool responses
            success: Whether the operation was successful
            **additional_fields: Additional fields to include in response

        Returns:
            Standardized response dictionary
        """
        response = {
            "summary": summary,
            "details": details,
            "success": success,
            "agent_type": self.agent_type,
            **additional_fields,
        }

        return response

    async def __aenter__(self):
        """Async context manager entry."""
        await self.get_llm()  # Initialize LLM
        self.validate_configuration()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        # Cleanup if needed
        pass


class SingleTurnAgentMixin:
    """Mixin for agents that execute in a single turn."""

    def create_single_turn_response(self, agent_result: AgentRunResult, query: str) -> dict[str, Any]:
        """Create response for single-turn agent execution.

        Args:
            agent_result: Result from agent execution
            query: Original user query

        Returns:
            Standardized response dictionary
        """
        # Extract MCP tool responses for details
        tool_responses = self.extract_mcp_tool_responses(agent_result)

        # Get agent's response content
        summary = str(agent_result.data) if hasattr(agent_result, "data") else "Operation completed"

        return self.create_response(
            summary=summary, details=tool_responses, success=True, query=query, execution_mode="single_turn"
        )


class MultiTurnAgentMixin:
    """Mixin for agents that execute in multiple turns (like ReAct)."""

    def create_multi_turn_response(
        self, steps: list[dict[str, Any]], final_result: Any, query: str
    ) -> dict[str, Any]:
        """Create response for multi-turn agent execution.

        Args:
            steps: List of execution steps
            final_result: Final result from agent
            query: Original user query

        Returns:
            Standardized response dictionary
        """
        # Aggregate tool responses from all steps
        all_tool_responses = {}
        for step in steps:
            if "tool_responses" in step:
                all_tool_responses.update(step["tool_responses"])

        summary = str(final_result) if final_result else "Multi-step operation completed"

        return self.create_response(
            summary=summary,
            details=all_tool_responses,
            success=True,
            query=query,
            execution_mode="multi_turn",
            steps_count=len(steps),
            execution_steps=steps,
        )
