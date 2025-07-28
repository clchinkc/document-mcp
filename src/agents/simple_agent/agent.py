"""Simple Agent implementation for single-turn document operations.

This module provides a stateless agent that executes document operations
in a single turn using structured output and optimized tool descriptions.
"""

from typing import Any

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio

from src.agents.shared.agent_base import AgentBase
from src.agents.shared.agent_base import SingleTurnAgentMixin
from src.agents.shared.tool_descriptions import get_tool_descriptions_for_agent

from .prompts import get_simple_agent_system_prompt


class SimpleAgentResponse(BaseModel):
    """Response model for the Simple Agent."""

    summary: str
    details: dict[str, Any]


class SimpleAgent(AgentBase, SingleTurnAgentMixin):
    """Simple Agent implementation with single-turn execution."""

    def __init__(self):
        """Initialize the Simple Agent."""
        super().__init__("simple")
        self._pydantic_agent: Agent | None = None

    async def get_pydantic_agent(self) -> Agent:
        """Get or create the Pydantic AI agent instance.

        Returns:
            Configured Pydantic AI Agent
        """
        if self._pydantic_agent is None:
            llm = await self.get_llm()

            # Set up MCP server
            server_env = self.get_mcp_server_environment()
            from src.agents.shared.config import MCP_SERVER_CMD

            mcp_server = MCPServerStdio(
                command=MCP_SERVER_CMD[0],
                args=MCP_SERVER_CMD[1:],
                env=server_env,
            )

            # Get optimized tool descriptions for simple agent
            tool_descriptions = get_tool_descriptions_for_agent("simple")

            # Create the agent with system prompt and tool descriptions
            system_prompt = self.get_system_prompt()
            full_prompt = f"{system_prompt}\n\n{tool_descriptions}"

            self._pydantic_agent = Agent(
                llm,
                result_type=SimpleAgentResponse,
                system_prompt=full_prompt,
                mcp_servers=[mcp_server],
            )

        return self._pydantic_agent

    def get_system_prompt(self) -> str:
        """Get the system prompt for the Simple Agent.

        Returns:
            System prompt string
        """
        return get_simple_agent_system_prompt()

    async def run(self, query: str, **kwargs) -> dict[str, Any]:
        """Execute the Simple Agent with the given query.

        Args:
            query: User query to process
            **kwargs: Additional parameters (timeout, etc.)

        Returns:
            Agent response dictionary with standardized format
        """
        timeout = kwargs.get("timeout", self.settings.default_timeout)

        with self.create_error_context("simple_agent_execution"):
            agent = await self.get_pydantic_agent()

            # Execute the agent with timeout
            try:
                if timeout:
                    import asyncio

                    agent_result = await asyncio.wait_for(agent.run(query), timeout=timeout)
                else:
                    agent_result = await agent.run(query)

                # Create standardized response
                return self.create_single_turn_response(agent_result, query)

            except asyncio.TimeoutError:
                from document_mcp.exceptions import OperationError

                raise OperationError(
                    operation="simple_agent_execution",
                    reason=f"Agent execution timed out after {timeout}s",
                    details={"query": query, "timeout": timeout},
                )

    async def run_with_structured_output(self, query: str, **kwargs) -> SimpleAgentResponse:
        """Execute the agent and return structured output.

        Args:
            query: User query to process
            **kwargs: Additional parameters

        Returns:
            Structured agent response
        """
        agent = await self.get_pydantic_agent()
        result = await agent.run(query)
        return (
            result.data
            if hasattr(result, "data")
            else SimpleAgentResponse(
                summary="Operation completed", details=self.extract_mcp_tool_responses(result)
            )
        )

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup resources."""
        if self._pydantic_agent and hasattr(self._pydantic_agent, "__aexit__"):
            await self._pydantic_agent.__aexit__(exc_type, exc_val, exc_tb)
        await super().__aexit__(exc_type, exc_val, exc_tb)
