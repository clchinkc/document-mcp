"""Shared output formatter for consistent JSON responses across all agents.

This module provides a common interface for formatting agent responses
in JSON format while allowing each agent to maintain its unique characteristics.
"""

import json
from typing import Any


class AgentResponseFormatter:
    """Common JSON formatter for all agent types."""

    @staticmethod
    def format_as_json(
        agent_type: str,
        summary: str,
        details: Any | None = None,
        execution_log: str | None = None,
        error_message: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Format agent response as JSON with consistent structure.

        Args:
            agent_type: Type of agent (e.g., "simple", "react", "planner")
            summary: Human-readable summary of the agent's work
            details: Structured data containing the agent's actual results
            execution_log: Optional log of execution steps/thoughts
            error_message: Optional error message if the agent failed
            metadata: Optional additional metadata about the execution

        Returns:
            JSON string with consistent structure across all agents
        """
        response = {
            "agent_type": agent_type,
            "summary": summary,
            "details": details,
            "error_message": error_message,
        }

        # Add optional fields only if they have values
        if execution_log:
            response["execution_log"] = execution_log
        if metadata:
            response["metadata"] = metadata

        return json.dumps(response, indent=2, ensure_ascii=False)

    @staticmethod
    def format_simple_agent_response(summary: str, details: Any, error_message: str | None = None) -> str:
        """Format response for simple agent with minimal metadata.

        Args:
            summary: Human-readable summary
            details: Structured agent results
            error_message: Optional error message

        Returns:
            JSON string formatted for simple agent
        """
        return AgentResponseFormatter.format_as_json(
            agent_type="simple",
            summary=summary,
            details=details,
            error_message=error_message,
        )

    @staticmethod
    def format_react_agent_response(
        summary: str,
        mcp_tool_responses: dict[str, Any],
        steps_executed: list[dict[str, Any]],
        execution_log: str,
        max_steps: int,
        error_message: str | None = None,
    ) -> str:
        """Format response for ReAct agent with execution history.

        Args:
            summary: Human-readable summary of execution
            mcp_tool_responses: MCP tool responses for the details field (structured data)
            steps_executed: List of execution steps with thoughts/actions/observations
            execution_log: Formatted log of agent's reasoning
            max_steps: Maximum steps configured for the agent
            error_message: Optional error message

        Returns:
            JSON string formatted for ReAct agent following architectural principles:
            - summary: LLM-generated human-readable description
            - details: Structured data from MCP tool responses
        """
        # Use MCP tool responses for details field (architectural requirement)
        # Execution steps are available in metadata for debugging/analysis
        return AgentResponseFormatter.format_as_json(
            agent_type="react",
            summary=summary,
            details=mcp_tool_responses,
            execution_log=execution_log,
            error_message=error_message,
            metadata={
                "steps_executed": len(steps_executed),
                "max_steps": max_steps,
                "execution_steps": steps_executed,  # ReAct steps moved to metadata
            },
        )

    @staticmethod
    def format_planner_agent_response(
        summary: str,
        plan_generated: bool,
        execution_completed: bool,
        steps_executed: list[dict[str, Any]],
        error_message: str | None = None,
    ) -> str:
        """Format response for planner agent with plan execution details.

        Args:
            summary: Human-readable summary of planning and execution
            plan_generated: Whether plan generation succeeded
            execution_completed: Whether execution completed successfully
            steps_executed: List of executed plan steps with results
            error_message: Optional error message

        Returns:
            JSON string formatted for planner agent
        """
        return AgentResponseFormatter.format_as_json(
            agent_type="planner",
            summary=summary,
            details={
                "plan_generated": plan_generated,
                "execution_completed": execution_completed,
                "steps_executed": steps_executed,
            },
            error_message=error_message,
            metadata={
                "steps_count": len(steps_executed),
                "plan_generated": plan_generated,
                "execution_completed": execution_completed,
            },
        )
