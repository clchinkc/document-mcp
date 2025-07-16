"""Real performance metrics infrastructure for document-mcp agents.

This module provides comprehensive performance tracking for all agent types,
replacing hardcoded mock values with actual LLM usage data and execution metrics.
Works cleanly with the test-layer LLM evaluation system when enabled.
"""

import time
from dataclasses import dataclass
from dataclasses import field
from typing import Any

from pydantic_ai.agent import AgentRunResult
from pydantic_ai.result import Usage


@dataclass
class AgentPerformanceMetrics:
    """Comprehensive performance metrics for agent execution.

    This class captures real performance data from agent execution,
    replacing hardcoded mock values with actual measurements.
    Designed to work cleanly with optional test-layer LLM evaluation.
    """

    # Core metrics
    token_usage: int = 0
    request_tokens: int = 0
    response_tokens: int = 0
    tool_calls_count: int = 0
    execution_time: float = 0.0
    api_requests: int = 0

    # Success tracking
    success: bool = False
    error_message: str | None = None

    # Detailed metrics
    tool_names: list[str] = field(default_factory=list)
    message_count: int = 0
    cached_tokens: int = 0
    thoughts_tokens: int = 0

    # Agent-specific data
    agent_type: str = "unknown"
    response_data: dict[str, Any] | None = None

    # File system tracking
    file_system_changes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary for reporting and analysis."""
        return {
            "token_usage": self.token_usage,
            "request_tokens": self.request_tokens,
            "response_tokens": self.response_tokens,
            "tool_calls_count": self.tool_calls_count,
            "execution_time": self.execution_time,
            "api_requests": self.api_requests,
            "success": self.success,
            "error_message": self.error_message,
            "tool_names": self.tool_names,
            "message_count": self.message_count,
            "cached_tokens": self.cached_tokens,
            "thoughts_tokens": self.thoughts_tokens,
            "agent_type": self.agent_type,
            "file_system_changes": self.file_system_changes,
        }

    @property
    def tokens_per_second(self) -> float:
        """Calculate tokens processed per second."""
        if self.execution_time > 0:
            return self.token_usage / self.execution_time
        return 0.0

    @property
    def average_tokens_per_tool_call(self) -> float:
        """Calculate average tokens per tool call."""
        if self.tool_calls_count > 0:
            return self.token_usage / self.tool_calls_count
        return 0.0

    @property
    def efficiency_score(self) -> float:
        """Calculate efficiency score based on token usage and execution time.
        Higher scores indicate better efficiency.
        """
        if self.execution_time > 0 and self.token_usage > 0:
            # Normalize by expected baseline (lower is better for both metrics)
            time_factor = 1.0 / max(self.execution_time, 0.1)
            token_factor = 1000.0 / max(self.token_usage, 50)
            return (time_factor + token_factor) / 2.0
        return 0.0


class PerformanceMetricsCollector:
    """Utility class for collecting real performance metrics from agent execution.

    This class extracts actual performance data from Pydantic AI agent responses,
    replacing the hardcoded mock values in the evaluation framework.
    Clean architecture: agents collect performance only, tests optionally enhance.
    """

    @staticmethod
    def collect_from_agent_result(
        agent_result: AgentRunResult,
        agent_type: str,
        execution_start_time: float,
        response_data: dict[str, Any] | None = None,
    ) -> AgentPerformanceMetrics:
        """Collect real performance metrics from an AgentRunResult.

        Args:
            agent_result: The result from agent execution
            agent_type: Type of agent (simple, react, planner)
            execution_start_time: When agent execution started (from time.time())
            response_data: Optional structured response data

        Returns:
            AgentPerformanceMetrics with real performance data
        """
        metrics = AgentPerformanceMetrics(agent_type=agent_type)

        # Calculate real execution time
        metrics.execution_time = time.time() - execution_start_time

        # Extract real token usage from agent result
        try:
            if hasattr(agent_result, "usage"):
                # Try calling usage() method first
                if callable(agent_result.usage):
                    usage: Usage = agent_result.usage()
                else:
                    usage: Usage = agent_result.usage

                if usage:
                    metrics.token_usage = usage.total_tokens
                    metrics.request_tokens = usage.request_tokens
                    metrics.response_tokens = usage.response_tokens
                    metrics.api_requests = usage.requests

                    # Extract additional details if available
                    if hasattr(usage, "details") and usage.details:
                        details = usage.details
                        metrics.cached_tokens = details.get("cached_content_tokens", 0)
                        metrics.thoughts_tokens = details.get("thoughts_tokens", 0)
        except Exception as e:
            # If we can't get usage data, set defaults and continue
            print(f"Warning: Could not extract usage data: {e}")
            metrics.token_usage = 0

        # Extract tool call information from message history
        if hasattr(agent_result, "all_messages"):
            messages = agent_result.all_messages()
            metrics.message_count = len(messages)

            # Count tool calls and extract tool names
            tool_calls = []
            for message in messages:
                if hasattr(message, "parts"):
                    for part in message.parts:
                        if hasattr(part, "tool_name"):
                            tool_calls.append(part.tool_name)

            metrics.tool_calls_count = len(tool_calls)
            metrics.tool_names = list(set(tool_calls))  # Unique tool names

        # Determine success status
        metrics.success = agent_result.output is not None and not getattr(
            agent_result, "error_message", None
        )

        # Extract error message if present
        if hasattr(agent_result, "error_message") and agent_result.error_message:
            metrics.error_message = agent_result.error_message

        # Store response data
        if response_data:
            metrics.response_data = response_data
        elif hasattr(agent_result, "output") and agent_result.output:
            # Convert Pydantic model to dict if needed
            if hasattr(agent_result.output, "model_dump"):
                metrics.response_data = agent_result.output.model_dump()
            else:
                metrics.response_data = agent_result.output

        return metrics

    @staticmethod
    def collect_from_timing_and_response(
        execution_start_time: float,
        agent_type: str,
        response_data: dict[str, Any],
        success: bool = True,
        error_message: str | None = None,
        token_usage: int | None = None,
        tool_calls: list[str] | None = None,
    ) -> AgentPerformanceMetrics:
        """Collect metrics from timing and response data when full AgentRunResult is not available.

        This method is useful for cases where we need to construct metrics from
        partial data or when integrating with different agent execution patterns.

        Args:
            execution_start_time: When execution started
            agent_type: Type of agent
            response_data: Structured response data
            success: Whether the operation succeeded
            error_message: Optional error message
            token_usage: Optional token usage count
            tool_calls: Optional list of tool names called

        Returns:
            AgentPerformanceMetrics with available data
        """
        metrics = AgentPerformanceMetrics(agent_type=agent_type)

        # Calculate execution time
        metrics.execution_time = time.time() - execution_start_time

        # Set provided data
        metrics.response_data = response_data
        metrics.success = success
        metrics.error_message = error_message

        # Set token usage if provided
        if token_usage is not None:
            metrics.token_usage = token_usage

        # Set tool call information if provided
        if tool_calls:
            metrics.tool_calls_count = len(tool_calls)
            metrics.tool_names = list(set(tool_calls))

        return metrics

    @staticmethod
    def validate_performance_thresholds(
        metrics: AgentPerformanceMetrics,
        token_range: tuple[int, int],
        max_execution_time: float,
    ) -> tuple[bool, list[str]]:
        """Validate performance metrics against expected thresholds.

        Args:
            metrics: Performance metrics to validate
            token_range: Expected token usage range (min, max)
            max_execution_time: Maximum acceptable execution time

        Returns:
            Tuple of (is_valid, list_of_violations)
        """
        violations = []

        # Check token usage range
        if metrics.token_usage < token_range[0]:
            violations.append(
                f"Token usage {metrics.token_usage} below minimum {token_range[0]}"
            )
        elif metrics.token_usage > token_range[1]:
            violations.append(
                f"Token usage {metrics.token_usage} above maximum {token_range[1]}"
            )

        # Check execution time
        if metrics.execution_time > max_execution_time:
            violations.append(
                f"Execution time {metrics.execution_time:.2f}s exceeds maximum {max_execution_time}s"
            )

        # Check basic success criteria
        if not metrics.success:
            violations.append(f"Operation failed: {metrics.error_message}")

        return len(violations) == 0, violations


class TokenUsageAggregator:
    """Utility class for aggregating token usage from multiple agent results."""

    @staticmethod
    def aggregate_from_results(agent_results: list[Any]) -> dict[str, int]:
        """Aggregate token usage from multiple agent results.

        Args:
            agent_results: List of AgentRunResult objects

        Returns:
            Dictionary with aggregated token metrics
        """
        totals = {
            "total_tokens": 0,
            "request_tokens": 0,
            "response_tokens": 0,
            "api_requests": 0,
        }

        for agent_result in agent_results:
            usage = TokenUsageAggregator._extract_usage(agent_result)
            if usage:
                totals["total_tokens"] += usage.total_tokens
                totals["request_tokens"] += usage.request_tokens
                totals["response_tokens"] += usage.response_tokens
                totals["api_requests"] += usage.requests

        return totals

    @staticmethod
    def _extract_usage(agent_result: Any) -> Usage | None:
        """Extract usage from an agent result with error handling."""
        try:
            if hasattr(agent_result, "usage") and agent_result.usage:
                usage = (
                    agent_result.usage()
                    if callable(agent_result.usage)
                    else agent_result.usage
                )
                return usage if usage else None
        except Exception:
            # Continue aggregating even if one result fails
            pass
        return None


class MetricsCollectionContext:
    """Context manager for simplified metrics collection during agent execution."""

    def __init__(self, agent_type: str):
        self.agent_type = agent_type
        self.start_time = None
        self.agent_results = []
        self.tool_calls = []

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def add_agent_result(self, agent_result: Any):
        """Add an agent result for token tracking."""
        self.agent_results.append(agent_result)

    def add_tool_call(self, tool_name: str):
        """Add a tool call for tracking."""
        self.tool_calls.append(tool_name)

    def extract_tool_calls_from_history(self, history: list[dict]) -> list[str]:
        """Extract tool names from execution history."""
        extracted_tools = []
        for step in history:
            if step.get("action"):
                action_str = str(step["action"])
                if "(" in action_str:
                    tool_name = action_str.split("(")[0].strip()
                    extracted_tools.append(tool_name)
        return extracted_tools

    def create_metrics(self, response_data: dict[str, Any]) -> AgentPerformanceMetrics:
        """Create performance metrics from collected data."""
        if self.agent_results:
            # Use comprehensive metrics collection with token aggregation
            metrics = PerformanceMetricsCollector.collect_from_agent_result(
                agent_result=self.agent_results[0],
                agent_type=self.agent_type,
                execution_start_time=self.start_time,
                response_data=response_data,
            )

            # Override with aggregated token usage from all results
            if len(self.agent_results) > 1:
                aggregated = TokenUsageAggregator.aggregate_from_results(
                    self.agent_results
                )
                metrics.token_usage = aggregated["total_tokens"]
                metrics.request_tokens = aggregated["request_tokens"]
                metrics.response_tokens = aggregated["response_tokens"]
                metrics.api_requests = aggregated["api_requests"]

            # Update tool call information
            all_tools = self.tool_calls.copy()
            metrics.tool_calls_count = len(all_tools)
            metrics.tool_names = list(set(all_tools))

            return metrics
        else:
            # Fallback to timing-based metrics
            return PerformanceMetricsCollector.collect_from_timing_and_response(
                execution_start_time=self.start_time,
                agent_type=self.agent_type,
                response_data=response_data,
                success=response_data.get("success", True),
                error_message=response_data.get("error"),
                tool_calls=self.tool_calls,
            )


def build_response_data(agent_type: str, **kwargs) -> dict[str, Any]:
    """Build standardized response data dictionary."""
    base_data = {"agent_type": agent_type, "success": kwargs.get("success", True)}
    base_data.update(kwargs)
    return base_data
