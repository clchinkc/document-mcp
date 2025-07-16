"""Shared Error Handling Utilities for Agents

This module provides common classes for error classification and retry logic
that can be used across different agent implementations.
"""

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Any

# --- Error Classification and Retry Configuration ---


class ErrorType(Enum):
    """Enumeration of possible error types for classification."""

    NETWORK_ERROR = "network"
    AUTHENTICATION_ERROR = "auth"
    RATE_LIMIT_ERROR = "rate_limit"
    VALIDATION_ERROR = "validation"
    TOOL_ERROR = "tool"
    LLM_ERROR = "llm"
    CONFIGURATION_ERROR = "config"
    RESOURCE_ERROR = "resource"
    UNKNOWN_ERROR = "unknown"


@dataclass
class RetryConfig:
    """Data class to hold retry configuration."""

    max_retries: int
    initial_delay: float = 1.0
    max_delay: float = 30.0


@dataclass
class ErrorInfo:
    """Data class to hold detailed information about a classified error."""

    error_type: ErrorType
    is_retryable: bool
    max_retries: int
    initial_delay: float
    max_delay: float
    severity: str  # "low", "medium", "high", "critical"
    recovery_action: str
    user_message: str


class ErrorClassifier:
    """Classifies exceptions and provides structured error information."""

    def classify(self, error: Exception) -> ErrorInfo:
        """Classify an error and return appropriate retry and recovery information.

        Args:
            error: The exception to classify.

        Returns:
            An ErrorInfo object with details about the error.
        """
        error_str = str(error).lower()

        # TimeoutError and network/connection errors (502, 503, 504)
        if isinstance(error, asyncio.TimeoutError) or any(
            keyword in error_str
            for keyword in [
                "connection",
                "timeout",
                "network",
                "unreachable",
                "502",
                "503",
                "504",
            ]
        ):
            return ErrorInfo(
                error_type=ErrorType.NETWORK_ERROR,
                is_retryable=True,
                max_retries=3,
                initial_delay=1.0,
                max_delay=16.0,
                severity="medium",
                recovery_action="exponential_backoff",
                user_message="Network connectivity issue detected. Retrying...",
            )

        # Authentication errors (401)
        if any(
            keyword in error_str
            for keyword in ["api key", "authentication", "unauthorized", "401"]
        ):
            return ErrorInfo(
                error_type=ErrorType.AUTHENTICATION_ERROR,
                is_retryable=False,
                max_retries=0,
                initial_delay=0.0,
                max_delay=0.0,
                severity="high",
                recovery_action="check_config",
                user_message="Authentication error. Please check your API key configuration.",
            )

        # Rate limiting errors (429)
        if any(
            keyword in error_str
            for keyword in ["rate limit", "quota", "too many requests", "429"]
        ):
            return ErrorInfo(
                error_type=ErrorType.RATE_LIMIT_ERROR,
                is_retryable=True,
                max_retries=5,
                initial_delay=2.0,
                max_delay=60.0,
                severity="medium",
                recovery_action="exponential_backoff",
                user_message="API rate limit reached. Waiting before retry...",
            )

        # Validation errors
        if any(
            keyword in error_str
            for keyword in ["invalid", "validation", "format", "parse"]
        ):
            return ErrorInfo(
                error_type=ErrorType.VALIDATION_ERROR,
                is_retryable=False,
                max_retries=0,
                initial_delay=0.0,
                max_delay=0.0,
                severity="low",
                recovery_action="user_feedback",
                user_message="Input validation error. Please check the format of your request.",
            )

        # Tool execution errors
        if any(keyword in error_str for keyword in ["tool", "mcp", "execution"]):
            return ErrorInfo(
                error_type=ErrorType.TOOL_ERROR,
                is_retryable=True,
                max_retries=2,
                initial_delay=0.5,
                max_delay=4.0,
                severity="medium",
                recovery_action="retry_simplified",
                user_message="Tool execution error. Attempting recovery...",
            )

        # LLM generation errors
        if any(
            keyword in error_str
            for keyword in ["llm", "model", "generation", "completion"]
        ):
            return ErrorInfo(
                error_type=ErrorType.LLM_ERROR,
                is_retryable=True,
                max_retries=2,
                initial_delay=1.0,
                max_delay=8.0,
                severity="high",
                recovery_action="retry_with_fallback",
                user_message="AI service error. Retrying with fallback strategy...",
            )

        # Default: Unknown error
        return ErrorInfo(
            error_type=ErrorType.UNKNOWN_ERROR,
            is_retryable=True,
            max_retries=1,
            initial_delay=1.0,
            max_delay=4.0,
            severity="medium",
            recovery_action="basic_retry",
            user_message="Unexpected error encountered. Attempting recovery...",
        )


class RetryManager:
    """Manages retry logic with exponential backoff and jitter for function execution."""

    def __init__(self):
        self.error_classifier = ErrorClassifier()

    async def execute_with_retry(self, func, *args, **kwargs):
        """Execute an awaitable function with intelligent, classified retry logic.

        Args:
            func: The awaitable function to execute.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.

        Returns:
            The result of the function if successful.

        Raises:
            The last exception if all retry attempts fail.
        """
        last_error = None
        max_attempts = 5  # A sensible default maximum

        for attempt in range(1, max_attempts + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_error = e
                error_info = self.error_classifier.classify(e)

                if not error_info.is_retryable or attempt > error_info.max_retries:
                    print(
                        f"Final failure after {attempt} attempts: {error_info.user_message}"
                    )
                    raise e

                delay = self._calculate_delay(
                    attempt, error_info.initial_delay, error_info.max_delay
                )
                print(f"Attempt {attempt} failed: {error_info.user_message}")
                print(f"Retrying in {delay:.1f}s...")
                await asyncio.sleep(delay)

        # This line should theoretically not be reached, but is a safeguard.
        raise last_error

    def _calculate_delay(
        self, attempt: int, initial_delay: float, max_delay: float
    ) -> float:
        """Calculate exponential backoff delay with jitter.

        Args:
            attempt: Current attempt number (1-based)
            initial_delay: Initial delay in seconds
            max_delay: Maximum delay in seconds

        Returns:
            Calculated delay in seconds
        """
        import random

        # Exponential backoff: delay = initial_delay * (2 ^ (attempt - 1))
        exponential_delay = initial_delay * (2 ** (attempt - 1))

        # Cap at max_delay
        delay = min(exponential_delay, max_delay)

        # Add jitter (random variation of ±25%)
        jitter = delay * 0.25 * (random.random() * 2 - 1)
        final_delay = max(0.1, delay + jitter)  # Minimum 0.1s delay

        return final_delay


# --- Shared Error Response Utilities ---


def create_error_response(
    summary: str, error_type: str, details: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Create a standardized error response for agents.

    Args:
        summary: Human-readable error summary
        error_type: Type of error (e.g., "timeout", "validation", "network")
        details: Optional additional error details

    Returns:
        Standardized error response dictionary
    """
    return {
        "summary": summary,
        "details": details,
        "error_message": error_type,
    }
