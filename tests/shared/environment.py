"""
Shared environment testing utilities for Document MCP.

This module provides common environment checking and validation
functions used across different test types.
"""

import os
from typing import List, Optional
from pathlib import Path

# The root directory for all test-generated documents.
# This ensures tests run in an isolated directory.
TEST_DOCUMENT_ROOT = Path(".documents_storage/E2E_TEST_DOCUMENT_ROOT").resolve()

def check_api_keys_available(required_keys: Optional[List[str]] = None) -> bool:
    """
    Check if any of the supported API keys are available.

    Args:
        required_keys: Optional list of specific keys to check

    Returns:
        True if at least one API key is available
    """
    if required_keys is None:
        required_keys = ["OPENAI_API_KEY", "GEMINI_API_KEY"]

    return any(os.environ.get(key) for key in required_keys)


def validate_agent_environment() -> None:
    """
    Validate that the environment is properly set up for agent testing.

    Raises:
        AssertionError: If environment is not properly configured
    """
    api_keys = ["OPENAI_API_KEY", "GEMINI_API_KEY"]
    has_api_key = check_api_keys_available(api_keys)
    assert has_api_key, f"No API key found. Expected one of: {api_keys}"


def validate_package_imports() -> None:
    """
    Validate that all required packages can be imported.

    Raises:
        AssertionError: If required packages are missing
    """
    try:
        import pydantic_ai

        # Verify pydantic_ai has expected functionality
        assert hasattr(pydantic_ai, "Agent"), "pydantic_ai should provide Agent class"
        assert hasattr(
            pydantic_ai, "RunContext"
        ), "pydantic_ai should provide RunContext class"
    except ImportError:
        raise AssertionError(
            "Failed to import pydantic_ai - required for agent functionality"
        )


def validate_simple_agent_imports() -> None:
    """
    Validate that Simple Agent components can be imported.

    Raises:
        AssertionError: If Simple Agent imports fail
    """
    try:
        from src.agents.simple_agent import FinalAgentResponse, StatisticsReport

        # Verify imported classes are proper types
        assert isinstance(
            FinalAgentResponse, type
        ), "FinalAgentResponse should be a class type"
        assert isinstance(
            StatisticsReport, type
        ), "StatisticsReport should be a class type"

        # Verify they are pydantic models
        assert hasattr(
            FinalAgentResponse, "model_fields"
        ), "FinalAgentResponse should be a pydantic model"
        assert hasattr(
            StatisticsReport, "model_fields"
        ), "StatisticsReport should be a pydantic model"
    except ImportError as e:
        raise AssertionError(f"Failed to import Simple Agent models: {e}")


def validate_react_agent_imports() -> None:
    """
    Validate that React Agent components can be imported.

    Raises:
        AssertionError: If React Agent imports fail
    """
    try:
        from src.agents.react_agent.main import ReActStep, run_react_loop

        # Verify imported classes are proper types
        assert isinstance(ReActStep, type), "ReActStep should be a class type"
        assert callable(run_react_loop), "run_react_loop should be a callable function"

        # Verify ReActStep is a pydantic model
        assert hasattr(
            ReActStep, "model_fields"
        ), "ReActStep should be a pydantic model"
    except ImportError as e:
        raise AssertionError(f"Failed to import React Agent components: {e}")


def has_real_api_key() -> bool:
    """
    Check if a real (non-test) API key is available.

    Returns:
        True if a real API key is detected
    """
    api_keys = ["OPENAI_API_KEY", "GEMINI_API_KEY"]
    for key in api_keys:
        value = os.environ.get(key, "").strip()
        if value and value != "test_key" and not value.startswith("sk-test"):
            return True
    return False
