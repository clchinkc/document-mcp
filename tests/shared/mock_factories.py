"""
Mock factories for Document MCP testing.

This module provides consistent mock object creation patterns
used across all test types to reduce duplication and ensure
consistent testing behavior.
"""

import os
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, Mock

from src.agents.simple_agent import FinalAgentResponse


def create_mock_agent(response_data: Optional[Dict[str, Any]] = None) -> Mock:
    """
    Create a mock agent for testing, avoiding AsyncMock for context managers.

    Args:
        response_data: Optional data to include in mock response

    Returns:
        Mock agent with configured run method
    """
    mock_agent = Mock()

    # Default response if none provided
    if response_data is None:
        response_data = {
            "summary": "Mock agent response",
            "details": None,
            "error_message": None,
        }

    # Create mock run result
    mock_run_result = Mock()
    mock_run_result.output = FinalAgentResponse(**response_data)
    mock_run_result.error_message = None

    # Use simple async function instead of AsyncMock to avoid warnings
    async def mock_run(*args, **kwargs):
        return mock_run_result
    
    mock_agent.run = mock_run
    
    # Use a simple class with async methods to mock the context manager
    class MockContextManager:
        async def __aenter__(self):
            return mock_agent
        
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return False
            
    mock_agent.run_mcp_servers = Mock(return_value=MockContextManager())

    return mock_agent


def create_mock_mcp_server(host: str = "localhost", port: int = 3001) -> Mock:
    """
    Create a mock MCP server for testing.

    Args:
        host: Server host
        port: Server port

    Returns:
        Mock MCP server
    """
    mock_server = Mock()
    mock_server.host = host
    mock_server.port = port
    mock_server.server_url = f"http://{host}:{port}"

    # Context manager support
    async def mock_aenter():
        return mock_server
    
    async def mock_aexit(*args):
        return False
    
    mock_server.__aenter__ = mock_aenter
    mock_server.__aexit__ = mock_aexit

    return mock_server


def create_mock_llm_config(
    model_type: str = "openai", model_name: str = "gpt-4o"
) -> Mock:
    """
    Create a mock LLM configuration.

    Args:
        model_type: Type of model (openai, gemini)
        model_name: Name of the model

    Returns:
        Mock LLM model
    """
    mock_model = Mock()
    mock_model.model_type = model_type
    mock_model.model_name = model_name

    # Add common model attributes
    mock_model.name = model_name
    mock_model.provider = model_type

    return mock_model


def create_mock_environment(
    api_key_type: str = "openai",
    include_server_config: bool = True,
    custom_vars: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """
    Create a mock environment configuration for testing.

    Args:
        api_key_type: Type of API key to mock (openai, gemini)
        include_server_config: Whether to include server configuration
        custom_vars: Additional custom environment variables

    Returns:
        Dictionary of environment variables for patching
    """
    env_vars = {}

    # API key configuration
    if api_key_type == "openai":
        env_vars.update(
            {
                "OPENAI_API_KEY": "test_openai_key",
                "OPENAI_MODEL_NAME": "gpt-4o",
            }
        )
    elif api_key_type == "gemini":
        env_vars.update(
            {
                "GEMINI_API_KEY": "test_gemini_key",
                "GEMINI_MODEL_NAME": "gemini-2.5-flash",
            }
        )
    elif api_key_type == "both":
        env_vars.update(
            {
                "OPENAI_API_KEY": "test_openai_key",
                "OPENAI_MODEL_NAME": "gpt-4o",
                "GEMINI_API_KEY": "test_gemini_key",
                "GEMINI_MODEL_NAME": "gemini-2.5-flash",
            }
        )

    # Server configuration
    if include_server_config:
        env_vars.update(
            {
                "MCP_SERVER_HOST": "localhost",
                "MCP_SERVER_PORT": "3001",
            }
        )

    # Custom variables
    if custom_vars:
        env_vars.update(custom_vars)

    return env_vars


def create_mock_document_info(
    doc_name: str = "test_document",
    chapter_count: int = 3,
    word_count: int = 100,
    paragraph_count: int = 10,
) -> Dict[str, Any]:
    """
    Create mock document info data.

    Args:
        doc_name: Document name
        chapter_count: Number of chapters
        word_count: Total word count
        paragraph_count: Total paragraph count

    Returns:
        Dictionary representing document info
    """
    from datetime import datetime, timezone

    return {
        "document_name": doc_name,
        "total_chapters": chapter_count,
        "total_word_count": word_count,
        "total_paragraph_count": paragraph_count,
        "last_modified": datetime.now(timezone.utc),
        "chapters": [
            {
                "chapter_name": f"{i:02d}-chapter.md",
                "word_count": word_count // chapter_count,
                "paragraph_count": paragraph_count // chapter_count,
                "last_modified": datetime.now(timezone.utc),
            }
            for i in range(1, chapter_count + 1)
        ],
    }


def create_mock_operation_status(
    success: bool = True, message: str = "Operation completed"
) -> Dict[str, Any]:
    """
    Create mock operation status data.

    Args:
        success: Whether operation was successful
        message: Status message

    Returns:
        Dictionary representing operation status
    """
    return {
        "success": success,
        "message": message,
        "details": None,
    }


def create_mock_chapter_content(
    doc_name: str = "test_document",
    chapter_name: str = "01-test.md",
    content: str = "# Test Chapter\n\nThis is test content.",
) -> Dict[str, Any]:
    """
    Create mock chapter content data.

    Args:
        doc_name: Document name
        chapter_name: Chapter name
        content: Chapter content

    Returns:
        Dictionary representing chapter content
    """
    from datetime import datetime, timezone

    word_count = len(content.split())
    paragraph_count = len([p for p in content.split("\n\n") if p.strip()])

    return {
        "document_name": doc_name,
        "chapter_name": chapter_name,
        "content": content,
        "word_count": word_count,
        "paragraph_count": paragraph_count,
        "last_modified": datetime.now(timezone.utc),
    }


def create_mock_statistics_report(
    scope: str = "document: test_document",
    word_count: int = 100,
    paragraph_count: int = 10,
    chapter_count: Optional[int] = 3,
) -> Dict[str, Any]:
    """
    Create mock statistics report data.

    Args:
        scope: Scope of the statistics
        word_count: Word count
        paragraph_count: Paragraph count
        chapter_count: Chapter count (optional)

    Returns:
        Dictionary representing statistics report
    """
    stats = {
        "scope": scope,
        "word_count": word_count,
        "paragraph_count": paragraph_count,
    }

    if chapter_count is not None:
        stats["chapter_count"] = chapter_count

    return stats


def create_mock_error_response(error_message: str = "Test error") -> Dict[str, Any]:
    """
    Create mock error response data.

    Args:
        error_message: Error message

    Returns:
        Dictionary representing error response
    """
    return {
        "summary": f"Error: {error_message}",
        "details": None,
        "error_message": error_message,
    }


def create_mock_react_step(
    thought: str = "Test thought",
    action: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create mock ReAct step data.

    Args:
        thought: Thought content
        action: Action to take (None for final step)

    Returns:
        Dictionary representing ReAct step
    """
    return {
        "thought": thought,
        "action": action,
    }


def create_mock_react_history(steps: int = 3) -> list:
    """
    Create mock ReAct execution history.

    Args:
        steps: Number of steps in history

    Returns:
        List of mock history steps
    """
    history = []

    for i in range(1, steps + 1):
        if i < steps:
            # Intermediate step with action
            history.append(
                {
                    "step": i,
                    "thought": f"Step {i} reasoning",
                    "action": f"mock_action_{i}()",
                    "observation": f"Step {i} completed successfully",
                }
            )
        else:
            # Final step without action
            history.append(
                {
                    "step": i,
                    "thought": "Task completed successfully",
                    "action": None,
                    "observation": "All steps completed",
                }
            )

    return history
