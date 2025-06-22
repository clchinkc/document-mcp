"""
Shared testing utilities for Document MCP.

This package contains common testing infrastructure used across
unit, integration, and e2e tests to reduce duplication and ensure
consistent testing patterns.
"""

from .assertions import (
    assert_agent_response_valid,
    assert_chapter_metadata_valid,
    assert_document_info_valid,
    assert_operation_failure,
    assert_operation_success,
)
from .environment import (
    check_api_keys_available,
    has_real_api_key,
    validate_agent_environment,
    validate_package_imports,
    validate_react_agent_imports,
    validate_simple_agent_imports,
)
from .mock_factories import (
    create_mock_agent,
    create_mock_environment,
    create_mock_llm_config,
    create_mock_mcp_server,
)
from .server_manager import UnifiedMCPServerManager
from .test_data import (
    SAMPLE_CHAPTER_CONTENT,
    SAMPLE_SEARCH_TEXT,
    create_test_document,
    generate_unique_name,
)

__all__ = [
    # Assertions
    "assert_agent_response_valid",
    "assert_operation_success",
    "assert_operation_failure",
    "assert_document_info_valid",
    "assert_chapter_metadata_valid",
    # Mock factories
    "create_mock_agent",
    "create_mock_mcp_server",
    "create_mock_llm_config",
    "create_mock_environment",
    # Server management
    "UnifiedMCPServerManager",
    # Test data
    "create_test_document",
    "generate_unique_name",
    "SAMPLE_CHAPTER_CONTENT",
    "SAMPLE_SEARCH_TEXT",
    # Environment validation
    "check_api_keys_available",
    "validate_agent_environment",
    "validate_package_imports",
    "validate_simple_agent_imports",
    "validate_react_agent_imports",
    "has_real_api_key",
]
