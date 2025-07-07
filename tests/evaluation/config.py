"""
Configuration for evaluation tests.

This module provides centralized configuration for the evaluation test suite,
including performance thresholds, test scenarios, and environment settings.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class PerformanceThresholds:
    """Performance thresholds for different agent types and operations."""

    # Token usage thresholds
    simple_agent_max_tokens: Dict[str, int]
    react_agent_max_tokens: Dict[str, int]
    planner_agent_max_tokens: Dict[str, int]

    # Execution time thresholds (in seconds)
    simple_agent_max_time: Dict[str, float]
    react_agent_max_time: Dict[str, float]
    planner_agent_max_time: Dict[str, float]

    # Tool call thresholds
    max_tool_calls: Dict[str, int]


@dataclass
class EvaluationConfig:
    """Main configuration for evaluation tests."""

    # Performance thresholds (required)
    thresholds: PerformanceThresholds

    # Test scenarios (required)
    test_scenarios: List[Dict]

    # Test execution settings
    default_timeout: int = 60
    mock_llm_delay: float = 0.1
    real_llm_timeout: int = 120

    # Environment settings
    mock_llm_by_default: bool = True
    enable_performance_tracking: bool = True
    enable_file_system_validation: bool = True

    # Reporting settings
    generate_performance_report: bool = True
    save_metrics_to_file: bool = False
    metrics_file_path: str = "evaluation_metrics.json"


# Default performance thresholds
DEFAULT_THRESHOLDS = PerformanceThresholds(
    simple_agent_max_tokens={
        "create_document": 200,
        "create_chapter": 250,
        "list_documents": 150,
        "read_document": 300,
        "update_chapter": 400,
        "delete_document": 150,
        "search_content": 350,
        "document_statistics": 200,
    },
    react_agent_max_tokens={
        "create_document": 500,
        "create_chapter": 600,
        "list_documents": 400,
        "read_document": 700,
        "update_chapter": 800,
        "delete_document": 400,
        "search_content": 800,
        "document_statistics": 500,
    },
    planner_agent_max_tokens={
        "create_document": 350,
        "create_chapter": 400,
        "list_documents": 300,
        "read_document": 500,
        "update_chapter": 600,
        "delete_document": 300,
        "search_content": 600,
        "document_statistics": 350,
    },
    simple_agent_max_time={
        "create_document": 5.0,
        "create_chapter": 7.0,
        "list_documents": 3.0,
        "read_document": 5.0,
        "update_chapter": 8.0,
        "delete_document": 3.0,
        "search_content": 10.0,
        "document_statistics": 5.0,
    },
    react_agent_max_time={
        "create_document": 15.0,
        "create_chapter": 20.0,
        "list_documents": 10.0,
        "read_document": 15.0,
        "update_chapter": 25.0,
        "delete_document": 10.0,
        "search_content": 30.0,
        "document_statistics": 15.0,
    },
    planner_agent_max_time={
        "create_document": 10.0,
        "create_chapter": 12.0,
        "list_documents": 7.0,
        "read_document": 10.0,
        "update_chapter": 15.0,
        "delete_document": 7.0,
        "search_content": 20.0,
        "document_statistics": 10.0,
    },
    max_tool_calls={
        "create_document": 3,
        "create_chapter": 5,
        "list_documents": 2,
        "read_document": 3,
        "update_chapter": 7,
        "delete_document": 2,
        "search_content": 5,
        "document_statistics": 4,
    },
)

# Default test scenarios
DEFAULT_TEST_SCENARIOS = [
    {
        "name": "basic_document_creation",
        "description": "Create a simple document",
        "query": 'Create a new document called "evaluation_test_doc"',
        "expected_operations": ["create_document"],
        "expected_files": ["evaluation_test_doc/"],
        "category": "basic",
    },
    {
        "name": "document_with_single_chapter",
        "description": "Create a document with one chapter",
        "query": 'Create a document called "guide" with a chapter "introduction"',
        "expected_operations": ["create_document", "create_chapter"],
        "expected_files": ["guide/", "guide/introduction.md"],
        "category": "basic",
    },
    {
        "name": "multiple_chapters_creation",
        "description": "Create a document with multiple chapters",
        "query": 'Create a book called "tutorial" with chapters: basics, advanced, conclusion',
        "expected_operations": [
            "create_document",
            "create_chapter",
            "create_chapter",
            "create_chapter",
        ],
        "expected_files": [
            "tutorial/",
            "tutorial/basics.md",
            "tutorial/advanced.md",
            "tutorial/conclusion.md",
        ],
        "category": "intermediate",
    },
    {
        "name": "document_listing",
        "description": "List all documents in the system",
        "query": "List all documents",
        "expected_operations": ["list_documents"],
        "expected_files": [],
        "category": "query",
    },
    {
        "name": "document_reading",
        "description": "Read a specific document",
        "query": 'Read the contents of document "tutorial"',
        "expected_operations": ["read_document"],
        "expected_files": [],
        "category": "query",
    },
    {
        "name": "content_search",
        "description": "Search for content across documents",
        "query": 'Search for "introduction" in all documents',
        "expected_operations": ["search_content"],
        "expected_files": [],
        "category": "advanced",
    },
    {
        "name": "document_statistics",
        "description": "Get statistics for a document",
        "query": 'Show statistics for document "tutorial"',
        "expected_operations": ["get_document_statistics"],
        "expected_files": [],
        "category": "advanced",
    },
    {
        "name": "chapter_update",
        "description": "Update content of a chapter",
        "query": 'Update the "basics" chapter in "tutorial" with new content about fundamentals',
        "expected_operations": ["update_chapter"],
        "expected_files": [],
        "category": "intermediate",
    },
    {
        "name": "document_deletion",
        "description": "Delete a document",
        "query": 'Delete the document "old_tutorial"',
        "expected_operations": ["delete_document"],
        "expected_files": [],
        "category": "basic",
    },
    {
        "name": "complex_workflow",
        "description": "Complex multi-step workflow",
        "query": 'Create a manual called "user_guide" with chapters for installation, configuration, and troubleshooting, then show its statistics',
        "expected_operations": [
            "create_document",
            "create_chapter",
            "create_chapter",
            "create_chapter",
            "get_document_statistics",
        ],
        "expected_files": [
            "user_guide/",
            "user_guide/installation.md",
            "user_guide/configuration.md",
            "user_guide/troubleshooting.md",
        ],
        "category": "complex",
    },
]

# Default evaluation configuration
DEFAULT_EVALUATION_CONFIG = EvaluationConfig(
    thresholds=DEFAULT_THRESHOLDS,
    test_scenarios=DEFAULT_TEST_SCENARIOS,
    default_timeout=60,
    mock_llm_delay=0.1,
    real_llm_timeout=120,
    mock_llm_by_default=True,
    enable_performance_tracking=True,
    enable_file_system_validation=True,
    generate_performance_report=True,
    save_metrics_to_file=False,
    metrics_file_path="evaluation_metrics.json",
)


def get_evaluation_config() -> EvaluationConfig:
    """Get the default evaluation configuration."""
    return DEFAULT_EVALUATION_CONFIG


def get_performance_thresholds() -> PerformanceThresholds:
    """Get the default performance thresholds."""
    return DEFAULT_THRESHOLDS


def get_test_scenarios(category: Optional[str] = None) -> List[Dict]:
    """Get test scenarios, optionally filtered by category."""
    scenarios = DEFAULT_TEST_SCENARIOS

    if category:
        scenarios = [s for s in scenarios if s.get("category") == category]

    return scenarios


def get_operation_threshold(
    agent_type: str, operation: str, metric_type: str
) -> Optional[float]:
    """Get the threshold for a specific agent type, operation, and metric type."""
    thresholds = get_performance_thresholds()

    if metric_type == "tokens":
        if agent_type == "simple":
            return thresholds.simple_agent_max_tokens.get(operation)
        elif agent_type == "react":
            return thresholds.react_agent_max_tokens.get(operation)
    elif metric_type == "time":
        if agent_type == "simple":
            return thresholds.simple_agent_max_time.get(operation)
        elif agent_type == "react":
            return thresholds.react_agent_max_time.get(operation)
    elif metric_type == "tool_calls":
        return thresholds.max_tool_calls.get(operation)

    return None


# Test categories for organizational purposes
TEST_CATEGORIES = {
    "basic": "Basic single-operation tests",
    "intermediate": "Multi-step operations",
    "advanced": "Complex analysis and search operations",
    "complex": "Multi-step workflows with multiple operations",
    "query": "Read-only query operations",
    "error": "Error handling and edge cases",
}


def get_test_category_description(category: str) -> str:
    """Get description for a test category."""
    return TEST_CATEGORIES.get(category, "Unknown category")
