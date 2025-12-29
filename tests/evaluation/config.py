"""Configuration for evaluation tests.

This module provides centralized configuration for the evaluation test suite,
including performance thresholds, test scenarios, and environment settings.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PerformanceThresholds:
    """Performance thresholds for different agent types and operations."""

    # Token usage thresholds
    simple_agent_max_tokens: dict[str, int]
    react_agent_max_tokens: dict[str, int]
    planner_agent_max_tokens: dict[str, int]

    # Execution time thresholds (in seconds)
    simple_agent_max_time: dict[str, float]
    react_agent_max_time: dict[str, float]
    planner_agent_max_time: dict[str, float]

    # Tool call thresholds
    max_tool_calls: dict[str, int]


@dataclass
class EvaluationConfig:
    """Main configuration for evaluation tests."""

    # Performance thresholds (required)
    thresholds: PerformanceThresholds

    # Test scenarios (required)
    test_scenarios: list[dict]

    # Test execution settings
    default_timeout: int = 60
    mock_llm_delay: float = 0.1
    real_llm_timeout: int = 120

    # Environment settings
    mock_llm_by_default: bool = True
    enable_performance_tracking: bool = True
    enable_file_system_validation: bool = True

    # LLM Evaluation settings
    enable_llm_evaluation: bool = True
    llm_evaluator_model: str = "google/gemini-3-flash-preview"
    llm_evaluation_timeout: int = 30
    enable_multi_judge_evaluation: bool = False
    multi_judge_models: list[str] = None
    llm_evaluation_retries: int = 2

    # LLM Evaluation criteria weights
    llm_evaluation_weights: dict[str, float] = None

    # Combined scoring configuration
    performance_weight: float = 0.7  # Weight for performance metrics
    quality_weight: float = 0.3  # Weight for LLM quality assessment

    # Comparative evaluation settings
    enable_comparative_evaluation: bool = True
    comparative_evaluation_threshold: float = 0.1  # Min difference for significance

    # Reporting settings
    generate_performance_report: bool = True
    save_metrics_to_file: bool = False
    metrics_file_path: str = "evaluation_metrics.json"
    include_llm_feedback_in_reports: bool = True

    def __post_init__(self):
        if self.multi_judge_models is None:
            self.multi_judge_models = ["gemini-2.5-flash", "gemini-2.5-flash-lite"]

        if self.llm_evaluation_weights is None:
            self.llm_evaluation_weights = {
                "accuracy": 0.25,
                "clarity": 0.15,
                "completeness": 0.25,
                "relevance": 0.15,
                "tool_appropriateness": 0.10,
                "document_structure": 0.05,
                "content_quality": 0.05,
            }


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

# Default test scenarios enhanced with LLM evaluation criteria
DEFAULT_TEST_SCENARIOS = [
    {
        "name": "basic_document_creation",
        "description": "Create a simple document",
        "query": 'Create a new document called "evaluation_test_doc"',
        "expected_operations": ["create_document"],
        "expected_files": ["evaluation_test_doc/"],
        "category": "basic",
        "llm_evaluation_focus": ["accuracy", "completeness", "tool_appropriateness"],
        "quality_expectations": {
            "accuracy": 0.8,
            "completeness": 0.9,
            "tool_appropriateness": 0.8,
        },
    },
    {
        "name": "document_with_single_chapter",
        "description": "Create a document with one chapter",
        "query": 'Create a document called "guide" with a chapter "introduction"',
        "expected_operations": ["create_document", "create_chapter"],
        "expected_files": ["guide/", "guide/introduction.md"],
        "category": "basic",
        "llm_evaluation_focus": [
            "completeness",
            "document_structure",
            "content_quality",
        ],
        "quality_expectations": {
            "completeness": 0.8,
            "document_structure": 0.7,
            "content_quality": 0.7,
        },
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
        "llm_evaluation_focus": [
            "completeness",
            "document_structure",
            "tool_appropriateness",
        ],
        "quality_expectations": {
            "completeness": 0.9,
            "document_structure": 0.8,
            "tool_appropriateness": 0.8,
        },
    },
    {
        "name": "document_listing",
        "description": "List all documents in the system",
        "query": "List all documents",
        "expected_operations": ["list_documents"],
        "expected_files": [],
        "category": "query",
        "llm_evaluation_focus": ["accuracy", "clarity", "relevance"],
        "quality_expectations": {"accuracy": 0.9, "clarity": 0.8, "relevance": 0.9},
    },
    {
        "name": "document_reading",
        "description": "Read a specific document",
        "query": 'Read the contents of document "tutorial"',
        "expected_operations": ["read_document"],
        "expected_files": [],
        "category": "query",
        "llm_evaluation_focus": ["accuracy", "clarity", "completeness"],
        "quality_expectations": {"accuracy": 0.8, "clarity": 0.8, "completeness": 0.8},
    },
    {
        "name": "content_search",
        "description": "Search for content across documents",
        "query": 'Search for "introduction" in all documents',
        "expected_operations": ["search_content"],
        "expected_files": [],
        "category": "advanced",
        "llm_evaluation_focus": ["accuracy", "relevance", "completeness"],
        "quality_expectations": {
            "accuracy": 0.8,
            "relevance": 0.9,
            "completeness": 0.7,
        },
    },
    {
        "name": "document_statistics",
        "description": "Get statistics for a document",
        "query": 'Show statistics for document "tutorial"',
        "expected_operations": ["get_statistics"],
        "expected_files": [],
        "category": "advanced",
        "llm_evaluation_focus": ["accuracy", "clarity", "completeness"],
        "quality_expectations": {"accuracy": 0.9, "clarity": 0.8, "completeness": 0.8},
    },
    {
        "name": "chapter_update",
        "description": "Update content of a chapter",
        "query": 'Update the "basics" chapter in "tutorial" with new content about fundamentals',
        "expected_operations": ["update_chapter"],
        "expected_files": [],
        "category": "intermediate",
        "llm_evaluation_focus": ["accuracy", "content_quality", "tool_appropriateness"],
        "quality_expectations": {
            "accuracy": 0.8,
            "content_quality": 0.7,
            "tool_appropriateness": 0.8,
        },
    },
    {
        "name": "document_deletion",
        "description": "Delete a document",
        "query": 'Delete the document "old_tutorial"',
        "expected_operations": ["delete_document"],
        "expected_files": [],
        "category": "basic",
        "llm_evaluation_focus": ["accuracy", "tool_appropriateness", "completeness"],
        "quality_expectations": {
            "accuracy": 0.9,
            "tool_appropriateness": 0.8,
            "completeness": 0.8,
        },
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
            "get_statistics",
        ],
        "expected_files": [
            "user_guide/",
            "user_guide/installation.md",
            "user_guide/configuration.md",
            "user_guide/troubleshooting.md",
        ],
        "category": "complex",
        "llm_evaluation_focus": [
            "completeness",
            "document_structure",
            "tool_appropriateness",
        ],
        "quality_expectations": {
            "completeness": 0.9,
            "document_structure": 0.8,
            "tool_appropriateness": 0.8,
        },
    },
    # Enhanced scenarios for LLM evaluation
    {
        "name": "error_handling_scenario",
        "description": "Test error handling with invalid operations",
        "query": "Try to create a document with invalid characters: <>|*",
        "expected_operations": ["create_document"],
        "expected_files": [],
        "category": "error",
        "llm_evaluation_focus": ["accuracy", "clarity", "tool_appropriateness"],
        "quality_expectations": {
            "accuracy": 0.7,
            "clarity": 0.8,
            "tool_appropriateness": 0.7,
        },
    },
    {
        "name": "performance_optimization_scenario",
        "description": "Test performance with large-scale operations",
        "query": "Create a comprehensive technical documentation with 10 chapters covering: overview, installation, configuration, api-reference, tutorials, examples, troubleshooting, faq, changelog, and appendix",
        "expected_operations": ["create_document"] + ["create_chapter"] * 10,
        "expected_files": ["technical_docs/"]
        + [
            f"technical_docs/{chapter}.md"
            for chapter in [
                "overview",
                "installation",
                "configuration",
                "api-reference",
                "tutorials",
                "examples",
                "troubleshooting",
                "faq",
                "changelog",
                "appendix",
            ]
        ],
        "category": "performance",
        "llm_evaluation_focus": [
            "completeness",
            "document_structure",
            "tool_appropriateness",
        ],
        "quality_expectations": {
            "completeness": 0.9,
            "document_structure": 0.8,
            "tool_appropriateness": 0.8,
        },
    },
    {
        "name": "content_quality_scenario",
        "description": "Test content quality and formatting",
        "query": "Create a user manual called 'style_guide' with a chapter 'writing_standards' that includes proper formatting, headers, and structure",
        "expected_operations": ["create_document", "create_chapter"],
        "expected_files": ["style_guide/", "style_guide/writing_standards.md"],
        "category": "quality",
        "llm_evaluation_focus": ["content_quality", "document_structure", "clarity"],
        "quality_expectations": {
            "content_quality": 0.8,
            "document_structure": 0.8,
            "clarity": 0.8,
        },
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


def get_test_scenarios(category: str | None = None) -> list[dict]:
    """Get test scenarios, optionally filtered by category."""
    scenarios = DEFAULT_TEST_SCENARIOS

    if category:
        scenarios = [s for s in scenarios if s.get("category") == category]

    return scenarios


def get_operation_threshold(agent_type: str, operation: str, metric_type: str) -> float | None:
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


def get_llm_evaluation_config() -> dict[str, any]:
    """Get LLM evaluation configuration."""
    config = get_evaluation_config()
    return {
        "enable_llm_evaluation": config.enable_llm_evaluation,
        "llm_evaluator_model": config.llm_evaluator_model,
        "llm_evaluation_timeout": config.llm_evaluation_timeout,
        "enable_multi_judge_evaluation": config.enable_multi_judge_evaluation,
        "multi_judge_models": config.multi_judge_models,
        "llm_evaluation_retries": config.llm_evaluation_retries,
        "llm_evaluation_weights": config.llm_evaluation_weights,
        "performance_weight": config.performance_weight,
        "quality_weight": config.quality_weight,
        "enable_comparative_evaluation": config.enable_comparative_evaluation,
        "comparative_evaluation_threshold": config.comparative_evaluation_threshold,
        "include_llm_feedback_in_reports": config.include_llm_feedback_in_reports,
    }


def get_scenario_quality_expectations(scenario_name: str) -> dict[str, float]:
    """Get quality expectations for a specific scenario."""
    scenarios = get_test_scenarios()
    for scenario in scenarios:
        if scenario["name"] == scenario_name:
            return scenario.get("quality_expectations", {})
    return {}


def get_scenario_evaluation_focus(scenario_name: str) -> list[str]:
    """Get LLM evaluation focus criteria for a specific scenario."""
    scenarios = get_test_scenarios()
    for scenario in scenarios:
        if scenario["name"] == scenario_name:
            return scenario.get("llm_evaluation_focus", [])
    return []


def get_scenarios_by_category(category: str) -> list[dict]:
    """Get scenarios filtered by category with LLM evaluation criteria."""
    scenarios = get_test_scenarios(category)
    return [
        {
            **scenario,
            "has_llm_evaluation": "llm_evaluation_focus" in scenario,
            "quality_expectations": scenario.get("quality_expectations", {}),
            "evaluation_focus": scenario.get("llm_evaluation_focus", []),
        }
        for scenario in scenarios
    ]


def validate_scenario_quality_expectations(scenario: dict) -> bool:
    """Validate that scenario quality expectations are properly configured."""
    if "quality_expectations" not in scenario:
        return True  # Optional field

    expectations = scenario["quality_expectations"]
    valid_criteria = [
        "accuracy",
        "clarity",
        "completeness",
        "relevance",
        "tool_appropriateness",
        "document_structure",
        "content_quality",
    ]

    for criterion, expected_score in expectations.items():
        if criterion not in valid_criteria:
            return False
        if not isinstance(expected_score, int | float) or not (0.0 <= expected_score <= 1.0):
            return False

    return True


def get_combined_score_thresholds(agent_type: str, operation: str) -> dict[str, float]:
    """Get combined score thresholds for performance + quality evaluation."""
    config = get_evaluation_config()

    # Get base performance thresholds
    perf_thresholds = get_performance_thresholds()

    # Calculate combined thresholds based on weights
    performance_weight = config.performance_weight
    quality_weight = config.quality_weight

    thresholds = {}

    # Time-based threshold (inverted for performance index)
    if agent_type == "simple":
        max_time = perf_thresholds.simple_agent_max_time.get(operation, 5.0)
    elif agent_type == "react":
        max_time = perf_thresholds.react_agent_max_time.get(operation, 15.0)
    else:
        max_time = 10.0

    # Performance component (higher is better)
    performance_threshold = 1.0 / max_time

    # Quality component (assuming minimum acceptable quality of 0.7)
    quality_threshold = 0.7

    # Combined threshold
    combined_threshold = (performance_weight * performance_threshold) + (quality_weight * quality_threshold)

    thresholds = {
        "performance_threshold": performance_threshold,
        "quality_threshold": quality_threshold,
        "combined_threshold": combined_threshold,
        "performance_weight": performance_weight,
        "quality_weight": quality_weight,
    }

    return thresholds
