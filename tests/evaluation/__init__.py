"""Clean evaluation test suite for document-mcp agents.

This module provides an evaluation infrastructure following clean architecture:
- Agents collect performance metrics only (tokens, time, success)
- Tests optionally enhance with LLM evaluation in test layer only

Key Components:
- llm_evaluation_layer: Simple test-layer LLM evaluation enhancement
- test_simple_integration: Clean integration tests showing natural usage
- evaluation_utils: Legacy utilities for performance collection and analysis
- config: Configuration and thresholds for evaluation tests

Clean Architecture Usage:
    # Run clean integration tests
    pytest tests/evaluation/test_simple_integration.py -v

    # Run with LLM evaluation enabled (optional)
    ENABLE_LLM_EVALUATION=true pytest tests/evaluation/test_simple_integration.py -v

    # Demo clean architecture
    python3 tests/evaluation/test_simple_integration.py
"""

from .config import TEST_CATEGORIES
from .config import get_evaluation_config
from .config import get_operation_threshold
from .config import get_performance_thresholds
from .config import get_test_scenarios
from .evaluation_utils import EvaluationAssertions
from .evaluation_utils import MockDataGenerator
from .evaluation_utils import PerformanceTracker
from .evaluation_utils import TokenUsageMetrics
from .evaluation_utils import ToolCallMetrics
from .evaluation_utils import compare_agent_performance
from .evaluation_utils import generate_performance_summary

# Note: test_agent_performance.py removed due to clean architecture.
# Use simple test enhancement from llm_evaluation_layer.py instead.

__all__ = [
    # Utility classes
    "TokenUsageMetrics",
    "ToolCallMetrics",
    "EvaluationAssertions",
    "PerformanceTracker",
    "MockDataGenerator",
    # Configuration functions
    "get_evaluation_config",
    "get_performance_thresholds",
    "get_test_scenarios",
    "get_operation_threshold",
    "TEST_CATEGORIES",
    # Analysis functions
    "compare_agent_performance",
    "generate_performance_summary",
]
