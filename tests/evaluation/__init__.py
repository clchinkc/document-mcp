"""
Evaluation test suite for document-mcp agents.

This module provides comprehensive evaluation infrastructure for testing
agent performance, token usage, and reliability across different scenarios.

Key Components:
- test_agent_performance: Core evaluation tests with mock and real LLM support
- evaluation_utils: Utilities for metrics collection and analysis
- config: Configuration and thresholds for evaluation tests
- run_evaluation: Standalone evaluation runner script

Usage:
    # Run evaluation tests via pytest
    pytest tests/evaluation/ -m evaluation
    
    # Run standalone evaluation suite
    python3 tests/evaluation/run_evaluation.py
    
    # Run specific test categories
    python3 tests/evaluation/run_evaluation.py --categories basic intermediate
"""

from .test_agent_performance import (
    AgentPerformanceMetrics,
    AgentTestRunner,
    MockLLMResponse,
    print_performance_report
)
from .evaluation_utils import (
    TokenUsageMetrics,
    ToolCallMetrics,
    EvaluationAssertions,
    PerformanceTracker,
    MockDataGenerator,
    compare_agent_performance,
    generate_performance_summary
)
from .config import (
    get_evaluation_config,
    get_performance_thresholds,
    get_test_scenarios,
    get_operation_threshold,
    TEST_CATEGORIES
)

__all__ = [
    # Core evaluation classes
    'AgentPerformanceMetrics',
    'AgentTestRunner',
    'MockLLMResponse',
    
    # Utility classes
    'TokenUsageMetrics',
    'ToolCallMetrics',
    'EvaluationAssertions',
    'PerformanceTracker',
    'MockDataGenerator',
    
    # Configuration functions
    'get_evaluation_config',
    'get_performance_thresholds',
    'get_test_scenarios',
    'get_operation_threshold',
    'TEST_CATEGORIES',
    
    # Analysis functions
    'compare_agent_performance',
    'generate_performance_summary',
    'print_performance_report'
]