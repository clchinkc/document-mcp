"""Unit tests for agent performance metrics functionality.

This module tests core agent performance metrics collection and reporting
to ensure accurate performance monitoring capabilities.
"""

import time
from unittest.mock import Mock
from unittest.mock import patch

from src.agents.shared.performance_metrics import AgentPerformanceMetrics
from src.agents.shared.performance_metrics import MetricsCollectionContext
from src.agents.shared.performance_metrics import build_response_data


class TestAgentPerformanceMetrics:
    """Test AgentPerformanceMetrics class functionality."""

    def test_performance_metrics_initialization(self):
        """Test AgentPerformanceMetrics initialization."""
        metrics = AgentPerformanceMetrics("simple")

        assert metrics.agent_type == "simple"
        assert metrics.start_time is not None
        assert metrics.end_time is None
        assert metrics.execution_time == 0.0
        assert metrics.success is False

    def test_performance_metrics_mark_completed_success(self):
        """Test marking metrics as completed successfully."""
        metrics = AgentPerformanceMetrics("react")
        time.sleep(0.01)

        metrics.mark_completed(success=True)

        assert metrics.end_time is not None
        assert metrics.execution_time > 0
        assert metrics.success is True
        assert metrics.error_details is None

    def test_performance_metrics_mark_completed_failure(self):
        """Test marking metrics as completed with failure."""
        metrics = AgentPerformanceMetrics("simple")
        error_details = {"error_type": "ValidationError"}

        metrics.mark_completed(success=False, error_details=error_details)

        assert metrics.success is False
        assert metrics.error_details == error_details

    def test_performance_metrics_update_counters(self):
        """Test updating performance metric counters."""
        metrics = AgentPerformanceMetrics("react")

        metrics.total_tokens = 150
        metrics.llm_calls = 3
        metrics.tools_used = 7

        assert metrics.total_tokens == 150
        assert metrics.llm_calls == 3
        assert metrics.tools_used == 7


class TestMetricsCollectionContext:
    """Test MetricsCollectionContext functionality."""

    def test_metrics_collection_context_successful_completion(self):
        """Test MetricsCollectionContext with successful completion."""
        with patch("src.agents.shared.performance_metrics.AgentPerformanceMetrics") as mock_metrics_class:
            mock_metrics = Mock()
            mock_metrics_class.return_value = mock_metrics

            with MetricsCollectionContext("test_agent") as ctx:
                ctx.metrics.total_tokens = 100

            mock_metrics.mark_completed.assert_called_once_with(success=True)

    def test_metrics_collection_context_exception_handling(self):
        """Test MetricsCollectionContext with exception."""
        with patch("src.agents.shared.performance_metrics.AgentPerformanceMetrics") as mock_metrics_class:
            mock_metrics = Mock()
            mock_metrics_class.return_value = mock_metrics

            try:
                with MetricsCollectionContext("test_agent"):
                    raise Exception("Test error")
            except Exception:
                pass

            call_args = mock_metrics.mark_completed.call_args
            assert call_args[1]["success"] is False
            assert "error_details" in call_args[1]


class TestBuildResponseData:
    """Test build_response_data functionality."""

    def test_build_response_data_basic(self):
        """Test basic build_response_data functionality."""
        metrics = AgentPerformanceMetrics("simple")
        metrics.total_tokens = 150
        metrics.llm_calls = 2
        metrics.mark_completed(success=True)

        query = "Create a test document"
        response_data = {"summary": "Document created", "details": {"create_document": {"success": True}}}

        result = build_response_data(metrics, query, response_data)

        assert "agent_type" in result
        assert "execution_successful" in result
        assert "performance_metrics" in result
        assert result["agent_type"] == "simple"
        assert result["execution_successful"] is True
        assert result["query"] == query

    def test_build_response_data_with_error(self):
        """Test build_response_data with error metrics."""
        metrics = AgentPerformanceMetrics("react")
        error_details = {"error_type": "ValidationError"}
        metrics.mark_completed(success=False, error_details=error_details)

        query = "Invalid operation"
        response_data = {"summary": "Operation failed", "details": {}}

        result = build_response_data(metrics, query, response_data)

        assert result["execution_successful"] is False
        assert "error_details" in result["performance_metrics"]
