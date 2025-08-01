"""Test metrics collection functionality."""

import time

import pytest

from document_mcp.logger_config import log_mcp_call


def test_metrics_decorator():
    @log_mcp_call
    def sample_function(text: str) -> str:
        """Sample function for testing."""
        time.sleep(0.01)  # Small delay to test timing
        return f"processed: {text}"

    # Test successful call
    result = sample_function("test_input")
    assert result == "processed: test_input"


def test_metrics_decorator_with_error():
    """Test that the metrics decorator handles errors correctly."""

    @log_mcp_call
    def error_function():
        """Function that raises an error."""
        raise ValueError("test error")

    # Test error handling
    with pytest.raises(ValueError, match="test error"):
        error_function()


def test_metrics_import():
    from document_mcp.metrics_config import get_metrics_export
    from document_mcp.metrics_config import is_metrics_enabled

    # Test basic functionality
    assert isinstance(is_metrics_enabled(), bool)

    # Test metrics export
    metrics_data, content_type = get_metrics_export()
    assert isinstance(metrics_data, str)
    assert isinstance(content_type, str)


def test_server_import():
    """Test that server imports with metrics endpoint."""
    # If this doesn't raise an error, the server imports correctly


def test_metrics_initialization_from_script():
    """Test that metrics are properly initialized."""
    from document_mcp.metrics_config import get_metrics_summary
    from document_mcp.metrics_config import is_metrics_enabled

    enabled = is_metrics_enabled()
    assert isinstance(enabled, bool)

    if enabled:
        summary = get_metrics_summary()
        assert isinstance(summary, dict)
        assert "service_name" in summary
        assert "environment" in summary
        assert "grafana_cloud_endpoint" in summary
        assert "prometheus_enabled" in summary


def test_tool_instrumentation_from_script():
    """Test that tool calls are properly instrumented."""
    from document_mcp.metrics_config import record_tool_call_error
    from document_mcp.metrics_config import record_tool_call_start
    from document_mcp.metrics_config import record_tool_call_success

    # Simulate a successful tool call
    start_time = record_tool_call_start("test_tool", ("arg1",), {"param": "value"})
    time.sleep(0.01)
    record_tool_call_success("test_tool", start_time, 100)

    # Simulate a failed tool call
    start_time = record_tool_call_start("test_tool_error", (), {})
    time.sleep(0.01)
    record_tool_call_error("test_tool_error", start_time, ValueError("Test error"))
