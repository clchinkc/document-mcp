"""Test metrics collection functionality."""

import time
import pytest
from document_mcp.logger_config import log_mcp_call


def test_metrics_decorator():
    """Test that the metrics decorator works without errors."""
    
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
    """Test that metrics modules import correctly."""
    from document_mcp.metrics_config import (
        is_metrics_enabled,
        get_metrics_export
    )
    
    # Test basic functionality
    assert isinstance(is_metrics_enabled(), bool)
    
    # Test metrics export
    metrics_data, content_type = get_metrics_export()
    assert isinstance(metrics_data, str)
    assert isinstance(content_type, str)


def test_server_import():
    """Test that server imports with metrics endpoint."""
    import document_mcp.doc_tool_server
    # If this doesn't raise an error, the server imports correctly 