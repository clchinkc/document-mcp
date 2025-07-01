"""
Unit tests for logger_config module.

This module tests the enhanced error handling and logging system including:
- StructuredLogFormatter JSON output
- ErrorCategory classification  
- log_structured_error function
- safe_operation wrapper
- Error context and metadata handling
"""

import json
import logging
import pytest
from io import StringIO

from document_mcp.logger_config import (
    ErrorCategory,
    StructuredLogFormatter,
    log_structured_error,
    safe_operation,
    log_mcp_call,
    error_logger
)


class TestErrorCategory:
    """Test suite for ErrorCategory enum."""

    def test_error_category_values(self):
        """Test that ErrorCategory enum has correct values."""
        assert ErrorCategory.CRITICAL.value == "CRITICAL"
        assert ErrorCategory.ERROR.value == "ERROR"
        assert ErrorCategory.WARNING.value == "WARNING"
        assert ErrorCategory.INFO.value == "INFO"


class TestStructuredLogFormatter:
    """Test suite for StructuredLogFormatter."""

    def test_formatter_basic_log_entry(self):
        formatter = StructuredLogFormatter()
        
        # Create a basic log record
        record = logging.LogRecord(
            name="test_logger",
            level=logging.ERROR,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        # Format the record
        formatted = formatter.format(record)
        
        # Parse as JSON to verify structure
        log_data = json.loads(formatted)
        
        assert log_data["level"] == "ERROR"
        assert log_data["logger"] == "test_logger"
        assert log_data["message"] == "Test message"
        assert "timestamp" in log_data

    def test_formatter_with_exception_info(self):
        """Test log formatting with exception information."""
        formatter = StructuredLogFormatter()
        
        # Create exception info
        try:
            raise ValueError("Test exception")
        except ValueError:
            import sys
            exc_info = sys.exc_info()
            
            record = logging.LogRecord(
                name="test_logger",
                level=logging.ERROR,
                pathname="test.py",
                lineno=10,
                msg="Error occurred",
                args=(),
                exc_info=exc_info
            )
            
            formatted = formatter.format(record)
            log_data = json.loads(formatted)
            
            assert "exception" in log_data
            assert log_data["exception"]["type"] == "ValueError"
            assert log_data["exception"]["message"] == "Test exception"
            assert isinstance(log_data["exception"]["traceback"], list)

    def test_formatter_with_extra_fields(self):
        formatter = StructuredLogFormatter()
        
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Operation completed",
            args=(),
            exc_info=None
        )
        
        # Add extra fields
        record.operation = "test_operation"
        record.document_name = "test_doc"
        record.error_category = "INFO"
        
        formatted = formatter.format(record)
        log_data = json.loads(formatted)
        
        assert log_data["operation"] == "test_operation"
        assert log_data["document_name"] == "test_doc"
        assert log_data["error_category"] == "INFO"


class TestLogStructuredError:
    """Test suite for log_structured_error function."""

    def test_log_structured_error_basic(self, mocker):
        mock_logger = mocker.patch('document_mcp.logger_config.error_logger')
        log_structured_error(
            category=ErrorCategory.ERROR,
            message="Test error message",
            operation="test_operation"
        )
        mock_logger.log.assert_called_once()
        call_args = mock_logger.log.call_args
        assert call_args[0][0] == logging.ERROR
        assert call_args[0][1] == "Test error message"
        extra = call_args[1]['extra']
        assert extra['error_category'] == "ERROR"
        assert extra['operation'] == "test_operation"

    def test_log_structured_error_with_exception(self, mocker):
        mock_logger = mocker.patch('document_mcp.logger_config.error_logger')
        test_exception = ValueError("Test exception")
        log_structured_error(
            category=ErrorCategory.CRITICAL,
            message="Critical error occurred",
            exception=test_exception,
            operation="critical_operation"
        )
        mock_logger.log.assert_called_once()
        call_args = mock_logger.log.call_args
        assert call_args[0][0] == logging.CRITICAL
        assert call_args[1]['exc_info'] is True

    def test_log_structured_error_with_context(self, mocker):
        mock_logger = mocker.patch('document_mcp.logger_config.error_logger')
        context = {
            "document_name": "test_doc",
            "chapter_name": "test_chapter.md",
            "file_path": "/path/to/file"
        }
        log_structured_error(
            category=ErrorCategory.WARNING,
            message="Warning message",
            context=context,
            operation="file_operation"
        )
        mock_logger.log.assert_called_once()
        call_args = mock_logger.log.call_args
        extra = call_args[1]['extra']
        assert extra['document_name'] == "test_doc"
        assert extra['chapter_name'] == "test_chapter.md"
        assert extra['file_path'] == "/path/to/file"


class TestSafeOperation:
    """Test suite for safe_operation wrapper function."""

    def test_safe_operation_success(self):
        """Returns success for safe_operation with valid function."""
        def successful_function(x, y):
            return x + y
        success, result, error = safe_operation(
            "addition",
            successful_function,
            5, 3
        )
        assert success is True
        assert result == 8
        assert error is None

    def test_safe_operation_failure(self, mocker):
        """Logs error for safe_operation with failing function."""
        mock_log_error = mocker.patch('document_mcp.logger_config.log_structured_error')
        def failing_function():
            raise RuntimeError("Operation failed")
        success, result, error = safe_operation(
            "failing_operation",
            failing_function,
            error_category=ErrorCategory.CRITICAL
        )
        assert success is False
        assert result is None
        assert isinstance(error, RuntimeError)
        mock_log_error.assert_called_once()
        call_args = mock_log_error.call_args
        assert call_args[1]['category'] == ErrorCategory.CRITICAL
        assert call_args[1]['operation'] == "failing_operation"
        assert isinstance(call_args[1]['exception'], RuntimeError)


class TestLogMCPCallDecorator:
    """Test suite for log_mcp_call decorator."""

    def test_log_mcp_call_success_no_metrics(self, mocker):
        """Test log_mcp_call decorator with successful function execution."""
        mock_logger = mocker.patch('document_mcp.logger_config.mcp_call_logger')
        mocker.patch('document_mcp.logger_config.METRICS_AVAILABLE', False)
        
        @log_mcp_call
        def test_function(arg1, arg2="default"):
            return {"result": arg1 + arg2}
        
        result = test_function("hello", arg2="world")
        
        assert result == {"result": "helloworld"}
        assert mock_logger.info.call_count == 2  # Start and end calls

    def test_log_mcp_call_with_exception(self, mocker):
        """Test log_mcp_call decorator with function that raises exception."""
        mock_logger = mocker.patch('document_mcp.logger_config.mcp_call_logger')
        mock_log_error = mocker.patch('document_mcp.logger_config.log_structured_error')
        mocker.patch('document_mcp.logger_config.METRICS_AVAILABLE', False)
        
        @log_mcp_call
        def failing_function():
            raise FileNotFoundError("File not found")
        
        with pytest.raises(FileNotFoundError):
            failing_function()
        
        # Verify both regular logging and structured error logging
        mock_logger.error.assert_called_once()
        mock_log_error.assert_called_once()
        
        # Check structured error logging
        call_args = mock_log_error.call_args
        assert call_args[1]['category'] == ErrorCategory.ERROR
        assert call_args[1]['operation'] == "tool_execution"
        assert call_args[1]['function'] == "failing_function"


class TestIntegrationWithActualLogger:
    """Integration tests with actual logger instances."""

    def test_error_logger_configuration(self):
        """Test that error_logger is properly configured."""
        assert error_logger.name == "error_logger"
        assert error_logger.level == logging.INFO
        assert len(error_logger.handlers) > 0
        
        # Check that at least one handler uses StructuredLogFormatter
        has_structured_formatter = any(
            isinstance(handler.formatter, StructuredLogFormatter)
            for handler in error_logger.handlers
        )
        assert has_structured_formatter

    def test_actual_structured_logging_output(self):
        """Test actual structured logging output to verify JSON format."""
        # Create a string buffer to capture log output
        log_stream = StringIO()
        handler = logging.StreamHandler(log_stream)
        handler.setFormatter(StructuredLogFormatter())
        
        # Create a temporary logger
        test_logger = logging.getLogger("test_structured")
        test_logger.addHandler(handler)
        test_logger.setLevel(logging.INFO)
        
        # Log a structured error
        try:
            raise ValueError("Test exception for logging")
        except ValueError as e:
            test_logger.error(
                "Test structured log entry",
                exc_info=True,
                extra={
                    "error_category": "ERROR",
                    "operation": "test_operation",
                    "document_name": "test_doc"
                }
            )
        
        # Get the logged output
        log_output = log_stream.getvalue()
        
        # Verify it's valid JSON
        log_data = json.loads(log_output.strip())
        
        assert log_data["level"] == "ERROR"
        assert log_data["message"] == "Test structured log entry"
        assert log_data["error_category"] == "ERROR"
        assert log_data["operation"] == "test_operation"
        assert log_data["document_name"] == "test_doc"
        assert "exception" in log_data
        assert log_data["exception"]["type"] == "ValueError" 