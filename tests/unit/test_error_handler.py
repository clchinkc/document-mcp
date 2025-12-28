"""Unit tests for the error handler utilities."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from document_mcp.error_handler import ErrorContext
from document_mcp.error_handler import create_error_response
from document_mcp.error_handler import handle_mcp_tool_error
from document_mcp.error_handler import log_operation_start
from document_mcp.error_handler import log_operation_success
from document_mcp.error_handler import safe_operation
from document_mcp.error_handler import validate_field_type
from document_mcp.error_handler import validate_required_fields
from document_mcp.exceptions import DocumentMCPError
from document_mcp.exceptions import OperationError
from document_mcp.exceptions import ValidationError


class TestSafeOperation:
    """Tests for safe_operation decorator."""

    def test_successful_operation(self):
        """Test decorator with successful operation."""

        @safe_operation("test_op")
        def successful_func(x: int) -> int:
            return x * 2

        result = successful_func(5)
        assert result == 10

    def test_returns_default_on_document_mcp_error(self):
        """Test that default value is returned on DocumentMCPError."""

        @safe_operation("test_op", default_return="default")
        def failing_func():
            raise DocumentMCPError("Test error")

        result = failing_func()
        assert result == "default"

    def test_returns_default_on_generic_error(self):
        """Test that default value is returned on generic exception."""

        @safe_operation("test_op", default_return=None)
        def failing_func():
            raise ValueError("Generic error")

        result = failing_func()
        assert result is None

    def test_raises_on_error_when_enabled(self):
        """Test that exceptions are re-raised when raise_on_error=True."""

        @safe_operation("test_op", raise_on_error=True)
        def failing_func():
            raise DocumentMCPError("Test error")

        with pytest.raises(DocumentMCPError):
            failing_func()

    def test_raises_operation_error_on_generic_exception(self):
        """Test that generic exceptions are wrapped in OperationError."""

        @safe_operation("test_op", raise_on_error=True)
        def failing_func():
            raise ValueError("Original error")

        with pytest.raises(OperationError) as exc_info:
            failing_func()

        assert "test_op" in str(exc_info.value)

    def test_logs_document_mcp_error(self):
        """Test that DocumentMCPError is logged."""
        with patch("document_mcp.error_handler.logger") as mock_logger:

            @safe_operation("test_op", log_errors=True)
            def failing_func():
                raise DocumentMCPError("Test error", error_code="TEST")

            failing_func()
            mock_logger.error.assert_called()

    def test_logs_generic_error(self):
        """Test that generic exceptions are logged."""
        with patch("document_mcp.error_handler.logger") as mock_logger:

            @safe_operation("test_op", log_errors=True)
            def failing_func():
                raise ValueError("Generic error")

            failing_func()
            mock_logger.error.assert_called()

    def test_no_logging_when_disabled(self):
        """Test that errors are not logged when log_errors=False."""
        with patch("document_mcp.error_handler.logger") as mock_logger:

            @safe_operation("test_op", log_errors=False)
            def failing_func():
                raise ValueError("Error")

            failing_func()
            mock_logger.error.assert_not_called()


class TestCreateErrorResponse:
    """Tests for create_error_response function."""

    def test_with_document_mcp_error(self):
        """Test response creation with DocumentMCPError."""
        error = DocumentMCPError(
            message="Technical message",
            error_code="TEST_ERROR",
            user_message="User message",
            details={"key": "value"},
        )

        response = create_error_response(error, "test_operation")

        assert response.success is False
        assert response.message == "User message"
        assert response.details["error_code"] == "TEST_ERROR"
        assert response.details["operation"] == "test_operation"

    def test_with_generic_exception(self):
        """Test response creation with generic exception."""
        error = ValueError("Something went wrong")

        response = create_error_response(error, "test_operation")

        assert response.success is False
        assert "unexpected error" in response.message.lower()
        assert response.details["error_code"] == "UNEXPECTED_ERROR"
        assert response.details["error_type"] == "ValueError"


class TestHandleMcpToolError:
    """Tests for handle_mcp_tool_error function."""

    def test_handles_document_mcp_error(self):
        """Test handling DocumentMCPError."""
        error = DocumentMCPError("Tool failed")

        with patch("document_mcp.error_handler.logger"):
            response = handle_mcp_tool_error("list_documents", error)

        assert response.success is False

    def test_handles_generic_error(self):
        """Test handling generic exception."""
        error = ValueError("Connection failed")

        with patch("document_mcp.error_handler.logger"):
            response = handle_mcp_tool_error("create_document", error, {"doc": "test"})

        assert response.success is False

    def test_logs_error_with_context(self):
        """Test that error is logged with context."""
        with patch("document_mcp.error_handler.logger") as mock_logger:
            handle_mcp_tool_error("test_tool", ValueError("Error"), {"key": "value"})

            mock_logger.error.assert_called_once()
            call_kwargs = mock_logger.error.call_args
            assert "test_tool" in call_kwargs[0][0]


class TestLogOperationStart:
    """Tests for log_operation_start function."""

    def test_logs_operation_name(self):
        """Test that operation name is logged."""
        with patch("document_mcp.error_handler.logger") as mock_logger:
            log_operation_start("my_operation")

            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args
            assert "my_operation" in call_args[0][0]

    def test_logs_with_context(self):
        """Test that context is included in log."""
        with patch("document_mcp.error_handler.logger") as mock_logger:
            log_operation_start("operation", key="value", count=5)

            mock_logger.info.assert_called_once()
            call_kwargs = mock_logger.info.call_args[1]
            assert "context" in call_kwargs["extra"]


class TestLogOperationSuccess:
    """Tests for log_operation_success function."""

    def test_logs_without_result(self):
        """Test logging without result."""
        with patch("document_mcp.error_handler.logger") as mock_logger:
            log_operation_success("my_operation")

            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args
            assert "my_operation" in call_args[0][0]
            assert "successfully" in call_args[0][0]

    def test_logs_object_result(self):
        """Test logging with object result."""

        class MyObject:
            pass

        with patch("document_mcp.error_handler.logger") as mock_logger:
            log_operation_success("operation", result=MyObject())

            call_kwargs = mock_logger.info.call_args[1]
            assert "MyObject" in call_kwargs["extra"]["result_summary"]

    def test_logs_list_result(self):
        """Test logging with list result."""
        with patch("document_mcp.error_handler.logger") as mock_logger:
            log_operation_success("operation", result=[1, 2, 3])

            call_kwargs = mock_logger.info.call_args[1]
            assert "3 items" in call_kwargs["extra"]["result_summary"]

    def test_logs_dict_result(self):
        """Test logging with dict result."""
        with patch("document_mcp.error_handler.logger") as mock_logger:
            log_operation_success("operation", result={"a": 1, "b": 2})

            call_kwargs = mock_logger.info.call_args[1]
            assert "2 items" in call_kwargs["extra"]["result_summary"]

    def test_logs_string_result_truncated(self):
        """Test that long string results are truncated."""
        with patch("document_mcp.error_handler.logger") as mock_logger:
            long_string = "x" * 200
            log_operation_success("operation", result=long_string)

            call_kwargs = mock_logger.info.call_args[1]
            # Should be truncated to 100 chars
            assert len(call_kwargs["extra"]["result_summary"]) <= 100


class TestValidateRequiredFields:
    """Tests for validate_required_fields function."""

    def test_valid_data(self):
        """Test with all required fields present."""
        data = {"name": "test", "value": 123}
        # Should not raise
        validate_required_fields(data, ["name", "value"])

    def test_missing_field(self):
        """Test with missing required field."""
        data = {"name": "test"}

        with pytest.raises(ValidationError) as exc_info:
            validate_required_fields(data, ["name", "value"])

        assert "value" in str(exc_info.value)
        assert "missing_fields" in exc_info.value.details

    def test_none_value_is_considered_missing(self):
        """Test that None values are considered missing."""
        data = {"name": "test", "value": None}

        with pytest.raises(ValidationError):
            validate_required_fields(data, ["name", "value"])


class TestValidateFieldType:
    """Tests for validate_field_type function."""

    def test_valid_type(self):
        """Test with correct type."""
        data = {"count": 5}
        # Should not raise
        validate_field_type(data, "count", int)

    def test_invalid_type(self):
        """Test with incorrect type."""
        data = {"count": "five"}

        with pytest.raises(ValidationError) as exc_info:
            validate_field_type(data, "count", int)

        assert "count" in str(exc_info.value)
        assert exc_info.value.details["expected_type"] == "int"
        assert exc_info.value.details["actual_type"] == "str"

    def test_missing_required_field(self):
        """Test with missing required field."""
        data = {}

        with pytest.raises(ValidationError) as exc_info:
            validate_field_type(data, "count", int, required=True)

        assert "count" in str(exc_info.value)

    def test_missing_optional_field(self):
        """Test with missing optional field."""
        data = {}
        # Should not raise
        validate_field_type(data, "count", int, required=False)

    def test_none_value_optional_field(self):
        """Test with None value for optional field."""
        data = {"count": None}
        # Should not raise
        validate_field_type(data, "count", int, required=False)


class TestErrorContext:
    """Tests for ErrorContext context manager."""

    def test_successful_operation(self):
        """Test context manager with successful operation."""
        with patch("document_mcp.error_handler.logger"):
            with ErrorContext("test_operation") as ctx:
                ctx.result = "success"

            # Should complete without error

    def test_logs_start_and_success(self):
        """Test that start and success are logged."""
        with patch("document_mcp.error_handler.log_operation_start") as mock_start:
            with patch("document_mcp.error_handler.log_operation_success") as mock_success:
                with ErrorContext("test_op", log_start=True, log_success=True) as ctx:
                    ctx.result = "done"

                mock_start.assert_called_once()
                mock_success.assert_called_once()

    def test_no_logging_when_disabled(self):
        """Test that logging can be disabled."""
        with patch("document_mcp.error_handler.log_operation_start") as mock_start:
            with patch("document_mcp.error_handler.log_operation_success") as mock_success:
                with ErrorContext("test_op", log_start=False, log_success=False):
                    pass

                mock_start.assert_not_called()
                mock_success.assert_not_called()

    def test_handles_document_mcp_error(self):
        """Test handling of DocumentMCPError."""
        with patch("document_mcp.error_handler.logger") as mock_logger:
            with pytest.raises(DocumentMCPError):
                with ErrorContext("test_op"):
                    raise DocumentMCPError("Test error")

            mock_logger.error.assert_called()

    def test_handles_generic_error(self):
        """Test handling of generic exception."""
        with patch("document_mcp.error_handler.logger") as mock_logger:
            with pytest.raises(ValueError):
                with ErrorContext("test_op"):
                    raise ValueError("Generic error")

            mock_logger.error.assert_called()

    def test_suppresses_error_when_configured(self):
        """Test that errors are suppressed when raise_on_error=False."""
        with patch("document_mcp.error_handler.logger"):
            # Should not raise
            with ErrorContext("test_op", raise_on_error=False):
                raise ValueError("This should be suppressed")

    def test_context_passed_to_logs(self):
        """Test that context is passed to log functions."""
        with patch("document_mcp.error_handler.log_operation_start") as mock_start:
            with ErrorContext("test_op", key="value"):
                pass

            call_kwargs = mock_start.call_args[1]
            assert call_kwargs["key"] == "value"
