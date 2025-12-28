"""Unit tests for the GCP observability module."""

from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from document_mcp import observability


class TestCloudRunJSONFormatter:
    """Tests for CloudRunJSONFormatter."""

    def test_format_basic_message(self):
        """Test basic message formatting."""
        formatter = observability.CloudRunJSONFormatter()
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        parsed = json.loads(result)

        assert parsed["severity"] == "INFO"
        assert parsed["message"] == "Test message"
        assert parsed["logger"] == "test_logger"
        assert "timestamp" in parsed

    def test_format_with_trace_id(self):
        """Test formatting with trace correlation."""
        formatter = observability.CloudRunJSONFormatter()
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test with trace",
            args=(),
            exc_info=None,
        )
        record.trace_id = "abc123"

        with patch.object(observability, "GCP_PROJECT", "test-project"):
            result = formatter.format(record)

        parsed = json.loads(result)
        assert parsed["logging.googleapis.com/trace"] == "projects/test-project/traces/abc123"

    def test_format_with_span_id(self):
        """Test formatting with span ID."""
        formatter = observability.CloudRunJSONFormatter()
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test with span",
            args=(),
            exc_info=None,
        )
        record.span_id = "span456"

        result = formatter.format(record)
        parsed = json.loads(result)

        assert parsed["logging.googleapis.com/spanId"] == "span456"

    def test_format_with_extra_fields(self):
        """Test formatting with extra fields."""
        formatter = observability.CloudRunJSONFormatter()
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test with extras",
            args=(),
            exc_info=None,
        )
        record.extra_fields = {"tool_name": "list_documents", "duration": 0.5}

        result = formatter.format(record)
        parsed = json.loads(result)

        assert parsed["tool_name"] == "list_documents"
        assert parsed["duration"] == 0.5

    def test_format_with_exception(self):
        """Test formatting with exception info."""
        formatter = observability.CloudRunJSONFormatter()

        try:
            raise ValueError("Test error")
        except ValueError:
            import sys

            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test_logger",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="Error occurred",
            args=(),
            exc_info=exc_info,
        )

        result = formatter.format(record)
        parsed = json.loads(result)

        assert "exception" in parsed
        assert "ValueError" in parsed["exception"]
        assert "Test error" in parsed["exception"]


class TestSetupLogging:
    """Tests for setup_logging function."""

    def setup_method(self):
        """Reset global logger before each test."""
        observability._logger = None

    def teardown_method(self):
        """Clean up after each test."""
        observability._logger = None

    def test_setup_logging_returns_logger(self):
        """Test that setup_logging returns a logger."""
        logger = observability.setup_logging()
        assert isinstance(logger, logging.Logger)
        assert logger.name == "document_mcp"

    def test_setup_logging_caches_logger(self):
        """Test that logger is cached."""
        logger1 = observability.setup_logging()
        logger2 = observability.setup_logging()
        assert logger1 is logger2

    def test_setup_logging_local_format(self):
        """Test local development format when not on Cloud Run."""
        with patch.object(observability, "IS_CLOUD_RUN", False):
            observability._logger = None
            logger = observability.setup_logging()

            # Check handler exists
            assert len(logger.handlers) == 1
            handler = logger.handlers[0]

            # Local format should not be JSON
            assert not isinstance(handler.formatter, observability.CloudRunJSONFormatter)

    def test_setup_logging_cloud_run_format(self):
        """Test Cloud Run JSON format."""
        with patch.object(observability, "IS_CLOUD_RUN", True):
            observability._logger = None
            logger = observability.setup_logging()

            assert len(logger.handlers) == 1
            handler = logger.handlers[0]
            assert isinstance(handler.formatter, observability.CloudRunJSONFormatter)


class TestSetupTracing:
    """Tests for setup_tracing function."""

    def setup_method(self):
        """Reset global tracer before each test."""
        observability._tracer = None
        observability._tracer_provider = None
        observability._shutting_down = False

    def teardown_method(self):
        """Clean up after each test."""
        observability._tracer = None
        observability._tracer_provider = None
        observability._shutting_down = False

    def test_setup_tracing_returns_tracer_locally(self):
        """Test that tracing returns a tracer when running locally (uses ConsoleSpanExporter)."""
        with patch.object(observability, "IS_CLOUD_RUN", False):
            observability._tracer = None
            result = observability.setup_tracing()
            # Should return a tracer (uses ConsoleSpanExporter locally)
            assert result is not None

    def test_setup_tracing_returns_none_when_disabled(self):
        """Test that tracing returns None when observability is disabled."""
        with patch.object(observability, "OBSERVABILITY_ENABLED", False):
            observability._tracer = None
            result = observability.setup_tracing()
            assert result is None

    def test_setup_tracing_handles_import_error(self):
        """Test graceful handling of missing OpenTelemetry packages."""
        observability._tracer = None

        with patch.object(observability, "OBSERVABILITY_ENABLED", True):
            # Mock the import to raise ImportError
            import builtins

            original_import = builtins.__import__

            def mock_import(name, *args, **kwargs):
                if "opentelemetry" in name:
                    raise ImportError(f"No module named '{name}'")
                return original_import(name, *args, **kwargs)

            with patch.object(builtins, "__import__", side_effect=mock_import):
                result = observability.setup_tracing()
                assert result is None

    def test_setup_tracing_caches_tracer(self):
        """Test that tracer is cached."""
        mock_tracer = MagicMock()
        observability._tracer = mock_tracer

        result = observability.setup_tracing()
        assert result is mock_tracer


class TestSetupMetrics:
    """Tests for setup_metrics function."""

    def setup_method(self):
        """Reset global metrics before each test."""
        observability._meter_instruments = None
        observability._meter_provider = None
        observability._shutting_down = False

    def teardown_method(self):
        """Clean up after each test."""
        observability._meter_instruments = None
        observability._meter_provider = None
        observability._shutting_down = False

    def test_setup_metrics_returns_instruments_locally(self):
        """Test that metrics returns instruments when running locally (uses ConsoleMetricExporter)."""
        with patch.object(observability, "IS_CLOUD_RUN", False):
            observability._meter_instruments = None
            result = observability.setup_metrics()
            # Should return instruments (uses ConsoleMetricExporter locally)
            assert result is not None
            assert "tool_calls" in result
            assert "tool_duration" in result
            assert "tool_errors" in result

    def test_setup_metrics_returns_none_when_disabled(self):
        """Test that metrics returns None when observability is disabled."""
        with patch.object(observability, "OBSERVABILITY_ENABLED", False):
            observability._meter_instruments = None
            result = observability.setup_metrics()
            assert result is None

    def test_setup_metrics_caches_instruments(self):
        """Test that metrics instruments are cached."""
        mock_instruments = {"tool_calls": MagicMock()}
        observability._meter_instruments = mock_instruments

        result = observability.setup_metrics()
        assert result is mock_instruments


class TestInitializeObservability:
    """Tests for initialize_observability function."""

    def setup_method(self):
        """Reset globals before each test."""
        observability._logger = None
        observability._tracer = None
        observability._tracer_provider = None
        observability._meter_instruments = None
        observability._meter_provider = None
        observability._initialized = False
        observability._shutting_down = False

    def teardown_method(self):
        """Clean up after each test."""
        observability._logger = None
        observability._tracer = None
        observability._tracer_provider = None
        observability._meter_instruments = None
        observability._meter_provider = None
        observability._initialized = False
        observability._shutting_down = False

    def test_initialize_observability_calls_all_setup_functions(self):
        """Test that initialize calls all setup functions."""
        with (
            patch.object(observability, "setup_logging") as mock_logging,
            patch.object(observability, "setup_tracing") as mock_tracing,
            patch.object(observability, "setup_metrics") as mock_metrics,
        ):
            mock_logging.return_value = MagicMock()
            observability.initialize_observability()

            mock_logging.assert_called_once()
            mock_tracing.assert_called_once()
            mock_metrics.assert_called_once()

    def test_initialize_observability_only_runs_once(self):
        """Test that initialize only runs once when called multiple times."""
        with (
            patch.object(observability, "setup_logging") as mock_logging,
            patch.object(observability, "setup_tracing") as mock_tracing,
            patch.object(observability, "setup_metrics") as mock_metrics,
        ):
            mock_logging.return_value = MagicMock()

            observability.initialize_observability()
            observability.initialize_observability()
            observability.initialize_observability()

            # Each setup should only be called once
            assert mock_logging.call_count == 1
            assert mock_tracing.call_count == 1
            assert mock_metrics.call_count == 1


class TestTraceMcpTool:
    """Tests for trace_mcp_tool context manager."""

    def setup_method(self):
        """Reset globals before each test."""
        observability._logger = None
        observability._tracer = None
        observability._tracer_provider = None
        observability._meter_instruments = None
        observability._meter_provider = None
        observability._initialized = False
        observability._shutting_down = False

    def teardown_method(self):
        """Clean up after each test."""
        observability._logger = None
        observability._tracer = None
        observability._tracer_provider = None
        observability._meter_instruments = None
        observability._meter_provider = None
        observability._initialized = False
        observability._shutting_down = False

    def test_trace_mcp_tool_works_locally(self):
        """Test context manager works locally with tracing enabled."""
        with patch.object(observability, "IS_CLOUD_RUN", False):
            with observability.trace_mcp_tool("test_tool", param1="value1") as span:
                # Should have a span when tracing is enabled
                assert span is not None

    def test_trace_mcp_tool_yields_none_without_tracer(self):
        """Test that span is None without tracer."""
        with patch.object(observability, "setup_tracing", return_value=None):
            with observability.trace_mcp_tool("list_documents") as span:
                assert span is None

    def test_trace_mcp_tool_logs_execution(self):
        """Test that tool execution is logged."""
        with patch.object(observability, "setup_tracing", return_value=None):
            mock_logger = MagicMock()
            with patch.object(observability, "setup_logging", return_value=mock_logger):
                with observability.trace_mcp_tool("test_tool"):
                    pass

            # Logger should have been called
            mock_logger.info.assert_called()

    def test_trace_mcp_tool_handles_exception(self):
        """Test that exceptions are properly tracked."""
        with patch.object(observability, "setup_tracing", return_value=None):
            with pytest.raises(ValueError, match="Test error"):
                with observability.trace_mcp_tool("failing_tool"):
                    raise ValueError("Test error")

    def test_trace_mcp_tool_logs_errors_as_warning(self):
        """Test that errors are logged as warnings."""
        mock_logger = MagicMock()
        with (
            patch.object(observability, "setup_tracing", return_value=None),
            patch.object(observability, "setup_logging", return_value=mock_logger),
            patch.object(observability, "setup_metrics", return_value=None),
        ):
            with pytest.raises(ValueError):
                with observability.trace_mcp_tool("failing_tool"):
                    raise ValueError("Test error")

            # Should log as warning for errors
            mock_logger.warning.assert_called()


class TestLogToolExecution:
    """Tests for _log_tool_execution helper function."""

    def test_log_tool_execution_success(self):
        """Test logging successful tool execution."""
        mock_logger = MagicMock()
        observability._log_tool_execution(mock_logger, "test_tool", 0.5, "success", None)

        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args
        assert "test_tool" in call_args[0][0]
        assert "success" in call_args[0][0]

    def test_log_tool_execution_error(self):
        """Test logging failed tool execution."""
        mock_logger = MagicMock()
        observability._log_tool_execution(mock_logger, "test_tool", 0.5, "error", "ValueError")

        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args
        assert "test_tool" in call_args[0][0]
        assert "error" in call_args[0][0]


class TestRecordMetrics:
    """Tests for _record_metrics helper function."""

    def test_record_metrics_with_none_instruments(self):
        """Test that None instruments are handled gracefully."""
        # Should not raise
        observability._record_metrics("test_tool", 0.5, "success", None, None)

    def test_record_metrics_with_instruments(self):
        """Test that metrics are recorded when instruments exist."""
        mock_counter = MagicMock()
        mock_histogram = MagicMock()
        mock_error_counter = MagicMock()

        instruments = {
            "tool_calls": mock_counter,
            "tool_duration": mock_histogram,
            "tool_errors": mock_error_counter,
        }

        observability._record_metrics("test_tool", 0.5, "success", None, instruments)

        mock_counter.add.assert_called_once()
        mock_histogram.record.assert_called_once()
        mock_error_counter.add.assert_not_called()

    def test_record_metrics_records_errors(self):
        """Test that error metrics are recorded on failure."""
        mock_counter = MagicMock()
        mock_histogram = MagicMock()
        mock_error_counter = MagicMock()

        instruments = {
            "tool_calls": mock_counter,
            "tool_duration": mock_histogram,
            "tool_errors": mock_error_counter,
        }

        observability._record_metrics("test_tool", 0.5, "error", "ValueError", instruments)

        mock_error_counter.add.assert_called_once()


class TestLogMcpCall:
    """Tests for log_mcp_call decorator."""

    def setup_method(self):
        """Reset globals before each test."""
        observability._logger = None
        observability._tracer = None
        observability._tracer_provider = None
        observability._meter_instruments = None
        observability._meter_provider = None
        observability._initialized = False
        observability._shutting_down = False

    def teardown_method(self):
        """Clean up after each test."""
        observability._logger = None
        observability._tracer = None
        observability._tracer_provider = None
        observability._meter_instruments = None
        observability._meter_provider = None
        observability._initialized = False
        observability._shutting_down = False

    def test_log_mcp_call_sync_function(self):
        """Test decorator with synchronous function."""

        @observability.log_mcp_call
        def sync_tool(x: int) -> int:
            return x * 2

        result = sync_tool(5)
        assert result == 10

    @pytest.mark.asyncio
    async def test_log_mcp_call_async_function(self):
        """Test decorator with asynchronous function."""

        @observability.log_mcp_call
        async def async_tool(x: int) -> int:
            return x * 2

        result = await async_tool(5)
        assert result == 10


class TestGetLogger:
    """Tests for get_logger function."""

    def setup_method(self):
        """Reset global logger before each test."""
        observability._logger = None

    def teardown_method(self):
        """Clean up after each test."""
        observability._logger = None

    def test_get_logger_returns_configured_logger(self):
        """Test that get_logger returns the configured logger."""
        logger = observability.get_logger()
        assert isinstance(logger, logging.Logger)
        assert logger.name == "document_mcp"

    def test_get_logger_same_as_setup_logging(self):
        """Test that get_logger returns same logger as setup_logging."""
        logger1 = observability.get_logger()
        logger2 = observability.setup_logging()
        assert logger1 is logger2


class TestObservabilityStatus:
    """Tests for observability status functions."""

    def setup_method(self):
        """Reset globals before each test."""
        observability._logger = None
        observability._tracer = None
        observability._tracer_provider = None
        observability._meter_instruments = None
        observability._meter_provider = None
        observability._initialized = False
        observability._shutting_down = False

    def teardown_method(self):
        """Clean up after each test."""
        observability._logger = None
        observability._tracer = None
        observability._tracer_provider = None
        observability._meter_instruments = None
        observability._meter_provider = None
        observability._initialized = False
        observability._shutting_down = False

    def test_is_observability_enabled_returns_bool(self):
        """Test that is_observability_enabled returns a boolean."""
        result = observability.is_observability_enabled()
        assert isinstance(result, bool)

    def test_is_observability_enabled_respects_env_var(self):
        """Test that is_observability_enabled respects environment variable."""
        with patch.object(observability, "OBSERVABILITY_ENABLED", True):
            assert observability.is_observability_enabled() is True

        with patch.object(observability, "OBSERVABILITY_ENABLED", False):
            assert observability.is_observability_enabled() is False

    def test_get_observability_status_returns_dict(self):
        """Test that get_observability_status returns a dictionary."""
        status = observability.get_observability_status()
        assert isinstance(status, dict)
        assert "enabled" in status
        assert "initialized" in status
        assert "environment" in status
        assert "tracing" in status
        assert "metrics" in status
        assert "logging" in status

    def test_get_observability_status_reflects_state(self):
        """Test that get_observability_status reflects actual state."""
        # Before initialization
        status = observability.get_observability_status()
        assert status["initialized"] is False
        assert status["logging"] is False

        # After setting up logging
        observability.setup_logging()
        status = observability.get_observability_status()
        assert status["logging"] is True

    def test_get_observability_status_shows_environment(self):
        """Test that get_observability_status shows correct environment."""
        with patch.object(observability, "IS_CLOUD_RUN", False):
            status = observability.get_observability_status()
            assert status["environment"] == "local"

        with patch.object(observability, "IS_CLOUD_RUN", True):
            status = observability.get_observability_status()
            assert status["environment"] == "cloud_run"
