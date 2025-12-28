"""Observability for Document MCP - works on Cloud Run and locally.

This module provides comprehensive observability:
- Cloud Run: Uses Cloud Logging, Cloud Trace, Cloud Monitoring
- Local: Uses console exporters for traces/metrics, standard logging

No hardcoded credentials - relies on GCP service account for auth on Cloud Run.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import sys
import time
from collections.abc import Callable
from contextlib import contextmanager
from functools import wraps
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from opentelemetry.trace import Tracer

# Environment detection
IS_CLOUD_RUN = bool(os.environ.get("K_SERVICE"))
GCP_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT")
OBSERVABILITY_ENABLED = os.environ.get("MCP_OBSERVABILITY_ENABLED", "true").lower() == "true"

# Global instances (initialized lazily)
_tracer: Tracer | None = None
_meter_instruments: dict[str, Any] | None = None
_logger: logging.Logger | None = None
_initialized = False
_tracer_provider: Any = None  # TracerProvider for shutdown
_meter_provider: Any = None  # MeterProvider for shutdown
_shutdown_registered = False
_shutting_down = False  # Flag to prevent exports during shutdown


class SafeConsoleSpanExporter:
    """Wrapper for ConsoleSpanExporter that handles I/O errors during shutdown."""

    def __init__(self):
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter

        self._exporter = ConsoleSpanExporter()

    def export(self, spans):
        if _shutting_down:
            from opentelemetry.sdk.trace.export import SpanExportResult

            return SpanExportResult.SUCCESS
        try:
            return self._exporter.export(spans)
        except (ValueError, OSError):
            # Suppress I/O errors during shutdown (stdout may be closed)
            from opentelemetry.sdk.trace.export import SpanExportResult

            return SpanExportResult.SUCCESS

    def shutdown(self):
        try:
            self._exporter.shutdown()
        except (ValueError, OSError):
            pass

    def force_flush(self, timeout_millis=30000):
        try:
            return self._exporter.force_flush(timeout_millis)
        except (ValueError, OSError):
            return True


class SafeConsoleMetricExporter:
    """Wrapper for ConsoleMetricExporter that handles I/O errors during shutdown."""

    def __init__(self):
        from opentelemetry.sdk.metrics.export import ConsoleMetricExporter

        self._exporter = ConsoleMetricExporter()

    @property
    def _preferred_temporality(self):
        return self._exporter._preferred_temporality

    @property
    def _preferred_aggregation(self):
        return self._exporter._preferred_aggregation

    def export(self, metrics_data, timeout_millis=10000, **kwargs):
        if _shutting_down:
            from opentelemetry.sdk.metrics.export import MetricExportResult

            return MetricExportResult.SUCCESS
        try:
            return self._exporter.export(metrics_data, timeout_millis, **kwargs)
        except (ValueError, OSError):
            # Suppress I/O errors during shutdown (stdout may be closed)
            from opentelemetry.sdk.metrics.export import MetricExportResult

            return MetricExportResult.SUCCESS

    def shutdown(self, timeout_millis=30000, **kwargs):
        try:
            self._exporter.shutdown(timeout_millis, **kwargs)
        except (ValueError, OSError):
            pass

    def force_flush(self, timeout_millis=10000):
        try:
            return self._exporter.force_flush(timeout_millis)
        except (ValueError, OSError):
            return True


class CloudRunJSONFormatter(logging.Formatter):
    """JSON formatter compatible with Cloud Logging structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON for Cloud Logging."""
        log_entry: dict[str, Any] = {
            "severity": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }

        # Add timestamp
        log_entry["timestamp"] = self.formatTime(record, "%Y-%m-%dT%H:%M:%S.%fZ")

        # Add trace context if available
        trace_id = getattr(record, "trace_id", None)
        if trace_id and GCP_PROJECT:
            log_entry["logging.googleapis.com/trace"] = f"projects/{GCP_PROJECT}/traces/{trace_id}"

        span_id = getattr(record, "span_id", None)
        if span_id:
            log_entry["logging.googleapis.com/spanId"] = span_id

        # Add extra fields
        extra_fields = getattr(record, "extra_fields", None)
        if extra_fields and isinstance(extra_fields, dict):
            log_entry.update(extra_fields)

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry)


def setup_logging() -> logging.Logger:
    """Configure logging appropriate for the environment."""
    global _logger

    if _logger is not None:
        return _logger

    logger = logging.getLogger("document_mcp")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    handler = logging.StreamHandler(sys.stdout)

    if IS_CLOUD_RUN:
        # JSON format for Cloud Logging
        handler.setFormatter(CloudRunJSONFormatter())
    else:
        # Human-readable format for local development
        handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

    logger.addHandler(handler)
    _logger = logger

    return logger


def _shutdown_observability() -> None:
    """Gracefully shutdown OpenTelemetry providers to avoid I/O errors at process exit."""
    global _tracer_provider, _meter_provider, _shutting_down

    # Prevent multiple shutdown calls
    if _shutting_down:
        return

    # Set flag to prevent further exports and re-entry
    _shutting_down = True

    # Suppress stderr during shutdown to avoid "shutdown can only be called once" messages
    # from OpenTelemetry's internal state tracking
    import io

    try:
        if _meter_provider is not None:
            _meter_provider.force_flush()
            # Temporarily redirect stderr to suppress OpenTelemetry warnings
            original_stderr = sys.stderr
            sys.stderr = io.StringIO()
            try:
                _meter_provider.shutdown()
            finally:
                sys.stderr = original_stderr
            _meter_provider = None
    except Exception:
        pass  # Suppress errors during shutdown

    try:
        if _tracer_provider is not None:
            _tracer_provider.force_flush()
            # Temporarily redirect stderr to suppress OpenTelemetry warnings
            original_stderr = sys.stderr
            sys.stderr = io.StringIO()
            try:
                _tracer_provider.shutdown()
            finally:
                sys.stderr = original_stderr
            _tracer_provider = None
    except Exception:
        pass  # Suppress errors during shutdown


def _register_shutdown_handler() -> None:
    """Register the shutdown handler once."""
    global _shutdown_registered
    if not _shutdown_registered:
        atexit.register(_shutdown_observability)
        _shutdown_registered = True


def setup_tracing() -> Tracer | None:
    """Configure OpenTelemetry tracing - works on Cloud Run and locally."""
    global _tracer, _tracer_provider

    if _tracer is not None:
        return _tracer

    if not OBSERVABILITY_ENABLED:
        return None

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import SERVICE_NAME
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        resource = Resource.create(
            {
                SERVICE_NAME: "document-mcp",
                "service.version": os.environ.get("K_REVISION", "local"),
                "cloud.provider": "gcp" if IS_CLOUD_RUN else "local",
                "cloud.platform": "gcp_cloud_run" if IS_CLOUD_RUN else "local",
            }
        )

        provider = TracerProvider(resource=resource)

        if IS_CLOUD_RUN:
            # Cloud Run: Use Cloud Trace exporter
            try:
                from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter

                provider.add_span_processor(SimpleSpanProcessor(CloudTraceSpanExporter()))
            except ImportError:
                # Fall back to safe console exporter if GCP packages not installed
                provider.add_span_processor(BatchSpanProcessor(SafeConsoleSpanExporter()))
        else:
            # Local: Use safe console exporter (handles I/O errors during shutdown)
            provider.add_span_processor(BatchSpanProcessor(SafeConsoleSpanExporter()))

        trace.set_tracer_provider(provider)
        _tracer_provider = provider  # Store for shutdown
        _register_shutdown_handler()
        _tracer = trace.get_tracer("document_mcp")
        return _tracer

    except ImportError:
        # OpenTelemetry not installed
        logger = setup_logging()
        logger.debug("OpenTelemetry not installed - tracing disabled")
        return None
    except Exception as e:
        logger = setup_logging()
        logger.warning(f"Failed to initialize tracing: {e}")
        return None


def setup_metrics() -> dict[str, Any] | None:
    """Configure OpenTelemetry metrics - works on Cloud Run and locally."""
    global _meter_instruments, _meter_provider

    if _meter_instruments is not None:
        return _meter_instruments

    if not OBSERVABILITY_ENABLED:
        return None

    try:
        from opentelemetry import metrics
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

        if IS_CLOUD_RUN:
            # Cloud Run: Use Cloud Monitoring exporter
            try:
                from opentelemetry.exporter.cloud_monitoring import CloudMonitoringMetricsExporter

                exporter = CloudMonitoringMetricsExporter()
            except ImportError:
                # Fall back to safe console exporter if GCP packages not installed
                exporter = SafeConsoleMetricExporter()
        else:
            # Local: Use safe console exporter (handles I/O errors during shutdown)
            exporter = SafeConsoleMetricExporter()

        # Export every 30 seconds locally, 60 seconds on Cloud Run
        export_interval = 60000 if IS_CLOUD_RUN else 30000
        reader = PeriodicExportingMetricReader(exporter, export_interval_millis=export_interval)
        provider = MeterProvider(metric_readers=[reader])
        metrics.set_meter_provider(provider)
        _meter_provider = provider  # Store for shutdown
        _register_shutdown_handler()

        meter = metrics.get_meter("document_mcp")

        _meter_instruments = {
            "tool_calls": meter.create_counter(
                "mcp_tool_calls_total",
                unit="1",
                description="Total MCP tool calls",
            ),
            "tool_duration": meter.create_histogram(
                "mcp_tool_duration_seconds",
                unit="s",
                description="MCP tool execution duration",
            ),
            "tool_errors": meter.create_counter(
                "mcp_tool_errors_total",
                unit="1",
                description="Total MCP tool errors",
            ),
        }
        return _meter_instruments

    except ImportError:
        # OpenTelemetry not installed
        logger = setup_logging()
        logger.debug("OpenTelemetry metrics not installed - metrics disabled")
        return None
    except Exception as e:
        logger = setup_logging()
        logger.warning(f"Failed to initialize metrics: {e}")
        return None


def initialize_observability() -> None:
    """Initialize all observability components."""
    global _initialized

    if _initialized:
        return

    logger = setup_logging()
    tracer = setup_tracing()
    metrics_instruments = setup_metrics()

    _initialized = True

    # Log initialization status
    env = "Cloud Run" if IS_CLOUD_RUN else "Local"
    logger.info(
        f"Observability initialized ({env})",
        extra={
            "extra_fields": {
                "environment": env,
                "tracing_enabled": tracer is not None,
                "metrics_enabled": metrics_instruments is not None,
            }
        },
    )


@contextmanager
def trace_mcp_tool(tool_name: str, **attributes: Any):
    """Context manager for tracing MCP tool execution.

    Usage:
        with trace_mcp_tool("list_documents", include_chapters=True):
            # tool implementation
    """
    tracer = setup_tracing()
    metrics_instruments = setup_metrics()
    logger = setup_logging()

    start_time = time.perf_counter()
    status = "success"
    error_type = None

    if tracer:
        with tracer.start_as_current_span(f"mcp.tool.{tool_name}") as span:
            span.set_attribute("mcp.tool.name", tool_name)
            for key, value in attributes.items():
                span.set_attribute(f"mcp.tool.{key}", str(value))

            try:
                yield span
            except Exception as e:
                status = "error"
                error_type = type(e).__name__
                span.set_attribute("mcp.tool.status", "error")
                span.set_attribute("mcp.tool.error_type", error_type)
                span.record_exception(e)
                raise
            finally:
                duration = time.perf_counter() - start_time
                span.set_attribute("mcp.tool.duration_seconds", duration)
                span.set_attribute("mcp.tool.status", status)

                _record_metrics(tool_name, duration, status, error_type, metrics_instruments)
                _log_tool_execution(logger, tool_name, duration, status, error_type)
    else:
        # No tracing - just record metrics and log
        try:
            yield None
        except Exception as e:
            status = "error"
            error_type = type(e).__name__
            raise
        finally:
            duration = time.perf_counter() - start_time
            _record_metrics(tool_name, duration, status, error_type, metrics_instruments)
            _log_tool_execution(logger, tool_name, duration, status, error_type)


def _log_tool_execution(
    logger: logging.Logger,
    tool_name: str,
    duration: float,
    status: str,
    error_type: str | None,
) -> None:
    """Log tool execution details."""
    extra_fields = {
        "tool_name": tool_name,
        "duration_seconds": round(duration, 3),
        "status": status,
    }
    if error_type:
        extra_fields["error_type"] = error_type

    log_msg = f"MCP tool: {tool_name} ({status}, {duration:.3f}s)"
    if status == "error":
        logger.warning(log_msg, extra={"extra_fields": extra_fields})
    else:
        logger.info(log_msg, extra={"extra_fields": extra_fields})


def _record_metrics(
    tool_name: str,
    duration: float,
    status: str,
    error_type: str | None,
    instruments: dict[str, Any] | None,
) -> None:
    """Record metrics for tool execution."""
    if not instruments:
        return

    labels = {"tool_name": tool_name, "status": status}

    if "tool_calls" in instruments:
        instruments["tool_calls"].add(1, labels)

    if "tool_duration" in instruments:
        instruments["tool_duration"].record(duration, {"tool_name": tool_name})

    if status == "error" and "tool_errors" in instruments:
        instruments["tool_errors"].add(1, {"tool_name": tool_name, "error_type": error_type or "unknown"})


def log_mcp_call(func: Callable) -> Callable:
    """Decorator for logging and tracing MCP tool calls.

    Usage:
        @log_mcp_call
        async def list_documents(...):
            ...
    """

    @wraps(func)
    async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
        tool_name = func.__name__
        with trace_mcp_tool(tool_name):
            return await func(*args, **kwargs)

    @wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        tool_name = func.__name__
        with trace_mcp_tool(tool_name):
            return func(*args, **kwargs)

    import asyncio

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper


def get_logger() -> logging.Logger:
    """Get the configured logger instance."""
    return setup_logging()


def is_observability_enabled() -> bool:
    """Check if observability is enabled."""
    return OBSERVABILITY_ENABLED


def get_observability_status() -> dict[str, Any]:
    """Get current observability status for debugging."""
    return {
        "enabled": OBSERVABILITY_ENABLED,
        "initialized": _initialized,
        "environment": "cloud_run" if IS_CLOUD_RUN else "local",
        "tracing": _tracer is not None,
        "metrics": _meter_instruments is not None,
        "logging": _logger is not None,
    }
