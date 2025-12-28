"""Document MCP Metrics Configuration.

Local-only metrics collection using OpenTelemetry.
Stores logs, traces, and metrics locally for development.
For cloud deployment, configure external exporters via environment variables.
"""
from __future__ import annotations


import json
import os
import socket
import time
from functools import wraps
from typing import Any

from opentelemetry import metrics
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource
from prometheus_client import CONTENT_TYPE_LATEST
from prometheus_client import generate_latest

# =============================================================================
# SERVICE CONFIGURATION
# =============================================================================

SERVICE_NAME = os.getenv("OTEL_SERVICE_NAME", "document-mcp")
SERVICE_VERSION = os.getenv("OTEL_SERVICE_VERSION", "1.0.0")
DEPLOYMENT_ENVIRONMENT = os.getenv("DEPLOYMENT_ENVIRONMENT", "local")


def is_test_environment() -> bool:
    """Detect if running in test environment."""
    return (
        "PYTEST_CURRENT_TEST" in os.environ
        or "DOCUMENT_ROOT_DIR" in os.environ
        or "CI" in os.environ
        or "GITHUB_ACTIONS" in os.environ
    )


# Disable metrics in test/CI environments by default
default_metrics_enabled = "false" if is_test_environment() else "true"
METRICS_ENABLED = os.getenv("MCP_METRICS_ENABLED", default_metrics_enabled).lower() == "true"

# Debug mode
DEBUG_METRICS = os.getenv("MCP_METRICS_DEBUG", "false").lower() == "true"

# Metrics instances
meter = None
tool_calls_counter = None
prometheus_reader = None

# Global state
_active_operations = {}
_metrics_initialized = False


def get_resource() -> Resource:
    """Create OpenTelemetry resource with service information."""
    return Resource.create(
        {
            "service.name": SERVICE_NAME,
            "service.version": SERVICE_VERSION,
            "deployment.environment": DEPLOYMENT_ENVIRONMENT,
            "host.name": socket.gethostname(),
            "telemetry.sdk.name": "opentelemetry",
            "telemetry.sdk.language": "python",
        }
    )


def initialize_metrics():
    """Initialize local metrics collection with Prometheus endpoint."""
    global meter, tool_calls_counter, prometheus_reader

    if not METRICS_ENABLED:
        if DEBUG_METRICS:
            print("[METRICS] Telemetry disabled")
        return

    try:
        resource = get_resource()

        # Create Prometheus reader for local /metrics endpoint
        prometheus_reader = PrometheusMetricReader()

        # Create meter provider
        meter_provider = MeterProvider(
            resource=resource,
            metric_readers=[prometheus_reader],
        )
        metrics.set_meter_provider(meter_provider)
        meter = metrics.get_meter(__name__)

        # Create counter metric
        tool_calls_counter = meter.create_counter(
            name="mcp_tool_calls_total",
            description="Total number of MCP tool calls",
            unit="1",
        )

        if DEBUG_METRICS:
            print(f"[METRICS] Initialized: {SERVICE_NAME} v{SERVICE_VERSION}")
            print(f"[METRICS] Environment: {DEPLOYMENT_ENVIRONMENT}")
            print("[METRICS] Local endpoint: http://localhost:8000/metrics")

    except Exception as e:
        if DEBUG_METRICS:
            print(f"[METRICS] Init failed: {e}")


def is_metrics_enabled() -> bool:
    """Check if metrics collection is enabled."""
    return METRICS_ENABLED and meter is not None


def record_tool_call_start(tool_name: str, args: tuple, kwargs: dict) -> float | None:
    """Record start of tool call, return start time."""
    if not is_metrics_enabled():
        return None

    start_time = time.time()
    operation_id = f"{tool_name}_{start_time}"
    _active_operations[operation_id] = start_time
    return start_time


def record_tool_call_success(tool_name: str, start_time: float | None, result_size: int = 0):
    """Record successful tool call."""
    if not is_metrics_enabled():
        return

    try:
        if tool_calls_counter:
            tool_calls_counter.add(
                1, {"tool_name": tool_name, "status": "success", "environment": DEPLOYMENT_ENVIRONMENT}
            )

        if start_time:
            operation_id = f"{tool_name}_{start_time}"
            _active_operations.pop(operation_id, None)
    except Exception as e:
        if DEBUG_METRICS:
            print(f"[METRICS] Record success failed: {e}")


def record_tool_call_error(tool_name: str, start_time: float | None, error: Exception):
    """Record failed tool call."""
    if not is_metrics_enabled():
        return

    try:
        if tool_calls_counter:
            tool_calls_counter.add(
                1, {"tool_name": tool_name, "status": "error", "environment": DEPLOYMENT_ENVIRONMENT}
            )

        if start_time:
            operation_id = f"{tool_name}_{start_time}"
            _active_operations.pop(operation_id, None)
    except Exception as e:
        if DEBUG_METRICS:
            print(f"[METRICS] Record error failed: {e}")


def get_metrics_export() -> tuple[str, str]:
    """Export metrics in Prometheus format."""
    if not is_metrics_enabled() or not prometheus_reader:
        return "# Metrics not available\n", "text/plain"

    try:
        metrics_data = generate_latest()
        return metrics_data.decode("utf-8"), CONTENT_TYPE_LATEST
    except Exception as e:
        return f"# Error: {e}\n", "text/plain"


def get_metrics_summary() -> dict[str, Any]:
    """Get metrics summary for debugging."""
    if not is_metrics_enabled():
        return {"status": "disabled"}

    return {
        "status": "active",
        "service_name": SERVICE_NAME,
        "service_version": SERVICE_VERSION,
        "environment": DEPLOYMENT_ENVIRONMENT,
        "active_operations": len(_active_operations),
        "prometheus_enabled": prometheus_reader is not None,
    }


def instrument_tool(func):
    """Decorator to instrument MCP tools with metrics."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        tool_name = func.__name__
        start_time = record_tool_call_start(tool_name, args, kwargs)

        try:
            result = func(*args, **kwargs)
            result_size = len(str(result)) if result else 0
            record_tool_call_success(tool_name, start_time, result_size)
            return result
        except Exception as e:
            record_tool_call_error(tool_name, start_time, e)
            raise

    return wrapper


def ensure_metrics_initialized():
    """Initialize metrics when server starts."""
    global _metrics_initialized
    if _metrics_initialized:
        return

    if METRICS_ENABLED and not is_test_environment():
        initialize_metrics()
    _metrics_initialized = True


def shutdown_metrics():
    """Shutdown metrics collection."""
    global prometheus_reader
    if prometheus_reader:
        try:
            prometheus_reader.shutdown()
        except Exception:
            pass


# Backward compatibility exports
def calculate_argument_size(args: tuple, kwargs: dict) -> int:
    """Calculate size of arguments in bytes."""
    try:
        return len(json.dumps(args, default=str).encode()) + len(json.dumps(kwargs, default=str).encode())
    except Exception:
        return len(repr(args).encode()) + len(repr(kwargs).encode())


def flush_metrics():
    """Flush pending metrics (no-op for local-only)."""
    pass
