import os
import time
import json
import socket
from typing import Optional, Dict, Any
from functools import wraps

# OpenTelemetry imports
from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
    
# Prometheus client for endpoint
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

# Configuration from environment variables
METRICS_ENABLED = os.getenv("MCP_METRICS_ENABLED", "true").lower() == "true"
OTEL_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")
OTEL_HEADERS = os.getenv("OTEL_EXPORTER_OTLP_HEADERS", "")
SERVICE_NAME = os.getenv("OTEL_SERVICE_NAME", "document-mcp-server")
SERVICE_VERSION = os.getenv("OTEL_SERVICE_VERSION", "0.0.1")
DEPLOYMENT_ENVIRONMENT = os.getenv("DEPLOYMENT_ENVIRONMENT", "development")

# Metrics instances
meter = None
tool_calls_counter = None
tool_duration_histogram = None
tool_errors_counter = None
tool_argument_sizes_histogram = None
concurrent_operations_gauge = None
server_info_counter = None
prometheus_reader = None

# Global state for tracking
_active_operations = {}

def get_resource() -> 'Resource':
    """Create OpenTelemetry resource with service information."""
    return Resource.create({
        "service.name": SERVICE_NAME,
        "service.version": SERVICE_VERSION,
        "deployment.environment": DEPLOYMENT_ENVIRONMENT,
        "host.name": socket.gethostname(),
        "telemetry.sdk.name": "opentelemetry",
        "telemetry.sdk.language": "python"
    })

def initialize_metrics():
    """Initialize OpenTelemetry metrics with both local Prometheus and remote OTLP export."""
    global meter, tool_calls_counter, tool_duration_histogram, tool_errors_counter
    global tool_argument_sizes_histogram, concurrent_operations_gauge, server_info_counter
    global prometheus_reader
    
    if not METRICS_ENABLED:
        print("OpenTelemetry metrics disabled")
        return
    
    try:
        # Create resource
        resource = get_resource()
        
        # Create metric readers
        metric_readers = []
        
        # Always add Prometheus reader for local /metrics endpoint
        prometheus_reader = PrometheusMetricReader()
        metric_readers.append(prometheus_reader)
        
        # Add OTLP exporter if endpoint is configured
        if OTEL_ENDPOINT:
            headers = {}
            if OTEL_HEADERS:
                # Parse headers in format "key1=value1,key2=value2"
                for header in OTEL_HEADERS.split(","):
                    if "=" in header:
                        key, value = header.strip().split("=", 1)
                        headers[key] = value
            
            otlp_exporter = OTLPMetricExporter(
                endpoint=OTEL_ENDPOINT,
                headers=headers
            )
            
            # Export every 30 seconds
            otlp_reader = PeriodicExportingMetricReader(
                exporter=otlp_exporter,
                export_interval_millis=30000
            )
            metric_readers.append(otlp_reader)
            print(f"OTLP metrics exporter configured for {OTEL_ENDPOINT}")
        
        # Create meter provider
        meter_provider = MeterProvider(
            resource=resource,
            metric_readers=metric_readers
        )
        
        # Set the global meter provider
        metrics.set_meter_provider(meter_provider)
        
        # Create meter
        meter = metrics.get_meter(__name__)
        
        # Initialize metrics instruments
        tool_calls_counter = meter.create_counter(
            name="mcp_tool_calls_total",
            description="Total number of MCP tool calls",
            unit="1"
        )
        
        tool_duration_histogram = meter.create_histogram(
            name="mcp_tool_duration_seconds",
            description="MCP tool execution time in seconds",
            unit="s"
        )
        
        tool_errors_counter = meter.create_counter(
            name="mcp_tool_errors_total", 
            description="Total number of MCP tool errors",
            unit="1"
        )
        
        tool_argument_sizes_histogram = meter.create_histogram(
            name="mcp_tool_argument_sizes_bytes",
            description="Size of arguments passed to MCP tools",
            unit="bytes"
        )
        
        concurrent_operations_gauge = meter.create_up_down_counter(
            name="mcp_concurrent_operations",
            description="Number of concurrent MCP tool operations",
            unit="1"
        )
        
        server_info_counter = meter.create_counter(
            name="mcp_server_info",
            description="Information about the MCP server",
            unit="1"
        )
        
        # Record server startup
        server_info_counter.add(1, {
            "version": SERVICE_VERSION,
            "server_type": "document_mcp",
            "environment": DEPLOYMENT_ENVIRONMENT,
            "hostname": socket.gethostname()
        })
        
        print(f"OpenTelemetry metrics initialized for service: {SERVICE_NAME}")
        
        # Initialize FastAPI instrumentation if available
        try:
            FastAPIInstrumentor().instrument()
        except Exception as e:
            print(f"Warning: Could not instrument FastAPI: {e}")
        
        # Initialize requests instrumentation
        try:
            RequestsInstrumentor().instrument()
        except Exception as e:
            print(f"Warning: Could not instrument requests: {e}")
            
    except Exception as e:
        print(f"Error: Failed to initialize OpenTelemetry metrics: {e}")

def is_metrics_enabled() -> bool:
    """Check if metrics collection is enabled and available."""
    return METRICS_ENABLED and meter is not None

def calculate_argument_size(args: tuple, kwargs: dict) -> int:
    """Calculate size of arguments in bytes."""
    try:
        args_str = json.dumps(args, default=str)
        kwargs_str = json.dumps(kwargs, default=str)
        return len(args_str.encode('utf-8')) + len(kwargs_str.encode('utf-8'))
    except Exception:
        # Fallback to repr if JSON serialization fails
        args_str = repr(args)
        kwargs_str = repr(kwargs)
        return len(args_str.encode('utf-8')) + len(kwargs_str.encode('utf-8'))

def record_tool_call_start(tool_name: str, args: tuple, kwargs: dict) -> Optional[float]:
    """Record the start of a tool call and return start time for duration calculation."""
    if not is_metrics_enabled():
        return None
    
    start_time = time.time()
    operation_id = f"{tool_name}_{start_time}"
    
    try:
        # Track concurrent operations
        _active_operations[operation_id] = start_time
        if concurrent_operations_gauge:
            concurrent_operations_gauge.add(1, {"tool_name": tool_name})
        
        # Record argument size
        if tool_argument_sizes_histogram:
            arg_size = calculate_argument_size(args, kwargs)
            tool_argument_sizes_histogram.record(arg_size, {"tool_name": tool_name})
            
    except Exception as e:
        print(f"Warning: Failed to record tool call start metrics: {e}")
    
    return start_time

def record_tool_call_success(tool_name: str, start_time: Optional[float], result_size: int = 0):
    """Record a successful tool call completion."""
    if not is_metrics_enabled() or start_time is None:
        return
    
    operation_id = f"{tool_name}_{start_time}"
    
    try:
        # Record success
        if tool_calls_counter:
            tool_calls_counter.add(1, {
                "tool_name": tool_name, 
                "status": "success",
                "environment": DEPLOYMENT_ENVIRONMENT
            })
        
        # Record duration
        if tool_duration_histogram:
            duration = time.time() - start_time
            tool_duration_histogram.record(duration, {
                "tool_name": tool_name,
                "status": "success"
            })
        
        # Update concurrent operations
        if concurrent_operations_gauge and operation_id in _active_operations:
            concurrent_operations_gauge.add(-1, {"tool_name": tool_name})
            del _active_operations[operation_id]
            
    except Exception as e:
        print(f"Warning: Failed to record tool call success metrics: {e}")

def record_tool_call_error(tool_name: str, start_time: Optional[float], error: Exception):
    """Record a failed tool call."""
    if not is_metrics_enabled():
        return
    
    operation_id = f"{tool_name}_{start_time}" if start_time else ""
    
    try:
        # Record error
        if tool_calls_counter:
            tool_calls_counter.add(1, {
                "tool_name": tool_name, 
                "status": "error",
                "environment": DEPLOYMENT_ENVIRONMENT
            })
        
        if tool_errors_counter:
            error_type = type(error).__name__
            tool_errors_counter.add(1, {
                "tool_name": tool_name, 
                "error_type": error_type,
                "environment": DEPLOYMENT_ENVIRONMENT
            })
        
        # Record duration even for errors if we have start time
        if tool_duration_histogram and start_time is not None:
            duration = time.time() - start_time
            tool_duration_histogram.record(duration, {
                "tool_name": tool_name,
                "status": "error"
            })
        
        # Update concurrent operations
        if concurrent_operations_gauge and operation_id and operation_id in _active_operations:
            concurrent_operations_gauge.add(-1, {"tool_name": tool_name})
            del _active_operations[operation_id]
            
    except Exception as e:
        print(f"Warning: Failed to record tool call error metrics: {e}")

def get_metrics_export() -> tuple[str, str]:
    """Export metrics in Prometheus format for the /metrics endpoint."""
    if not is_metrics_enabled() or not prometheus_reader:
        return "# Metrics not available or disabled\n", "text/plain"
    
    try:
        # Generate metrics from the Prometheus reader
        metrics_data = generate_latest()
        return metrics_data.decode('utf-8'), CONTENT_TYPE_LATEST
    except Exception as e:
        error_msg = f"# Error generating metrics: {e}\n"
        return error_msg, "text/plain"

def get_metrics_summary() -> Dict[str, Any]:
    """Get a summary of current metrics for debugging/monitoring."""
    if not is_metrics_enabled():
        return {"status": "disabled", "reason": "OpenTelemetry not available or disabled"}
    
    return {
        "status": "enabled",
        "service_name": SERVICE_NAME,
        "service_version": SERVICE_VERSION,
        "environment": DEPLOYMENT_ENVIRONMENT,
        "otlp_endpoint": OTEL_ENDPOINT if OTEL_ENDPOINT else "not_configured",
        "active_operations": len(_active_operations),
        "prometheus_enabled": prometheus_reader is not None
    }

def instrument_tool(func):
    """Decorator to automatically instrument MCP tools with metrics."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        tool_name = func.__name__
        start_time = record_tool_call_start(tool_name, args, kwargs)
        
        try:
            result = func(*args, **kwargs)
            # Calculate result size if possible
            result_size = 0
            try:
                if isinstance(result, int):
                    result_size = result
                elif hasattr(result, '__len__'):
                    result_size = len(str(result))
            except:
                pass
            record_tool_call_success(tool_name, start_time, result_size)
            return result
        except Exception as e:
            record_tool_call_error(tool_name, start_time, e)
            raise
    
    return wrapper

# Initialize metrics on module import
initialize_metrics() 