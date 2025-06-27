import os
import time
from typing import Optional

try:
    from prometheus_client import Counter, Histogram, Info, generate_latest, CONTENT_TYPE_LATEST
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    # Create stub classes for type hints when prometheus_client is not available
    class Counter: pass
    class Histogram: pass  
    class Info: pass

# Configuration from environment variables
METRICS_ENABLED = os.getenv("MCP_METRICS_ENABLED", "true").lower() == "true"

# Metrics instances - only created if metrics are available and enabled
mcp_tool_calls_total: Optional[Counter] = None
mcp_tool_duration_seconds: Optional[Histogram] = None
mcp_tool_errors_total: Optional[Counter] = None
mcp_tool_argument_sizes: Optional[Histogram] = None
mcp_server_info: Optional[Info] = None

def initialize_metrics():
    """Initialize Prometheus metrics if the library is available and metrics are enabled."""
    global mcp_tool_calls_total, mcp_tool_duration_seconds, mcp_tool_errors_total
    global mcp_tool_argument_sizes, mcp_server_info
    
    if not METRICS_AVAILABLE or not METRICS_ENABLED:
        return
    
    try:
        # Tool usage metrics
        mcp_tool_calls_total = Counter(
            'mcp_tool_calls_total',
            'Total number of MCP tool calls',
            ['tool_name', 'status']
        )
        
        # Tool execution time
        mcp_tool_duration_seconds = Histogram(
            'mcp_tool_duration_seconds',
            'MCP tool execution time in seconds',
            ['tool_name'],
            buckets=[0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, float('inf')]
        )
        
        # Tool errors
        mcp_tool_errors_total = Counter(
            'mcp_tool_errors_total',
            'Total number of MCP tool errors',
            ['tool_name', 'error_type']
        )
        
        # Argument size metrics
        mcp_tool_argument_sizes = Histogram(
            'mcp_tool_argument_sizes_bytes',
            'Size of arguments passed to MCP tools',
            ['tool_name'],
            buckets=[100, 1000, 10000, 100000, 1000000, float('inf')]
        )
        
        # Server info
        mcp_server_info = Info(
            'mcp_server_info',
            'Information about the MCP server'
        )
        
        # Set server info
        mcp_server_info.info({
            'version': '0.0.1',
            'server_type': 'document_mcp',
            'metrics_enabled': 'true'
        })
        
    except Exception as e:
        print(f"Warning: Failed to initialize metrics: {e}")

def is_metrics_enabled() -> bool:
    """Check if metrics collection is enabled and available."""
    return METRICS_AVAILABLE and METRICS_ENABLED and mcp_tool_calls_total is not None

def record_tool_call_start(tool_name: str, args: tuple, kwargs: dict) -> Optional[float]:
    """Record the start of a tool call and return start time for duration calculation."""
    if not is_metrics_enabled():
        return None
    
    start_time = time.time()
    
    try:
        # Record argument size
        if mcp_tool_argument_sizes:
            # Simple size calculation
            args_str = repr(args)
            kwargs_str = repr(kwargs)
            arg_size = len(args_str.encode('utf-8')) + len(kwargs_str.encode('utf-8'))
            mcp_tool_argument_sizes.labels(tool_name=tool_name).observe(arg_size)
    except Exception as e:
        print(f"Warning: Failed to record tool call start metrics: {e}")
    
    return start_time

def record_tool_call_success(tool_name: str, start_time: Optional[float]):
    """Record a successful tool call completion."""
    if not is_metrics_enabled() or start_time is None:
        return
    
    try:
        # Record success
        if mcp_tool_calls_total:
            mcp_tool_calls_total.labels(tool_name=tool_name, status='success').inc()
        
        # Record duration
        if mcp_tool_duration_seconds:
            duration = time.time() - start_time
            mcp_tool_duration_seconds.labels(tool_name=tool_name).observe(duration)
    except Exception as e:
        print(f"Warning: Failed to record tool call success metrics: {e}")

def record_tool_call_error(tool_name: str, start_time: Optional[float], error: Exception):
    """Record a failed tool call."""
    if not is_metrics_enabled():
        return
    
    try:
        # Record error
        if mcp_tool_calls_total:
            mcp_tool_calls_total.labels(tool_name=tool_name, status='error').inc()
        
        if mcp_tool_errors_total:
            error_type = type(error).__name__
            mcp_tool_errors_total.labels(tool_name=tool_name, error_type=error_type).inc()
        
        # Record duration even for errors if we have start time
        if mcp_tool_duration_seconds and start_time is not None:
            duration = time.time() - start_time
            mcp_tool_duration_seconds.labels(tool_name=tool_name).observe(duration)
    except Exception as e:
        print(f"Warning: Failed to record tool call error metrics: {e}")

def get_metrics_export() -> tuple[str, str]:
    """Export metrics in Prometheus format."""
    if not is_metrics_enabled():
        return "# Metrics not available or disabled\n", "text/plain"
    
    try:
        metrics_data = generate_latest()
        return metrics_data.decode('utf-8'), CONTENT_TYPE_LATEST
    except Exception as e:
        error_msg = f"# Error generating metrics: {e}\n"
        return error_msg, "text/plain"

# Initialize metrics on module import
initialize_metrics() 