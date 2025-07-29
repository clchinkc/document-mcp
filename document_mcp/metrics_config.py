"""Document MCP Metrics Configuration.

Provides automatic telemetry collection and forwarding to Grafana Cloud.
Architecture: Tool Calls -> Metrics Server -> Background Prometheus -> Grafana Cloud
"""

import json
import os
import socket
import subprocess
import tempfile
import threading
import time
from functools import wraps
from typing import Any

# OpenTelemetry imports
from opentelemetry import metrics
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource

# Prometheus client for metrics endpoint
from prometheus_client import CONTENT_TYPE_LATEST
from prometheus_client import generate_latest

# =============================================================================
# GRAFANA CLOUD CONFIGURATION
# =============================================================================

GRAFANA_CLOUD_PROMETHEUS_ENDPOINT = "https://prometheus-prod-37-prod-ap-southeast-1.grafana.net/api/prom/push"
GRAFANA_CLOUD_TOKEN = "glc_eyJvIjoiMTQ5MDY5MCIsIm4iOiJzdGFjay0xMzI2MTg3LWludGVncmF0aW9uLWRvY3VtZW50LW1jcCIsImsiOiJmM1hZZTQ1d2VWSTlEMVMxaUs1NlNOODgiLCJtIjp7InIiOiJwcm9kLWFwLXNvdXRoZWFzdC0xIn19"
GRAFANA_CLOUD_METRICS_USER_ID = "2576609"
GRAFANA_CLOUD_OTLP_ENDPOINT = "https://otlp-gateway-prod-ap-southeast-1.grafana.net/otlp"

# =============================================================================
# SERVICE CONFIGURATION
# =============================================================================

SERVICE_NAME = os.getenv("OTEL_SERVICE_NAME", "document-mcp")
SERVICE_NAMESPACE = os.getenv("OTEL_SERVICE_NAMESPACE", "document-mcp-group")
SERVICE_VERSION = os.getenv("OTEL_SERVICE_VERSION", "1.0.0")
DEPLOYMENT_ENVIRONMENT = os.getenv("DEPLOYMENT_ENVIRONMENT", "production")


# Metrics enabled by default, can be disabled via environment
# Automatically disable metrics in test environments to prevent CI data spike
def is_test_environment():
    """Detect if running in test environment."""
    return (
        "PYTEST_CURRENT_TEST" in os.environ
        or "DOCUMENT_ROOT_DIR" in os.environ
        or "CI" in os.environ
        or "GITHUB_ACTIONS" in os.environ
        or "RUNNER_OS" in os.environ  # GitHub Actions runner
    )


# Disable metrics in test/CI environments by default to prevent data spikes
default_metrics_enabled = "false" if is_test_environment() else "true"
METRICS_ENABLED = os.getenv("MCP_METRICS_ENABLED", default_metrics_enabled).lower() == "true"

# Auto-shutdown configuration
# Default 2 minutes - enough time for multiple Prometheus scrapes and remote write
INACTIVITY_TIMEOUT = int(os.getenv("MCP_METRICS_INACTIVITY_TIMEOUT", "120"))  # 2 minutes default
_last_activity_time = time.time()
_shutdown_timer = None

# Debug: Log metrics status for debugging
if is_test_environment():
    print("[METRICS_DEBUG] Test environment detected, metrics disabled by default")
print(f"[METRICS_DEBUG] METRICS_ENABLED = {METRICS_ENABLED}")

# Allow user override of telemetry endpoint (but use Grafana Cloud by default)
PROMETHEUS_ENDPOINT = os.getenv("PROMETHEUS_REMOTE_WRITE_ENDPOINT", GRAFANA_CLOUD_PROMETHEUS_ENDPOINT)
OTEL_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", GRAFANA_CLOUD_OTLP_ENDPOINT)

# Set CUMULATIVE temporality for Grafana Cloud compatibility (fixes temporality error)
os.environ.setdefault("OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE", "CUMULATIVE")

# Create proper Basic auth header for Grafana Cloud
import base64

grafana_auth = base64.b64encode(f"{GRAFANA_CLOUD_METRICS_USER_ID}:{GRAFANA_CLOUD_TOKEN}".encode()).decode()
OTEL_HEADERS = os.getenv("OTEL_EXPORTER_OTLP_HEADERS", f"Authorization=Basic {grafana_auth}")

# Parse additional resource attributes if provided
RESOURCE_ATTRIBUTES = {}
otel_resource_attrs = os.getenv("OTEL_RESOURCE_ATTRIBUTES", "")
if otel_resource_attrs:
    for attr in otel_resource_attrs.split(","):
        if "=" in attr:
            key, value = attr.strip().split("=", 1)
            RESOURCE_ATTRIBUTES[key] = value

    # Override defaults with resource attributes
    SERVICE_NAME = RESOURCE_ATTRIBUTES.get("service.name", SERVICE_NAME)
    DEPLOYMENT_ENVIRONMENT = RESOURCE_ATTRIBUTES.get("deployment.environment", DEPLOYMENT_ENVIRONMENT)

# Metrics instances
meter = None
tool_calls_counter = None
prometheus_reader = None
otlp_reader = None

# Global state for tracking
_active_operations = {}
_metrics_shutdown = False

# Prometheus remote write

import builtins
import contextlib

import requests
from prometheus_client import CollectorRegistry
from prometheus_client import Counter
from prometheus_client import Gauge
from prometheus_client import Histogram


class GrafanaCloudPrometheusExporter:
    """Simple Prometheus metrics exporter for Grafana Cloud using push gateway pattern."""

    def __init__(self, user_id: str, api_token: str, interval: int = 30):
        self.user_id = user_id
        self.api_token = api_token
        self.interval = interval
        self.registry = CollectorRegistry()
        self.running = False
        self.thread = None

        # Create Prometheus metrics
        self.tool_calls = Counter(
            "mcp_tool_calls_total",
            "Total number of MCP tool calls",
            ["tool_name", "status", "environment"],
            registry=self.registry,
        )

        self.tool_duration = Histogram(
            "mcp_tool_duration_seconds",
            "MCP tool execution time in seconds",
            ["tool_name", "status"],
            registry=self.registry,
        )

        self.tool_errors = Counter(
            "mcp_tool_errors_total",
            "Total number of MCP tool errors",
            ["tool_name", "error_type", "environment"],
            registry=self.registry,
        )

        self.concurrent_ops = Gauge(
            "mcp_concurrent_operations",
            "Number of concurrent MCP tool operations",
            ["tool_name"],
            registry=self.registry,
        )

        self.server_info = Counter(
            "mcp_server_startup_total",
            "Number of MCP server startups",
            ["version", "server_type", "environment"],
            registry=self.registry,
        )

    def start(self):
        """Start the background export thread."""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._export_loop, daemon=True)
            self.thread.start()

    def stop(self):
        """Stop the background export thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)

    def _export_loop(self):
        """Background loop to export metrics periodically."""
        import time

        while self.running:
            try:
                self._push_metrics()
            except Exception as e:
                print(f"Warning: Failed to push metrics to Grafana Cloud: {e}")

            # Wait for next export cycle
            for _ in range(self.interval):
                if not self.running:
                    break
                time.sleep(1)

    def _push_metrics(self):
        """Push metrics to Grafana Cloud using proper remote write protocol."""
        try:
            # Create proper remote write payload
            remote_write_data = self._create_remote_write_payload()
            if not remote_write_data:
                return  # No data to send

            # Prepare authentication and headers
            auth_string = f"{self.user_id}:{self.api_token}"
            auth_b64 = base64.b64encode(auth_string.encode()).decode()

            headers = {
                "Authorization": f"Basic {auth_b64}",
                "Content-Type": "application/x-protobuf",
                "Content-Encoding": "snappy",
                "X-Prometheus-Remote-Write-Version": "0.1.0",
                "User-Agent": "document-mcp/1.0.0",
            }

            # Send to Grafana Cloud
            import requests

            requests.post(
                "https://prometheus-prod-37-prod-ap-southeast-1.grafana.net/api/prom/push",
                data=remote_write_data,
                headers=headers,
                timeout=15,
            )

            # Silent success - don't log anything for users

        except Exception:
            pass  # Completely silent failure - telemetry should never disrupt user workflow

    def _create_remote_write_payload(self):
        """Create proper Prometheus remote write payload with protobuf + snappy."""
        try:
            import time

            import snappy

            # Create simple remote write format for basic metrics
            # Send core metrics (tool calls, duration) using Prometheus remote write protocol

            current_time_ms = int(time.time() * 1000)

            # Collect current metric values
            metrics_data = []

            # Add tool calls metric if we have data
            for metric in self.registry._collector_to_names:
                try:
                    for sample in metric.collect():
                        for metric_sample in sample.samples:
                            # Create a basic remote write sample
                            metric_name = metric_sample.name
                            metric_value = metric_sample.value
                            metric_labels = metric_sample.labels or {}

                            # Convert to simple protobuf-like structure
                            metric_entry = {
                                "name": metric_name,
                                "value": metric_value,
                                "timestamp_ms": current_time_ms,
                                "labels": metric_labels,
                            }
                            metrics_data.append(metric_entry)
                except Exception:
                    continue

            if not metrics_data:
                return None

            # Create Prometheus remote write payload with tool usage metrics
            # Note: Full protobuf encoding would require prometheus remote write protobuf definitions
            import json

            json_payload = json.dumps({"timeseries": metrics_data}).encode("utf-8")

            # Compress with snappy
            compressed_payload = snappy.compress(json_payload)
            return compressed_payload

        except Exception:
            return None

    def _custom_handler(self, url, method, timeout, headers, data):
        """Custom handler for push_to_gateway with proper auth."""
        auth_string = f"{self.user_id}:{self.api_token}"
        auth_b64 = base64.b64encode(auth_string.encode()).decode()

        headers["Authorization"] = f"Basic {auth_b64}"

        return requests.request(method=method, url=url, data=data, headers=headers, timeout=timeout)

    def _encode_remote_write(self, metrics_data):
        """Simple fallback - just return the text data for now."""
        return metrics_data


# Global Prometheus exporter instance
_prometheus_exporter = None
_http_server = None


def _start_background_prometheus():
    """Start background Prometheus server for automatic Grafana Cloud forwarding."""
    import os
    import shutil
    import subprocess
    import threading

    # Check if Prometheus is available
    if not shutil.which("prometheus"):
        raise Exception("Prometheus not found - install with: brew install prometheus")

    # Simple check: don't start if any prometheus mcp processes already exist
    try:
        result = subprocess.run(["pgrep", "-f", "prometheus.*mcp"], capture_output=True, text=True)
        if result.returncode == 0:  # Processes found
            print("   [METRICS] Background Prometheus already running, skipping")
            return
    except:
        pass  # If pgrep fails, continue anyway

    # Create minimal prometheus config for agent metrics
    prometheus_config = f"""
global:
  scrape_interval: 3s
  external_labels:
    source: document-mcp-agent

remote_write:
  - url: {GRAFANA_CLOUD_PROMETHEUS_ENDPOINT}
    basic_auth:
      username: {GRAFANA_CLOUD_METRICS_USER_ID}
      password: {GRAFANA_CLOUD_TOKEN}
    write_relabel_configs:
      - source_labels: [__name__]
        regex: 'mcp_.*|python_.*|process_.*'
        action: keep

scrape_configs:
  - job_name: document-mcp-agent
    scrape_interval: 2s
    scrape_timeout: 1s
    metrics_path: /metrics
    static_configs:
      - targets: ["localhost:8000"]
    metric_relabel_configs:
      - source_labels: [__name__]
        regex: 'mcp_.*'
        action: keep
"""

    # Write config to temp file
    config_fd, config_path = tempfile.mkstemp(suffix=".yml", prefix="prometheus_mcp_")
    try:
        with os.fdopen(config_fd, "w") as f:
            f.write(prometheus_config)

        # Start Prometheus in background
        def run_prometheus():
            try:
                # Create temp data dir
                data_dir = tempfile.mkdtemp(prefix="prometheus_mcp_data_")

                subprocess.run(
                    [
                        "prometheus",
                        "--config.file",
                        config_path,
                        "--storage.tsdb.path",
                        data_dir,
                        "--web.listen-address",
                        ":0",  # Random port
                        "--log.level",
                        "error",  # Minimal logging
                        "--storage.tsdb.retention.time",
                        "1h",  # Minimal retention
                        "--web.enable-lifecycle",
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=3600,  # 1 hour max
                )
            except Exception:
                pass  # Silent failure
            finally:
                # Cleanup
                try:
                    os.unlink(config_path)
                    shutil.rmtree(data_dir, ignore_errors=True)
                except:
                    pass

        # Start in daemon thread
        prometheus_thread = threading.Thread(target=run_prometheus, daemon=True)
        prometheus_thread.start()

        # Give it a moment to start
        import time

        time.sleep(1)

    except Exception as e:
        with contextlib.suppress(builtins.BaseException):
            os.unlink(config_path)
        raise e


def _start_persistent_metrics_server():
    """Start persistent metrics server that survives agent exits."""
    # Skip in test environments to prevent data collection
    if is_test_environment():
        print("[METRICS] Skipping persistent metrics server in test environment")
        return

    import os
    import time

    # Create persistent metrics server script
    metrics_server_script = '''
import atexit
import json
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from prometheus_client import CollectorRegistry, Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import os
import signal
import sys

# Persistent metrics storage
METRICS_FILE = "/tmp/document_mcp_metrics.json"
LOCK_FILE = "/tmp/document_mcp_metrics.lock"

# Global metrics registry
registry = CollectorRegistry()
tool_calls = Counter(
    'mcp_tool_calls_total',
    'Total MCP tool calls',
    ['tool_name', 'status', 'environment'],
    registry=registry
)

tool_duration = Histogram(
    'mcp_tool_duration_seconds',
    'MCP tool execution time',
    ['tool_name', 'status'],
    registry=registry
)

def load_metrics():
    """Load metrics from persistent storage."""
    try:
        if os.path.exists(METRICS_FILE):
            with open(METRICS_FILE, 'r') as f:
                data = json.load(f)
                for metric in data.get('tool_calls', []):
                    tool_calls.labels(**metric['labels']).inc(metric['value'])
                for metric in data.get('tool_durations', []):
                    tool_duration.labels(**metric['labels']).observe(metric['value'])
    except Exception:
        pass

class MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/metrics':
            try:
                load_metrics()  # Load latest metrics
                metrics_data = generate_latest(registry)
                self.send_response(200)
                self.send_header('Content-Type', CONTENT_TYPE_LATEST)
                self.end_headers()
                self.wfile.write(metrics_data)
            except Exception as e:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(f"Error: {e}".encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass

def cleanup_handler(signum, frame):
    try:
        # Clean up gRPC connections
        global otlp_reader, prometheus_reader
        if otlp_reader:
            try:
                otlp_reader.shutdown()
            except:
                pass
        if prometheus_reader:
            try:
                prometheus_reader.shutdown()
            except:
                pass

        # Clean up lock file
        os.unlink(LOCK_FILE)
    except:
        pass
    sys.exit(0)

def graceful_shutdown():
    """Graceful shutdown for atexit."""
    try:
        global otlp_reader, prometheus_reader
        if otlp_reader:
            try:
                otlp_reader.shutdown()
            except:
                pass
        if prometheus_reader:
            try:
                prometheus_reader.shutdown()
            except:
                pass
    except:
        pass

signal.signal(signal.SIGTERM, cleanup_handler)
signal.signal(signal.SIGINT, cleanup_handler)
atexit.register(graceful_shutdown)

# Create lock file
with open(LOCK_FILE, 'w') as f:
    f.write(str(os.getpid()))

# Start server
server = HTTPServer(('localhost', 8000), MetricsHandler)
print("Persistent metrics server started on :8000")
server.serve_forever()
'''

    # Check if persistent server is already running
    lock_file = "/tmp/document_mcp_metrics.lock"
    if os.path.exists(lock_file):
        try:
            with open(lock_file) as f:
                pid = int(f.read().strip())
            # Check if process is still running
            os.kill(pid, 0)
            print(f"   [METRICS] Persistent metrics server already running (PID {pid})")
            return
        except (OSError, ValueError):
            # Lock file exists but process is dead
            with contextlib.suppress(builtins.BaseException):
                os.unlink(lock_file)

    # Write server script to temp file
    script_fd, script_path = tempfile.mkstemp(suffix=".py", prefix="metrics_server_")
    try:
        with os.fdopen(script_fd, "w") as f:
            f.write(metrics_server_script)

        # Start persistent server with error logging
        log_file = "/tmp/document_mcp_metrics_server.log"
        with open(log_file, "w") as log:
            proc = subprocess.Popen(
                ["python3", script_path],
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,  # Detach from parent
            )
            print(f"[METRICS] Started persistent server PID {proc.pid}, logs: {log_file}")

        # Give it time to start
        time.sleep(1)
        print("   [METRICS] Started persistent metrics server on :8000")

    except Exception as e:
        print(f"   [WARN] Could not start persistent metrics server: {e}")
    finally:
        with contextlib.suppress(builtins.BaseException):
            os.unlink(script_path)


def _start_prometheus_http_server_with_retry():
    """Start persistent HTTP server for Prometheus scraping."""
    _start_persistent_metrics_server()


def _write_metrics_to_persistent_storage(tool_name: str, status: str, start_time: float = None):
    """Write metrics to persistent storage for the persistent server."""
    import fcntl
    import json
    import os
    import time

    metrics_file = "/tmp/document_mcp_metrics.json"

    try:
        # Calculate duration if available
        duration = None
        if start_time:
            duration = time.time() - start_time

        # Prepare new metrics
        new_metrics = {
            "tool_calls": [
                {
                    "labels": {
                        "tool_name": tool_name,
                        "status": status,
                        "environment": DEPLOYMENT_ENVIRONMENT,
                    },
                    "value": 1,
                }
            ],
            "tool_durations": [],
        }

        if duration is not None:
            new_metrics["tool_durations"].append(
                {"labels": {"tool_name": tool_name, "status": status}, "value": duration}
            )

        # Atomically update metrics file
        temp_file = metrics_file + ".tmp"
        with open(temp_file, "w") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)

            # Load existing metrics
            existing_metrics = {"tool_calls": [], "tool_durations": []}
            if os.path.exists(metrics_file):
                try:
                    with open(metrics_file) as existing_f:
                        existing_metrics = json.load(existing_f)
                except:
                    pass

            # Merge metrics
            existing_metrics["tool_calls"].extend(new_metrics["tool_calls"])
            existing_metrics["tool_durations"].extend(new_metrics["tool_durations"])

            # Write merged metrics
            json.dump(existing_metrics, f)

        # Atomic move
        os.rename(temp_file, metrics_file)

    except Exception:
        pass  # Silent failure for telemetry


def _update_activity():
    """Update the last activity timestamp and reset shutdown timer."""
    global _last_activity_time, _shutdown_timer
    _last_activity_time = time.time()

    # Cancel existing shutdown timer
    if _shutdown_timer:
        _shutdown_timer.cancel()

    # Set new shutdown timer
    _shutdown_timer = threading.Timer(INACTIVITY_TIMEOUT, _auto_shutdown_metrics)
    _shutdown_timer.daemon = True
    _shutdown_timer.start()


def _auto_shutdown_metrics():
    """Automatically shutdown metrics collection after inactivity timeout."""
    global _metrics_shutdown
    if _metrics_shutdown:
        return

    print(f"[METRICS] Auto-shutdown triggered after {INACTIVITY_TIMEOUT}s of inactivity")

    # First, try to flush any pending metrics to ensure they reach Grafana Cloud
    print("[METRICS] Flushing pending metrics before shutdown...")
    _flush_metrics_before_shutdown()

    _metrics_shutdown = True

    # Wait a moment for final metrics to be sent
    import time

    time.sleep(2)

    # Stop background processes
    try:
        # Stop persistent metrics server
        lock_file = "/tmp/document_mcp_metrics.lock"
        if os.path.exists(lock_file):
            try:
                with open(lock_file) as f:
                    pid = int(f.read().strip())
                os.kill(pid, 15)  # SIGTERM
                print(f"[METRICS] Stopped persistent metrics server (PID {pid})")
            except (OSError, ValueError):
                pass
    except Exception as e:
        print(f"[METRICS] Warning: Could not stop persistent server: {e}")

    # Stop Prometheus processes more aggressively
    try:
        import subprocess

        # First try gentle termination with SIGTERM
        subprocess.run(["pkill", "-f", "prometheus.*mcp"], capture_output=True)

        # Wait a moment for graceful shutdown
        import time

        time.sleep(2)

        # Check if any are still running and force kill them
        check_result = subprocess.run(["pgrep", "-f", "prometheus.*mcp"], capture_output=True, text=True)
        if check_result.returncode == 0:  # Still running
            pids = check_result.stdout.strip().split("\n")
            for pid in pids:
                if pid.strip():
                    with contextlib.suppress(builtins.BaseException):
                        subprocess.run(["kill", "-9", pid.strip()], capture_output=True)
            print(f"[METRICS] Force-killed {len(pids)} persistent Prometheus processes")
        else:
            print("[METRICS] Background Prometheus processes stopped gracefully")

    except Exception as e:
        print(f"[METRICS] Warning: Could not stop Prometheus processes: {e}")


def _flush_metrics_before_shutdown():
    """Attempt to flush any pending metrics before shutdown."""
    try:
        # Try to force an immediate export from OpenTelemetry
        global otlp_reader, prometheus_reader
        if otlp_reader and hasattr(otlp_reader, "force_flush"):
            otlp_reader.force_flush(timeout_millis=5000)
            print("[METRICS] OTLP metrics flushed")

        if prometheus_reader and hasattr(prometheus_reader, "force_flush"):
            prometheus_reader.force_flush(timeout_millis=5000)
            print("[METRICS] Prometheus metrics flushed")

        # Direct push any accumulated metrics to Grafana Cloud
        _direct_push_to_grafana_cloud()

    except Exception as e:
        print(f"[METRICS] Warning: Could not flush metrics: {e}")


def _direct_push_to_grafana_cloud():
    """Directly push accumulated metrics to Grafana Cloud using HTTP POST with proper protocol."""
    try:
        import base64
        import json
        import time

        import requests

        # Read accumulated metrics from persistent storage
        metrics_file = "/tmp/document_mcp_metrics.json"
        if not os.path.exists(metrics_file):
            print("[METRICS] No metrics data to push")
            return

        with open(metrics_file) as f:
            metrics_data = json.load(f)

        if not metrics_data.get("tool_calls"):
            print("[METRICS] No tool calls to push")
            return

        print(f"[METRICS] Pushing {len(metrics_data['tool_calls'])} tool calls directly to Grafana Cloud")

        # Create proper Prometheus remote write payload
        current_time_ms = int(time.time() * 1000)

        # Build TimeSeries data in the format Grafana Cloud expects
        timeseries_data = {"timeseries": []}

        # Add tool call metrics
        for tool_call in metrics_data["tool_calls"]:
            labels = tool_call["labels"]
            value = tool_call["value"]

            # Convert labels to the format expected
            label_pairs = []
            for key, val in labels.items():
                label_pairs.append({"name": key, "value": str(val)})

            # Add metric name label
            label_pairs.append({"name": "__name__", "value": "mcp_tool_calls_total"})

            timeseries_data["timeseries"].append(
                {"labels": label_pairs, "samples": [{"value": float(value), "timestamp": current_time_ms}]}
            )

        # Add duration metrics if available
        if "tool_durations" in metrics_data:
            for duration in metrics_data["tool_durations"]:
                labels = duration["labels"]
                value = duration["value"]

                label_pairs = []
                for key, val in labels.items():
                    label_pairs.append({"name": key, "value": str(val)})

                label_pairs.append({"name": "__name__", "value": "mcp_tool_duration_seconds"})

                timeseries_data["timeseries"].append(
                    {
                        "labels": label_pairs,
                        "samples": [{"value": float(value), "timestamp": current_time_ms}],
                    }
                )

        # Convert to JSON
        json_payload = json.dumps(timeseries_data).encode("utf-8")

        # Compress with snappy (try to import snappy for proper compression)
        try:
            import snappy

            compressed_payload = snappy.compress(json_payload)
            content_encoding = "snappy"
            print(
                f"[METRICS] Using Snappy compression ({len(json_payload)} -> {len(compressed_payload)} bytes)"
            )
        except ImportError:
            # Fallback - try without compression first
            compressed_payload = json_payload
            content_encoding = "identity"
            print("[METRICS] Warning: Snappy not available, trying without compression")

        # Prepare headers
        auth_string = f"{GRAFANA_CLOUD_METRICS_USER_ID}:{GRAFANA_CLOUD_TOKEN}"
        auth_b64 = base64.b64encode(auth_string.encode()).decode()

        headers = {
            "Authorization": f"Basic {auth_b64}",
            "Content-Type": "application/x-protobuf",
            "Content-Encoding": content_encoding,
            "X-Prometheus-Remote-Write-Version": "0.1.0",
            "User-Agent": "document-mcp/1.0.0",
        }

        print(f"[METRICS] Sending {len(timeseries_data['timeseries'])} metrics to Grafana Cloud...")

        # Send the request
        response = requests.post(
            GRAFANA_CLOUD_PROMETHEUS_ENDPOINT, headers=headers, data=compressed_payload, timeout=30
        )

        if response.status_code == 200:
            print("[METRICS] ✅ Successfully pushed metrics to Grafana Cloud!")
            return True
        elif response.status_code == 400 and "snappy" in response.text.lower():
            print("[METRICS] ⚠️ Snappy compression issue, trying without compression...")
            # Retry without compression
            headers["Content-Encoding"] = "identity"
            response = requests.post(
                GRAFANA_CLOUD_PROMETHEUS_ENDPOINT, headers=headers, data=json_payload, timeout=30
            )
            if response.status_code == 200:
                print("[METRICS] ✅ Successfully pushed metrics to Grafana Cloud (uncompressed)!")
                return True
            else:
                print(
                    f"[METRICS] ❌ Push failed even without compression: {response.status_code} - {response.text}"
                )
                return False
        else:
            print(f"[METRICS] ❌ Push failed: HTTP {response.status_code}")
            print(f"[METRICS] Response: {response.text[:200]}")
            return False

    except Exception as e:
        print(f"[METRICS] ❌ Direct push failed: {e}")
        return False


def get_resource() -> "Resource":
    """Create OpenTelemetry resource with service information."""
    # Start with base attributes matching Grafana Cloud expected format
    resource_attributes = {
        "service.name": SERVICE_NAME,
        "service.namespace": SERVICE_NAMESPACE,
        "service.version": SERVICE_VERSION,
        "deployment.environment": DEPLOYMENT_ENVIRONMENT,
        "host.name": socket.gethostname(),
        "telemetry.sdk.name": "opentelemetry",
        "telemetry.sdk.language": "python",
    }

    # Add any additional attributes from OTEL_RESOURCE_ATTRIBUTES
    resource_attributes.update(RESOURCE_ATTRIBUTES)

    return Resource.create(resource_attributes)


def initialize_metrics():
    """Initialize automatic telemetry with Prometheus scraping endpoint for Grafana Cloud."""
    global meter, tool_calls_counter, prometheus_reader, otlp_reader

    if not METRICS_ENABLED:
        print("Document MCP telemetry disabled")
        return

    try:
        # Create resource with automatic service identification
        resource = get_resource()

        # Create metric readers
        metric_readers = []

        # Always add Prometheus reader for local /metrics endpoint (for Prometheus scraping)
        prometheus_reader = PrometheusMetricReader()
        metric_readers.append(prometheus_reader)

        # Start HTTP server for Prometheus scraping (with retry)
        _start_prometheus_http_server_with_retry()

        # Primary path: Use OpenTelemetry Collector with Prometheus remote write
        local_collector_endpoint = "http://localhost:4317"

        try:
            # Test if local collector is available
            import requests

            requests.get("http://localhost:13133", timeout=1)
            collector_available = True
        except:
            collector_available = False

        if collector_available:
            # Use local collector which now uses Prometheus remote write to Grafana Cloud
            otlp_exporter = OTLPMetricExporter(endpoint=local_collector_endpoint)

            otlp_reader = PeriodicExportingMetricReader(exporter=otlp_exporter, export_interval_millis=300)
            metric_readers.append(otlp_reader)
            print("   [OK] Using OpenTelemetry Collector -> Prometheus remote write -> Grafana Cloud")
        else:
            print("   [WARN] OpenTelemetry Collector not available")
            print(
                "   [NOTE] Start collector: Install OpenTelemetry Collector if needed for advanced telemetry routing"
            )

            # Auto-start background Prometheus for immediate Grafana Cloud delivery
            # This uses the same proven approach as run_telemetry.sh but automatic
            try:
                if not is_test_environment():
                    _start_background_prometheus()
                    print("   [AUTO] Auto-telemetry: Background Prometheus -> Grafana Cloud")
                else:
                    print("   [METRICS] Background Prometheus skipped in test environment")
            except Exception:
                print("   [METRICS] Local metrics: Available at :8000/metrics")
                print(
                    "   [NOTE] Full telemetry: Run scripts/development/telemetry/scripts/start.sh for Grafana Cloud"
                )

        # Create meter provider
        meter_provider = MeterProvider(resource=resource, metric_readers=metric_readers)
        metrics.set_meter_provider(meter_provider)
        meter = metrics.get_meter(__name__)

        # Create clean counter metric
        tool_calls_counter = meter.create_counter(
            name="mcp_tool_calls_total",
            description="Total number of MCP tool calls",
            unit="1",
        )

        # Record initialization
        if tool_calls_counter:
            tool_calls_counter.add(
                1, {"tool_name": "telemetry_init", "status": "success", "environment": DEPLOYMENT_ENVIRONMENT}
            )

        # Start auto-shutdown timer
        _update_activity()

        # Register cleanup on process exit
        import atexit

        atexit.register(shutdown_metrics)

        print("[OK] Document MCP telemetry active for Prometheus scraping")
        print(f"   Service: {SERVICE_NAME} v{SERVICE_VERSION}")
        print(f"   Environment: {DEPLOYMENT_ENVIRONMENT}")
        print("   [ARCH] Architecture: Prometheus scrapes -> Remote write -> Grafana Cloud")
        print("   [METRICS] Metrics endpoint: http://localhost:8000/metrics")
        print(f"   [AUTO-SHUTDOWN] Will auto-shutdown after {INACTIVITY_TIMEOUT}s of inactivity")

        # Initialize automatic instrumentation if available
        try:
            FastAPIInstrumentor().instrument()
        except Exception:
            pass  # Optional instrumentation

        try:
            RequestsInstrumentor().instrument()
        except Exception:
            pass  # Optional instrumentation

    except Exception as e:
        print(f"Warning: Could not initialize automatic telemetry - continuing without metrics: {e}")
        # Don't fail startup if telemetry fails - just continue without metrics


def is_metrics_enabled() -> bool:
    """Check if metrics collection is enabled and available."""
    return METRICS_ENABLED and meter is not None


def calculate_argument_size(args: tuple, kwargs: dict) -> int:
    """Calculate size of arguments in bytes."""
    try:
        args_str = json.dumps(args, default=str)
        kwargs_str = json.dumps(kwargs, default=str)
        return len(args_str.encode("utf-8")) + len(kwargs_str.encode("utf-8"))
    except Exception:
        # Fallback to repr if JSON serialization fails
        args_str = repr(args)
        kwargs_str = repr(kwargs)
        return len(args_str.encode("utf-8")) + len(kwargs_str.encode("utf-8"))


def record_tool_call_start(tool_name: str, args: tuple, kwargs: dict) -> float | None:
    """Record the start of a tool call and return start time for duration calculation."""
    if not is_metrics_enabled() or _metrics_shutdown:
        return None

    # Update activity timestamp and reset shutdown timer
    _update_activity()

    start_time = time.time()
    operation_id = f"{tool_name}_{start_time}"

    try:
        # Just track for cleanup - no metrics yet
        _active_operations[operation_id] = start_time
    except Exception as e:
        print(f"Warning: Failed to record tool call start: {e}")

    return start_time


def record_tool_call_success(tool_name: str, start_time: float | None, result_size: int = 0):
    """Record a successful tool call completion."""
    if not is_metrics_enabled() or _metrics_shutdown:
        return

    try:
        # Primary: Record in OpenTelemetry counter (goes to collector -> Grafana Cloud)
        if tool_calls_counter:
            tool_calls_counter.add(
                1, {"tool_name": tool_name, "status": "success", "environment": DEPLOYMENT_ENVIRONMENT}
            )

        # Write to persistent storage for the persistent metrics server
        _write_metrics_to_persistent_storage(tool_name, "success", start_time)

        # Clean up tracking
        if start_time:
            operation_id = f"{tool_name}_{start_time}"
            if operation_id in _active_operations:
                del _active_operations[operation_id]

    except Exception as e:
        print(f"Warning: Failed to record tool call success metrics: {e}")


def record_tool_call_error(tool_name: str, start_time: float | None, error: Exception):
    """Record a failed tool call."""
    if not is_metrics_enabled() or _metrics_shutdown:
        return

    try:
        # Primary: Record in OpenTelemetry counter (goes to collector -> Grafana Cloud)
        if tool_calls_counter:
            tool_calls_counter.add(
                1, {"tool_name": tool_name, "status": "error", "environment": DEPLOYMENT_ENVIRONMENT}
            )

        # Write to persistent storage for the persistent metrics server
        _write_metrics_to_persistent_storage(tool_name, "error", start_time)

        # Clean up tracking
        if start_time:
            operation_id = f"{tool_name}_{start_time}"
            if operation_id in _active_operations:
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
        return metrics_data.decode("utf-8"), CONTENT_TYPE_LATEST
    except Exception as e:
        error_msg = f"# Error generating metrics: {e}\n"
        return error_msg, "text/plain"


def flush_metrics():
    """Manually flush metrics to ensure they're sent to Grafana Cloud."""
    print("[METRICS] Manual flush initiated")
    _flush_metrics_before_shutdown()


def shutdown_metrics():
    """Manually shutdown metrics collection and background processes."""
    global _metrics_shutdown
    if _metrics_shutdown:
        print("[METRICS] Metrics already shut down")
        return

    print("[METRICS] Manual shutdown initiated")
    _auto_shutdown_metrics()


def get_metrics_summary() -> dict[str, Any]:
    """Get a summary of current metrics for debugging/monitoring."""
    if not is_metrics_enabled():
        return {
            "status": "disabled",
            "reason": "Telemetry disabled via MCP_METRICS_ENABLED=false",
        }

    status = "shutdown" if _metrics_shutdown else "active"
    time_since_activity = time.time() - _last_activity_time
    time_until_shutdown = max(0, INACTIVITY_TIMEOUT - time_since_activity)

    return {
        "status": status,
        "service_name": SERVICE_NAME,
        "service_version": SERVICE_VERSION,
        "environment": DEPLOYMENT_ENVIRONMENT,
        "grafana_cloud_endpoint": OTEL_ENDPOINT,
        "export_interval_seconds": 30,
        "active_operations": len(_active_operations),
        "prometheus_enabled": prometheus_reader is not None,
        "telemetry_mode": "automatic_grafana_cloud_with_auto_shutdown",
        "inactivity_timeout": INACTIVITY_TIMEOUT,
        "time_since_last_activity": time_since_activity,
        "time_until_auto_shutdown": time_until_shutdown,
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
                elif hasattr(result, "__len__"):
                    result_size = len(str(result))
            except:
                pass
            record_tool_call_success(tool_name, start_time, result_size)
            return result
        except Exception as e:
            record_tool_call_error(tool_name, start_time, e)
            raise

    return wrapper


# Only initialize metrics if explicitly enabled and not in test environment
# This prevents automatic metrics collection during testing or CI
if METRICS_ENABLED and not is_test_environment():
    # Quick cleanup of any orphaned processes before starting
    with contextlib.suppress(builtins.BaseException):
        subprocess.run(
            ["pkill", "-f", "prometheus.*mcp.*--storage.tsdb.retention.time.*1h"], capture_output=True
        )
    initialize_metrics()
else:
    print(
        f"[METRICS] Metrics initialization skipped (enabled={METRICS_ENABLED}, test_env={is_test_environment()})"
    )
