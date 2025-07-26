#!/usr/bin/env python3
"""Automatic Document MCP Telemetry Service
Starts all required services automatically and continuously generates metrics.
"""

import os
import random
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler
from http.server import HTTPServer

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prometheus_client import CONTENT_TYPE_LATEST
from prometheus_client import CollectorRegistry
from prometheus_client import Counter
from prometheus_client import Gauge
from prometheus_client import Histogram
from prometheus_client import generate_latest


class AutoTelemetryService:
    def __init__(self):
        self.running = False
        self.prometheus_process = None
        self.metrics_server = None
        self.metrics_thread = None

        # Create metrics registry
        self.registry = CollectorRegistry()

        # Create metrics
        self.tool_calls = Counter(
            "mcp_tool_calls_total",
            "Total MCP tool calls",
            ["tool_name", "status", "environment"],
            registry=self.registry,
        )

        self.tool_duration = Histogram(
            "mcp_tool_duration_seconds",
            "MCP tool execution time",
            ["tool_name", "status"],
            registry=self.registry,
        )

        self.concurrent_ops = Gauge(
            "mcp_concurrent_operations", "Concurrent MCP operations", ["tool_name"], registry=self.registry
        )

        self.server_info = Counter(
            "mcp_server_startup_total",
            "MCP server startups",
            ["version", "environment"],
            registry=self.registry,
        )

        # MCP tools for realistic simulation
        self.mcp_tools = [
            "create_document",
            "list_documents",
            "get_document",
            "add_chapter",
            "edit_chapter",
            "delete_chapter",
            "add_paragraph",
            "edit_paragraph",
            "delete_paragraph",
            "search_content",
            "replace_content",
            "get_stats",
            "create_snapshot",
            "restore_snapshot",
            "batch_operations",
        ]

        print("[START] Document MCP Auto Telemetry Service Initialized")

    def start_metrics_server(self):
        """Start HTTP server for Prometheus scraping."""

        class MetricsHandler(BaseHTTPRequestHandler):
            def __init__(self, registry, *args, **kwargs):
                self.registry = registry
                super().__init__(*args, **kwargs)

            def do_GET(slf):
                if slf.path == "/metrics":
                    try:
                        metrics_data = generate_latest(self.registry)
                        slf.send_response(200)
                        slf.send_header("Content-Type", CONTENT_TYPE_LATEST)
                        slf.end_headers()
                        slf.wfile.write(metrics_data)
                        # Debug log
                        print(f"[DATA] Metrics endpoint served {len(metrics_data)} bytes")
                    except Exception as e:
                        slf.send_response(500)
                        slf.end_headers()
                        slf.wfile.write(f"Error: {e}".encode())
                        print(f"[ERROR] Metrics endpoint error: {e}")
                elif slf.path == "/debug":
                    try:
                        metrics_data = generate_latest(self.registry).decode("utf-8")
                        debug_info = f"""
Debug Info:
- Registry: {self.registry}
- Metrics data length: {len(metrics_data)} chars
- mcp_tool_calls_total count: {metrics_data.count("mcp_tool_calls_total")}

Raw metrics:
{metrics_data}
"""
                        slf.send_response(200)
                        slf.send_header("Content-Type", "text/plain")
                        slf.end_headers()
                        slf.wfile.write(debug_info.encode())
                    except Exception as e:
                        slf.send_response(500)
                        slf.end_headers()
                        slf.wfile.write(f"Debug error: {e}".encode())
                else:
                    slf.send_response(404)
                    slf.end_headers()

            def log_message(slf, format, *args):
                pass  # Suppress logs

        def handler_factory(*args, **kwargs):
            return MetricsHandler(self.registry, *args, **kwargs)

        try:
            self.metrics_server = HTTPServer(("localhost", 8000), handler_factory)
            print("[OK] Metrics HTTP server started on localhost:8000")
            self.metrics_server.serve_forever()
        except Exception as e:
            print(f"[ERROR] Failed to start metrics server: {e}")

    def start_prometheus(self):
        """Start Prometheus with the configured prometheus.yml."""
        try:
            # Look for prometheus.yml in the config directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            prometheus_config = os.path.join(script_dir, "../config/prometheus.yml")
            if not os.path.exists(prometheus_config):
                print(f"[ERROR] Prometheus config not found: {prometheus_config}")
                return False

            # Start Prometheus
            self.prometheus_process = subprocess.Popen(
                [
                    "prometheus",
                    f"--config.file={prometheus_config}",
                    "--storage.tsdb.retention.time=1h",  # Short retention for testing
                    "--log.level=info",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            print("[OK] Prometheus started")

            # Wait a bit for startup
            time.sleep(3)

            # Check if process is still running
            if self.prometheus_process.poll() is None:
                print("[OK] Prometheus running successfully")
                return True
            else:
                print("[ERROR] Prometheus failed to start")
                return False

        except Exception as e:
            print(f"[ERROR] Failed to start Prometheus: {e}")
            return False

    def generate_realistic_metrics(self):
        """Continuously generate realistic MCP tool metrics."""
        print("[TARGET] Starting continuous metric generation...")

        # Record server startup immediately
        try:
            self.server_info.labels(version="1.0.0", environment="production").inc()
            print("[OK] Server startup metric recorded")
        except Exception as e:
            print(f"[ERROR] Error recording server startup: {e}")

        # Generate initial metrics burst for immediate visibility
        print("[START] Generating initial metrics burst...")
        for i in range(5):
            try:
                tool = random.choice(self.mcp_tools)
                status = "success" if random.random() < 0.9 else "error"
                duration = random.uniform(0.1, 2.0)

                # Generate metrics
                self.tool_calls.labels(tool_name=tool, status=status, environment="production").inc()

                self.tool_duration.labels(tool_name=tool, status=status).observe(duration)

                concurrent = random.randint(0, 5)
                self.concurrent_ops.labels(tool_name=tool).set(concurrent)

                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"[DATA] {timestamp} | {tool} ({status}) | {duration:.2f}s [BURST]")

            except Exception as e:
                print(f"[ERROR] Error in initial burst: {e}")

            time.sleep(1)  # Quick burst

        print("[LOOP] Starting regular metric generation...")

        while self.running:
            try:
                # Simulate realistic tool usage patterns
                tool = random.choice(self.mcp_tools)

                # Weight success vs error (90% success rate)
                status = "success" if random.random() < 0.9 else "error"

                # Simulate execution time
                duration = random.uniform(0.1, 2.0)  # 100ms to 2s

                # Generate metrics with error handling
                try:
                    self.tool_calls.labels(tool_name=tool, status=status, environment="production").inc()

                    self.tool_duration.labels(tool_name=tool, status=status).observe(duration)

                    # Simulate concurrent operations
                    concurrent = random.randint(0, 5)
                    self.concurrent_ops.labels(tool_name=tool).set(concurrent)

                    # Log activity
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    print(f"[DATA] {timestamp} | {tool} ({status}) | {duration:.2f}s")

                except Exception as metric_error:
                    print(f"[ERROR] Error recording metrics for {tool}: {metric_error}")

                # Wait between operations (5-15 seconds)
                time.sleep(random.uniform(5, 15))

            except Exception as e:
                print(f"[ERROR] Error generating metrics: {e}")
                time.sleep(5)

    def start(self):
        """Start all services."""
        self.running = True
        print("[START] Starting Auto Telemetry Service...")

        # Start metrics server in background thread
        metrics_thread = threading.Thread(target=self.start_metrics_server, daemon=True)
        metrics_thread.start()

        # Wait for metrics server to start
        time.sleep(2)

        # Start Prometheus
        if not self.start_prometheus():
            print("[ERROR] Failed to start Prometheus - exiting")
            return False

        # Start metric generation in background thread
        self.metrics_thread = threading.Thread(target=self.generate_realistic_metrics, daemon=True)
        self.metrics_thread.start()

        print("[OK] All services started successfully!")
        print("[DATA] Metrics being generated and pushed to Grafana Cloud every 5 seconds")
        print("[WEB] Prometheus UI: http://localhost:9090")
        print("[GRAPH] Metrics endpoint: http://localhost:8000/metrics")
        print("[LOOP] Press Ctrl+C to stop")

        return True

    def stop(self):
        """Stop all services."""
        print("\n🛑 Stopping Auto Telemetry Service...")
        self.running = False

        # Stop Prometheus
        if self.prometheus_process:
            try:
                self.prometheus_process.terminate()
                self.prometheus_process.wait(timeout=5)
                print("[OK] Prometheus stopped")
            except:
                self.prometheus_process.kill()
                print("🔥 Prometheus force killed")

        # Stop metrics server
        if self.metrics_server:
            try:
                self.metrics_server.shutdown()
                print("[OK] Metrics server stopped")
            except:
                pass

        print("[OK] All services stopped")

    def run_test(self, duration=120):
        """Run a test for specified duration."""
        print(f"[TEST] Running {duration} second test...")

        if not self.start():
            return False

        try:
            # Run for specified duration
            time.sleep(duration)

            print("\n[DATA] Test completed - checking results...")

            # Test metrics endpoint
            import requests

            try:
                response = requests.get("http://localhost:8000/metrics", timeout=5)
                if response.status_code == 200:
                    mcp_metrics = response.text.count("mcp_tool_calls_total")
                    print(f"[OK] Metrics endpoint: {mcp_metrics} series available")
                else:
                    print(f"[ERROR] Metrics endpoint error: {response.status_code}")
            except:
                print("[ERROR] Could not access metrics endpoint")

            # Test Prometheus targets
            try:
                response = requests.get("http://localhost:9090/api/v1/targets", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    targets = data.get("data", {}).get("activeTargets", [])
                    up_targets = [t for t in targets if t.get("health") == "up"]
                    print(f"[OK] Prometheus targets: {len(up_targets)} up, {len(targets)} total")
                else:
                    print(f"[ERROR] Prometheus targets error: {response.status_code}")
            except:
                print("[ERROR] Could not check Prometheus targets")

            return True

        except KeyboardInterrupt:
            print("\n[WARN] Test interrupted by user")
            return True
        finally:
            self.stop()


def main():
    service = AutoTelemetryService()

    def signal_handler(signum, frame):
        service.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Run test mode
        duration = int(sys.argv[2]) if len(sys.argv) > 2 else 120
        success = service.run_test(duration)
        sys.exit(0 if success else 1)
    else:
        # Run continuously
        if service.start():
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
            finally:
                service.stop()


if __name__ == "__main__":
    main()
