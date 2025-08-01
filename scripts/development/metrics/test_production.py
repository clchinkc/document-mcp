#!/usr/bin/env python3
"""Simple script to test metrics collection and Grafana Cloud connectivity."""

import time

import requests

from document_mcp.metrics_config import GRAFANA_CLOUD_METRICS_USER_ID
from document_mcp.metrics_config import GRAFANA_CLOUD_PROMETHEUS_ENDPOINT
from document_mcp.metrics_config import GRAFANA_CLOUD_TOKEN
from document_mcp.metrics_config import initialize_metrics
from document_mcp.metrics_config import record_tool_call_success


def test_metrics():
    """Test metrics generation and Grafana connectivity."""
    print("[TEST] Testing Document MCP metrics collection...")

    # Initialize metrics
    print("[DATA] Initializing metrics...")
    initialize_metrics()

    # Test recording some metrics
    print("[GRAPH] Recording test metrics...")
    for _i in range(3):
        record_tool_call_success("test_tool", time.time(), 100)
        time.sleep(1)

    # Test local metrics endpoint
    print("[CHECK] Testing local metrics endpoint...")
    try:
        response = requests.get("http://localhost:8000/metrics", timeout=5)
        if response.status_code == 200:
            print(f"[OK] Local metrics endpoint working: {len(response.text)} bytes")
            # Show first few lines
            lines = response.text.split("\n")[:10]
            for line in lines[:5]:
                if line.strip() and not line.startswith("#"):
                    print(f"   [DATA] {line}")
        else:
            print(f"[ERROR] Local metrics endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"[ERROR] Local metrics endpoint error: {e}")

    # Test Grafana Cloud connectivity
    print("[WEB] Testing Grafana Cloud connectivity...")
    try:
        import base64

        auth_string = f"{GRAFANA_CLOUD_METRICS_USER_ID}:{GRAFANA_CLOUD_TOKEN}"
        auth_b64 = base64.b64encode(auth_string.encode()).decode()

        # Simple health check (GET request to see if endpoint is reachable)
        headers = {"Authorization": f"Basic {auth_b64}", "User-Agent": "document-mcp-test/1.0.0"}

        test_url = GRAFANA_CLOUD_PROMETHEUS_ENDPOINT.replace("/api/prom/push", "/api/prom/api/v1/labels")
        response = requests.get(test_url, headers=headers, timeout=10)

        if response.status_code in [200, 401, 403]:  # 401/403 means endpoint is reachable
            print(f"[OK] Grafana Cloud endpoint reachable: {response.status_code}")
        else:
            print(f"[WARN] Grafana Cloud response: {response.status_code} - {response.text[:200]}")

    except Exception as e:
        print(f"[ERROR] Grafana Cloud connectivity error: {e}")

    print("[OK] Metrics test complete!")


if __name__ == "__main__":
    test_metrics()
