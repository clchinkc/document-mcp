#!/usr/bin/env python3
"""Quick test to verify mcp_tool_calls_total metrics are working."""

import os
import subprocess
import sys
import time

import requests


def test_metrics():
    print("[TEST] Testing MCP Tool Calls Metrics Fix")
    print("=" * 50)

    # Start the telemetry service
    print("[START] Starting telemetry service...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    service_path = os.path.join(script_dir, "../services/auto_service.py")
    process = subprocess.Popen(
        [sys.executable, service_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )

    try:
        # Wait for startup
        print("⏳ Waiting for services to start...")
        time.sleep(10)

        # Test metrics endpoint
        print("[DATA] Testing metrics endpoint...")
        try:
            response = requests.get("http://localhost:8000/metrics", timeout=5)
            if response.status_code == 200:
                metrics_text = response.text
                tool_calls_count = metrics_text.count("mcp_tool_calls_total")

                print(f"[OK] Metrics endpoint working: {tool_calls_count} mcp_tool_calls_total metrics found")

                # Show sample metrics
                for line in metrics_text.split("\n"):
                    if "mcp_tool_calls_total{" in line:
                        print(f"[GRAPH] {line}")
                        break

                if tool_calls_count > 0:
                    print("[SUCCESS] SUCCESS: mcp_tool_calls_total metrics are being generated!")
                    return True
                else:
                    print("[FAIL] FAIL: No mcp_tool_calls_total metrics found")
                    return False
            else:
                print(f"[ERROR] Metrics endpoint returned {response.status_code}")
                return False

        except Exception as e:
            print(f"[ERROR] Error accessing metrics endpoint: {e}")
            return False

    finally:
        # Clean up
        print("🧹 Cleaning up...")
        process.terminate()
        try:
            process.wait(timeout=5)
        except:
            process.kill()

        # Kill any remaining processes
        subprocess.run(["pkill", "-f", "auto_telemetry_service"], capture_output=True)
        subprocess.run(["pkill", "prometheus"], capture_output=True)


if __name__ == "__main__":
    success = test_metrics()
    if success:
        print("\n[OK] Test PASSED - Metrics fix working correctly!")
        print("[START] You can now run: ./start_telemetry.sh")
    else:
        print("\n[FAIL] Test FAILED - Check the output above for issues")

    sys.exit(0 if success else 1)
