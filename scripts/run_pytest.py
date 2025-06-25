#!/usr/bin/env python3
"""
Document MCP Pytest Test Runner

A dedicated script for running pytest tests across the Document MCP project.
Handles test execution with proper configuration and environment setup.
Supports coverage reporting and CI/CD integration.
Automatically manages MCP server for integration and E2E tests.

Test Statistics:
- Total: 265 tests across 13 files (5,761 lines of test code)
- Unit tests: 177 tests across 7 files (2,618 lines)
- Integration tests: 80 tests across 3 files (2,564 lines)
- E2E tests: 8 tests across 2 files (365 lines, runs in CI/CD with API keys)
- Fixtures: Demo and documentation tests (214 lines)

The runner automatically starts an MCP server when needed for integration/E2E tests.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


class MCPServerManager:
    """Manages MCP server for integration tests."""

    def __init__(self, port=3001):
        self.port = port
        self.process = None
        self.temp_docs_root = None

    def start_server(self):
        """Start the MCP server for testing."""
        if self.process is not None:
            return  # Already started

        # Create temporary docs root
        self.temp_docs_root = Path(tempfile.mkdtemp(prefix="pytest_mcp_docs_"))

        # Set environment for server
        env = os.environ.copy()
        env["DOCUMENT_ROOT_DIR"] = str(self.temp_docs_root)
        env["PYTHONPATH"] = "."

        # Start server process
        cmd = [
            sys.executable,
            "-m",
            "document_mcp.doc_tool_server",
            "sse",
            "--host",
            "localhost",
            "--port",
            str(self.port),
        ]

        print(f"🚀 Starting MCP server on port {self.port}...")
        self.process = subprocess.Popen(
            cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        # Wait for server to be ready
        self._wait_for_server()

    def _wait_for_server(self, timeout=15):
        """Wait for server to be ready."""
        print("⏳ Waiting for MCP server to be ready...")
        start_time = time.time()

        while time.time() - start_time < timeout:
            if self.process.poll() is not None:
                # Process has terminated
                stdout, stderr = self.process.communicate()
                raise RuntimeError(
                    f"MCP server failed to start:\nSTDOUT: {stdout}\nSTDERR: {stderr}"
                )

            try:
                # Try to connect (simple approach - check if port is in use)
                import socket

                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex(("localhost", self.port))
                sock.close()

                if result == 0:  # Connection successful
                    print("✅ MCP server is ready!")
                    return

            except Exception:
                pass

            time.sleep(0.5)

        raise TimeoutError(f"MCP server did not start within {timeout} seconds")

    def stop_server(self):
        """Stop the MCP server."""
        if self.process is not None:
            print("🛑 Stopping MCP server...")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)
            finally:
                self.process = None

        # Cleanup temp directory
        if self.temp_docs_root and self.temp_docs_root.exists():
            shutil.rmtree(self.temp_docs_root, ignore_errors=True)
            self.temp_docs_root = None


def needs_mcp_server(pytest_args):
    """Check if the test run needs an MCP server."""
    # Check if running integration or e2e tests
    for arg in pytest_args:
        if "integration" in arg or "e2e" in arg:
            return True
        if arg.startswith("tests/integration") or arg.startswith("tests/e2e"):
            return True
        if "test_react_agent" in arg or "test_simple_agent" in arg:
            return True

    # If no specific test path specified, assume we might need it
    test_paths = [
        arg
        for arg in pytest_args
        if arg.startswith("tests/") and not arg.startswith("-")
    ]
    if not test_paths:
        return True  # Running all tests (includes E2E tests in CI/CD)

    return False


def run_pytest_tests():
    """Run pytest tests with proper configuration and MCP server management."""
    print("Document MCP - Pytest Test Runner")
    print("=" * 40)

    # Parse command line arguments to pass through to pytest
    pytest_args = sys.argv[1:] if len(sys.argv) > 1 else ["-v", "--tb=short"]

    # Check if we need to start MCP server
    server_manager = None
    if needs_mcp_server(pytest_args):
        server_manager = MCPServerManager()
        try:
            server_manager.start_server()
        except Exception as e:
            print(f"❌ Failed to start MCP server: {e}")
            print("⚠️  Integration tests may fail. Running tests anyway...")

    try:
        # Set environment for testing
        env = os.environ.copy()
        env["PYTHONPATH"] = "."

        # Base pytest command
        base_cmd = [sys.executable, "-m", "pytest"]

        # Add the passed arguments
        full_cmd = base_cmd + pytest_args

        print(f"\n🧪 Running tests with command: {' '.join(full_cmd)}")
        print("-" * 40)

        # Run pytest with all arguments
        result = subprocess.run(full_cmd, env=env)

        # Summary
        print("\n" + "=" * 40)
        if result.returncode == 0:
            print("✅ All tests passed!")
            return 0
        else:
            print("❌ Some tests failed.")
            return 1

    finally:
        # Always cleanup server
        if server_manager:
            server_manager.stop_server()


if __name__ == "__main__":
    exit_code = run_pytest_tests()
    sys.exit(exit_code)
