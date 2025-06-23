"""
MCP Server Manager for testing.

This module provides a single, robust server management solution
that replaces the duplicate server managers in agent test files.
"""

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import requests


def _get_worker_port(base_port: int = 3001) -> int:
    """
    Get a unique port for this pytest worker to avoid conflicts.

    Args:
        base_port: Base port number to start from

    Returns:
        Unique port number for this worker
    """
    try:
        # Check if running with pytest-xdist
        worker_id = os.environ.get("PYTEST_XDIST_WORKER", "master")
        if worker_id == "master":
            return base_port
        else:
            # Extract worker number (e.g., 'gw0' -> 0, 'gw1' -> 1)
            worker_num = int(worker_id.replace("gw", ""))
            return base_port + worker_num + 1
    except (ValueError, TypeError):
        # Fallback to base port if parsing fails
        return base_port


class MCPServerManager:
    """
    MCP server manager for all testing scenarios.

    This replaces the duplicate MCPServerManager and ReactMCPServerManager
    classes with a single, robust implementation.
    """

    def __init__(
        self,
        test_docs_root: Optional[Path] = None,
        port: Optional[int] = None,
        host: str = "localhost",
    ):
        """
        Initialize the server manager.

        Args:
            test_docs_root: Root directory for test documents
            port: Port number (auto-assigned if None)
            host: Host address
        """
        self.port = port or _get_worker_port()
        self.host = host
        self.test_docs_root = test_docs_root
        self.process = None
        self.server_url = f"http://{self.host}:{self.port}"

    def start_server(self, timeout: int = 30) -> None:
        """
        Start the MCP server process.

        Args:
            timeout: Maximum time to wait for server startup

        Raises:
            TimeoutError: If server doesn't start within timeout
            RuntimeError: If server fails to start
        """
        if self.process is not None:
            return  # Already started

        # Set environment for the server process
        env = os.environ.copy()
        if self.test_docs_root:
            env["DOCUMENT_ROOT_DIR"] = str(self.test_docs_root)

        # Start the server process
        cmd = [
            sys.executable,
            "-m",
            "document_mcp.doc_tool_server",
            "sse",
            "--host",
            self.host,
            "--port",
            str(self.port),
        ]

        try:
            self.process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            # Wait for server to be ready
            self._wait_for_server(timeout)

        except Exception as e:
            self.stop_server()
            raise RuntimeError(f"Failed to start MCP server: {e}")

    def _wait_for_server(self, timeout: int) -> None:
        """
        Wait for the server to be ready to accept connections.

        Args:
            timeout: Maximum time to wait

        Raises:
            TimeoutError: If server doesn't respond within timeout
        """
        print(f"Waiting for MCP server at {self.server_url}")
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.server_url}/health", timeout=2)
                if response.status_code == 200:
                    print("MCP server ready")
                    return
            except requests.exceptions.RequestException:
                pass

            # Check if process has failed
            if self.process and self.process.poll() is not None:
                stdout, stderr = self.process.communicate()
                raise RuntimeError(
                    f"MCP server process failed. "
                    f"Return code: {self.process.returncode}, "
                    f"stderr: {stderr}"
                )

            time.sleep(1.0)

        raise TimeoutError(f"MCP server did not start within {timeout} seconds")

    def stop_server(self) -> None:
        """Stop the MCP server process gracefully."""
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass  # Process is stuck, nothing more we can do
            finally:
                self.process = None

    def is_running(self) -> bool:
        """
        Check if the server is currently running.

        Returns:
            True if server is running, False otherwise
        """
        if self.process is None:
            return False

        # Check if process is still alive
        if self.process.poll() is not None:
            return False

        # Check if server responds to health check
        try:
            response = requests.get(f"{self.server_url}/health", timeout=1)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def restart_server(self, timeout: int = 30) -> None:
        """
        Restart the server.

        Args:
            timeout: Maximum time to wait for restart
        """
        self.stop_server()
        self.start_server(timeout)

    def get_server_url(self) -> str:
        """Get the server URL."""
        return self.server_url

    def get_port(self) -> int:
        """Get the server port."""
        return self.port

    def __enter__(self):
        """Context manager entry."""
        self.start_server()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_server()
