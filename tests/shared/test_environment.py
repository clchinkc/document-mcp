"""Test environment management utilities for Document MCP testing.

This module provides centralized environment management to eliminate
complex environment manipulation patterns repeated across test files.
"""
from __future__ import annotations


import importlib
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path


class EnvironmentManager:
    """Manages test environment setup, isolation, and cleanup.

    Provides context managers for safe environment manipulation
    and automatic cleanup of environment variables and module reloads.
    """

    def __init__(self):
        """Initialize the test environment manager."""
        self._environment_backup: dict[str, str | None] = {}
        self._modules_to_reload = ["document_mcp.doc_tool_server"]

    def backup_environment_variable(self, key: str) -> None:
        """Backup an environment variable for later restoration."""
        self._environment_backup[key] = os.environ.get(key)

    def set_environment_variable(self, key: str, value: str) -> None:
        """Set an environment variable with automatic backup."""
        if key not in self._environment_backup:
            self.backup_environment_variable(key)
        os.environ[key] = value

    def remove_environment_variable(self, key: str) -> None:
        """Remove an environment variable with automatic backup."""
        if key not in self._environment_backup:
            self.backup_environment_variable(key)
        if key in os.environ:
            del os.environ[key]

    def restore_environment(self) -> None:
        """Restore all backed up environment variables."""
        for key, value in self._environment_backup.items():
            if value is not None:
                os.environ[key] = value
            elif key in os.environ:
                del os.environ[key]
        self._environment_backup.clear()

    def reload_modules(self) -> None:
        """Reload specified modules to pick up environment changes."""
        for module_name in self._modules_to_reload:
            if module_name in sys.modules:
                importlib.reload(sys.modules[module_name])

    @contextmanager
    def isolated_environment(self, **env_vars):
        """Context manager for isolated environment with automatic cleanup.

        Args:
            **env_vars: Environment variables to set during the context

        Example:
            with env_manager.isolated_environment(DOCUMENT_ROOT_DIR="/tmp/test"):
                # Test code here with modified environment
                pass
            # Environment automatically restored
        """
        # Backup and set environment variables
        for key, value in env_vars.items():
            if value is None:
                self.remove_environment_variable(key)
            else:
                self.set_environment_variable(key, str(value))

        # Reload modules to pick up changes
        self.reload_modules()

        try:
            yield
        finally:
            # Restore environment and reload modules
            self.restore_environment()
            self.reload_modules()

    @contextmanager
    def mock_api_environment(self, provider: str = "openai"):
        """Context manager for mock API environment setup.

        Args:
            provider: API provider to mock ("openai" or "gemini")
        """
        if provider == "openai":
            env_vars = {
                "OPENAI_API_KEY": "sk-test-key-for-integration-testing",
                "OPENAI_MODEL_NAME": "gpt-4.1-mini",
                "GEMINI_API_KEY": None,  # Remove to force OpenAI usage
            }
        elif provider == "gemini":
            env_vars = {
                "GEMINI_API_KEY": "test-gemini-key-for-integration-testing",
                "OPENAI_API_KEY": None,  # Remove to force Gemini usage
            }
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        with self.isolated_environment(**env_vars):
            yield


class TemporaryDocumentRoot:
    """Manages temporary document root directories for testing.

    Provides automatic cleanup and environment variable management
    for document storage testing.
    """

    def __init__(self, env_manager: EnvironmentManager | None = None):
        """Initialize with optional environment manager."""
        self.env_manager = env_manager or EnvironmentManager()
        self.temp_dir: tempfile.TemporaryDirectory | None = None
        self.path: Path | None = None

    def __enter__(self) -> Path:
        """Enter context and create temporary directory."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = Path(self.temp_dir.name)

        # Set up environment for the temporary directory
        self.env_manager.set_environment_variable("DOCUMENT_ROOT_DIR", str(self.path))
        self.env_manager.reload_modules()

        return self.path

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and cleanup."""
        # Restore environment
        self.env_manager.restore_environment()
        self.env_manager.reload_modules()

        # Cleanup temporary directory
        if self.temp_dir:
            self.temp_dir.cleanup()
            self.temp_dir = None
            self.path = None


def check_api_key_available() -> bool:
    """Check if a real API key is available for testing.

    Returns:
        True if a real API key is found, False otherwise
    """
    api_keys = ["OPENAI_API_KEY", "GEMINI_API_KEY"]
    for key in api_keys:
        value = os.environ.get(key, "").strip()
        if value and value != "test_key" and not value.startswith("sk-test"):
            return True
    return False


def get_test_api_keys() -> dict[str, str | None]:
    """Get available test API keys with validation.

    Returns:
        Dictionary of available API keys with provider names as keys
    """
    keys = {}

    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if openai_key and not openai_key.startswith("sk-test"):
        keys["openai"] = openai_key

    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if gemini_key and gemini_key != "test_key":
        keys["gemini"] = gemini_key

    return keys


# Global instance for convenience
default_env_manager = EnvironmentManager()
