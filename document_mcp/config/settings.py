"""Centralized configuration management for the Document MCP system.

This module provides a single source of truth for all configuration
including environment variables, file paths, timeouts, and other settings.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field
from pydantic import model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Centralized settings for the Document MCP system."""

    # === Document Storage Configuration ===
    document_root_dir: str = Field(
        default=".documents_storage", description="Root directory for document storage (local)"
    )
    storage_backend: str = Field(default="auto", description="Storage backend: 'auto', 'local', or 'gcs'")
    gcs_bucket: str | None = Field(default=None, description="GCS bucket name for cloud storage")
    gcs_prefix: str = Field(default="", description="Path prefix within GCS bucket")

    # === API Keys ===
    openai_api_key: str | None = Field(default=None, description="OpenAI API key")
    gemini_api_key: str | None = Field(default=None, description="Google Gemini API key")
    openrouter_api_key: str | None = Field(default=None, description="OpenRouter API key")

    # === Model Configuration ===
    openai_model_name: str = Field(default="gpt-4.1-mini", description="OpenAI model name")
    gemini_model_name: str = Field(default="gemini-2.5-flash", description="Gemini model name")
    openrouter_model_name: str = Field(default="openai/gpt-5-mini", description="OpenRouter model name")

    # === MCP Server Configuration ===
    mcp_server_cmd: list[str] = Field(
        default_factory=lambda: [sys.executable, "-m", "document_mcp.doc_tool_server", "stdio"],
        description="MCP server command",
    )

    # === Timeout Configuration ===
    default_timeout: float = Field(default=60.0, description="Default timeout for operations")
    max_retries: int = Field(default=3, description="Maximum number of retries")

    # === Test Environment Detection ===
    pytest_current_test: str | None = Field(default=None, description="Test mode indicator")

    # === HTTP SSE Server Configuration ===
    sse_host: str = Field(default="0.0.0.0", description="SSE server host")
    sse_port: int = Field(default=8000, description="SSE server port")

    # === Logging Configuration ===
    log_level: str = Field(default="INFO", description="Logging level")
    structured_logging: bool = Field(default=True, description="Enable structured JSON logging")

    # === Performance Configuration ===
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")
    metrics_port: int = Field(default=8001, description="Metrics server port")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",  # Ignore extra environment variables
    }

    def __init__(self, **kwargs):
        """Initialize settings with environment-specific adjustments."""
        super().__init__(**kwargs)
        self._adjust_for_test_environment()

    def _adjust_for_test_environment(self):
        """Adjust settings for test environment."""
        if self.is_test_environment:
            # Use shorter timeouts in test environments
            self.default_timeout = 30.0
            self.max_retries = 2
            # Override document root if specified
            if "DOCUMENT_ROOT_DIR" in os.environ:
                self.document_root_dir = os.environ["DOCUMENT_ROOT_DIR"]

    @model_validator(mode="after")
    def validate_configuration(self):
        """Validate configuration consistency."""
        # API keys are optional during initialization
        return self

    @property
    def is_test_environment(self) -> bool:
        """Check if running in test environment."""
        return (
            "PYTEST_CURRENT_TEST" in os.environ
            or self.pytest_current_test is not None
            or "DOCUMENT_ROOT_DIR" in os.environ
        )

    @property
    def is_cloud_environment(self) -> bool:
        """Check if running in GCP cloud environment (Cloud Run)."""
        return "K_SERVICE" in os.environ

    @property
    def storage_type(self) -> Literal["local", "gcs"]:
        """Determine the active storage backend type."""
        if self.storage_backend == "gcs":
            return "gcs"
        elif self.storage_backend == "local":
            return "local"
        # Auto-detect
        if self.is_cloud_environment and self.gcs_bucket:
            return "gcs"
        if self.gcs_bucket:
            return "gcs"
        return "local"

    @property
    def document_root_path(self) -> Path:
        """Get the document root path as a Path object."""
        path = Path(self.document_root_dir).resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def openai_configured(self) -> bool:
        """Check if OpenAI is properly configured."""
        return bool(self.openai_api_key and self.openai_api_key.strip())

    @property
    def gemini_configured(self) -> bool:
        """Check if Gemini is properly configured."""
        return bool(self.gemini_api_key and self.gemini_api_key.strip())

    @property
    def openrouter_configured(self) -> bool:
        """Check if OpenRouter is properly configured."""
        return bool(self.openrouter_api_key and self.openrouter_api_key.strip())

    @property
    def active_provider(self) -> Literal["openai", "gemini", "openrouter"] | None:
        """Get the active provider (OpenRouter > OpenAI > Gemini precedence)."""
        if self.openrouter_configured:
            return "openrouter"
        elif self.openai_configured:
            return "openai"
        elif self.gemini_configured:
            return "gemini"
        return None

    @property
    def active_model(self) -> str | None:
        """Get the active model name."""
        if self.active_provider == "openrouter":
            return self.openrouter_model_name
        elif self.active_provider == "openai":
            return self.openai_model_name
        elif self.active_provider == "gemini":
            return self.gemini_model_name
        return None

    def get_mcp_server_environment(self) -> dict[str, str]:
        """Get environment variables for MCP server subprocess."""
        # Start with current environment
        server_env = {**os.environ}

        # Add API keys if available
        if self.gemini_api_key:
            server_env["GEMINI_API_KEY"] = self.gemini_api_key
        if self.openai_api_key:
            server_env["OPENAI_API_KEY"] = self.openai_api_key

        # Pass storage configuration
        server_env["STORAGE_BACKEND"] = self.storage_backend
        if self.gcs_bucket:
            server_env["GCS_BUCKET"] = self.gcs_bucket
        if self.gcs_prefix:
            server_env["GCS_PREFIX"] = self.gcs_prefix

        # Pass DOCUMENT_ROOT_DIR for test isolation
        if "DOCUMENT_ROOT_DIR" in os.environ:
            server_env["DOCUMENT_ROOT_DIR"] = os.environ["DOCUMENT_ROOT_DIR"]
            server_env["PYTEST_CURRENT_TEST"] = "1"
        elif self.is_test_environment:
            server_env["PYTEST_CURRENT_TEST"] = "1"
            if self.document_root_dir != ".documents_storage":
                server_env["DOCUMENT_ROOT_DIR"] = self.document_root_dir

        return server_env


# Global settings instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get the global settings instance (singleton pattern)."""
    global _settings
    if _settings is None:
        load_dotenv()  # Load .env file
        _settings = Settings()
    else:
        # In test environment, refresh document root if changed
        if "DOCUMENT_ROOT_DIR" in os.environ:
            current_root = os.environ["DOCUMENT_ROOT_DIR"]
            if _settings.document_root_dir != current_root:
                _settings.document_root_dir = current_root
    return _settings


def reset_settings():
    """Reset the global settings instance (primarily for testing)."""
    global _settings
    _settings = None
