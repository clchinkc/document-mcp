"""Centralized configuration management for the Document MCP system.

This module provides a single source of truth for all configuration
including environment variables, file paths, timeouts, and other settings.
"""

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
        default=".documents_storage", description="Root directory for document storage"
    )

    # === API Keys ===
    openai_api_key: str | None = Field(default=None, description="OpenAI API key")
    gemini_api_key: str | None = Field(default=None, description="Google Gemini API key")

    # === Model Configuration ===
    openai_model_name: str = Field(default="gpt-4.1-mini", description="OpenAI model name")
    gemini_model_name: str = Field(default="gemini-2.5-flash", description="Gemini model name")

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
    def active_provider(self) -> Literal["openai", "gemini"] | None:
        """Get the active provider (OpenAI takes precedence)."""
        if self.openai_configured:
            return "openai"
        elif self.gemini_configured:
            return "gemini"
        return None

    @property
    def active_model(self) -> str | None:
        """Get the active model name."""
        if self.active_provider == "openai":
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

        # CRITICAL: Always pass DOCUMENT_ROOT_DIR if it's in the environment
        # This is essential for test isolation and Windows CI compatibility
        # Debug: Always log environment variable handling for CI debugging
        env_var_present = "DOCUMENT_ROOT_DIR" in os.environ
        print(f"[MCP_ENV_DEBUG] DOCUMENT_ROOT_DIR in os.environ: {env_var_present}")
        if env_var_present:
            env_value = os.environ["DOCUMENT_ROOT_DIR"]
            server_env["DOCUMENT_ROOT_DIR"] = env_value
            server_env["PYTEST_CURRENT_TEST"] = "1"  # Also set test flag when DOCUMENT_ROOT_DIR is present
            print(f"[MCP_ENV_DEBUG] Added DOCUMENT_ROOT_DIR to MCP env: {env_value}")
        else:
            print("[MCP_ENV_DEBUG] DOCUMENT_ROOT_DIR not in os.environ")
            # Add test mode flag if in test environment (for other test scenarios)
            if self.is_test_environment:
                server_env["PYTEST_CURRENT_TEST"] = "1"
                if self.document_root_dir != ".documents_storage":
                    server_env["DOCUMENT_ROOT_DIR"] = self.document_root_dir
                    print(f"[MCP_ENV_DEBUG] Added DOCUMENT_ROOT_DIR from settings: {self.document_root_dir}")

        # Debug: Show final MCP server environment subset
        doc_root_final = server_env.get("DOCUMENT_ROOT_DIR", "NOT_SET")
        pytest_flag = server_env.get("PYTEST_CURRENT_TEST", "NOT_SET")
        print(f"[MCP_ENV_DEBUG] Final MCP server env - DOCUMENT_ROOT_DIR: {doc_root_final}")
        print(f"[MCP_ENV_DEBUG] Final MCP server env - PYTEST_CURRENT_TEST: {pytest_flag}")

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
