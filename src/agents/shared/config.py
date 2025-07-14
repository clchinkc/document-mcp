"""
Shared configuration for agents using Pydantic Settings for validation.
"""

from typing import Literal, Optional

from dotenv import load_dotenv
from pydantic import Field, model_validator
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.models.openai import OpenAIModel
from pydantic_settings import BaseSettings

# --- Shared Constants ---
MCP_SERVER_CMD = ["python3", "-m", "document_mcp.doc_tool_server", "stdio"]
DEFAULT_TIMEOUT = 60.0
MAX_RETRIES = 3


class AgentSettings(BaseSettings):
    """Pydantic Settings model for agent configuration with validation."""

    # API Keys
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    gemini_api_key: Optional[str] = Field(
        default=None, description="Google Gemini API key"
    )

    # Model Names
    openai_model_name: str = Field(
        default="gpt-4.1-mini", description="OpenAI model name"
    )
    gemini_model_name: str = Field(
        default="gemini-2.5-flash", description="Gemini model name"
    )

    # Document Root
    document_root_dir: Optional[str] = Field(
        default=None, description="Document storage root directory"
    )

    # Test Mode
    pytest_current_test: Optional[str] = Field(
        default=None, description="Test mode indicator"
    )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }

    @model_validator(mode="after")
    def validate_api_keys(self):
        """Ensure at least one API key is provided."""
        if not self.openai_api_key and not self.gemini_api_key:
            # Don't raise error during initialization - let the application handle it
            pass
        return self

    @property
    def openai_configured(self) -> bool:
        """Check if OpenAI is properly configured."""
        return bool(self.openai_api_key and self.openai_api_key.strip())

    @property
    def gemini_configured(self) -> bool:
        """Check if Gemini is properly configured."""
        return bool(self.gemini_api_key and self.gemini_api_key.strip())

    @property
    def active_provider(self) -> Optional[Literal["openai", "gemini"]]:
        """Get the active provider (OpenAI takes precedence)."""
        if self.openai_configured:
            return "openai"
        elif self.gemini_configured:
            return "gemini"
        return None

    @property
    def active_model(self) -> Optional[str]:
        """Get the active model name."""
        if self.active_provider == "openai":
            return self.openai_model_name
        elif self.active_provider == "gemini":
            return self.gemini_model_name
        return None



def get_settings() -> AgentSettings:
    """Get the validated settings instance."""
    load_dotenv()  # Load .env file
    return AgentSettings()




async def load_llm_config():
    """Load and configure the LLM model based on available environment variables."""
    settings = get_settings()

    if settings.active_provider == "openai":
        print(f"Using OpenAI model: {settings.active_model}")
        return OpenAIModel(model_name=settings.active_model)

    if settings.active_provider == "gemini":
        print(f"Using Gemini model: {settings.active_model}")
        return GeminiModel(model_name=settings.active_model)

    # If no API keys are found, raise an error with helpful message
    raise ValueError(
        "No valid API key found in environment variables. "
        "Please set one of the following in your .env file:\n"
        "- OPENAI_API_KEY for OpenAI models\n"
        "- GEMINI_API_KEY for Google Gemini models\n"
        "\nOptionally, you can also set:\n"
        "- OPENAI_MODEL_NAME (default: gpt-4.1-mini)\n"
        "- GEMINI_MODEL_NAME (default: gemini-2.5-flash)"
    )
