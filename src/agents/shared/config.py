"""Shared configuration for agents using centralized configuration."""


from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.models.openai import OpenAIModel

from document_mcp.config import get_settings

# Get centralized settings
settings = get_settings()

# Shared constants from centralized config
MCP_SERVER_CMD = settings.mcp_server_cmd
DEFAULT_TIMEOUT = settings.default_timeout
MAX_RETRIES = settings.max_retries


AgentSettings = type(settings)

def get_agent_settings():
    """Get the centralized settings instance."""
    return settings


def prepare_mcp_server_environment() -> dict[str, str]:
    """Prepare environment variables for MCP server subprocess.

    This ensures that API keys from .env file are properly passed to the
    MCP server subprocess, which runs in a separate process.

    Returns:
        dict: Environment variables for MCP server subprocess
    """
    return settings.get_mcp_server_environment()


async def load_llm_config():
    """Load and configure the LLM model based on available environment variables."""
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
