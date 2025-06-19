import pytest
import os

# Basic environment setup for tests  
@pytest.fixture(autouse=True)
def ensure_api_keys():
    """Ensure API keys are available for testing."""
    api_keys = ["OPENAI_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"]
    has_api_key = any(os.environ.get(key) for key in api_keys)
    if not has_api_key:
        # Set a test placeholder for CI/CD environments
        os.environ["GEMINI_API_KEY"] = "test_api_key_placeholder" 