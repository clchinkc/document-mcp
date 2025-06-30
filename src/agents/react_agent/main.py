#!/usr/bin/env python3
"""
ReAct Document Management Agent

This module implements a ReAct (Reasoning and Acting) agent that can manage
structured markdown documents through systematic reasoning and tool execution.
"""

import argparse
import asyncio
import logging
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.models.openai import OpenAIModel
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from pathlib import Path

# Remove invalid imports
# from .react_agent import ReactAgent
# from .react_prompts import REACT_SYSTEM_PROMPT

# Import from simple agent for consistency
from src.agents.simple_agent import load_llm_config

# Import from the installed document_mcp package for direct manipulation
try:
    from document_mcp import doc_tool_server
except ImportError:
    # Fallback for development/testing without installed package
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from document_mcp import doc_tool_server

# Suppress verbose HTTP logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)

# Import models from the server to ensure compatibility

# Circuit breaker imports - conditional import moved to top level
try:
    CIRCUIT_BREAKER_AVAILABLE = True
except ImportError:
    # Fallback if circuit breaker not available
    CIRCUIT_BREAKER_AVAILABLE = False


# --- Error Handling and Circuit Breaker Classes (moved before usage) ---


class ErrorType(Enum):
    NETWORK_ERROR = "network"
    AUTHENTICATION_ERROR = "auth"
    RATE_LIMIT_ERROR = "rate_limit"
    VALIDATION_ERROR = "validation"
    TOOL_ERROR = "tool"
    LLM_ERROR = "llm"
    CONFIGURATION_ERROR = "config"
    RESOURCE_ERROR = "resource"
    UNKNOWN_ERROR = "unknown"


@dataclass
class RetryConfig:
    max_retries: int
    initial_delay: float = 1.0
    max_delay: float = 30.0


@dataclass
class ErrorInfo:
    error_type: ErrorType
    is_retryable: bool
    max_retries: int
    initial_delay: float
    max_delay: float
    severity: str  # "low", "medium", "high", "critical"
    recovery_action: str
    user_message: str


class ErrorClassifier:
    """Classifies errors and provides retry strategies."""

    def classify(self, error: Exception) -> ErrorInfo:
        """Classify an error and return appropriate retry information."""
        error_str = str(error).lower()

        # Network and connection errors
        if any(
            keyword in error_str
            for keyword in [
                "connection",
                "timeout",
                "network",
                "unreachable",
                "502",
                "503",
                "504",
            ]
        ):
            return ErrorInfo(
                error_type=ErrorType.NETWORK_ERROR,
                is_retryable=True,
                max_retries=3,
                initial_delay=1.0,
                max_delay=16.0,
                severity="medium",
                recovery_action="exponential_backoff",
                user_message="Network connectivity issue detected. Retrying...",
            )

        # Authentication errors
        if any(
            keyword in error_str
            for keyword in ["api key", "authentication", "unauthorized", "401"]
        ):
            return ErrorInfo(
                error_type=ErrorType.AUTHENTICATION_ERROR,
                is_retryable=False,
                max_retries=0,
                initial_delay=0.0,
                max_delay=0.0,
                severity="high",
                recovery_action="check_config",
                user_message="Authentication error. Please check your API key configuration.",
            )

        # Rate limiting
        if any(
            keyword in error_str
            for keyword in ["rate limit", "quota", "too many requests", "429"]
        ):
            return ErrorInfo(
                error_type=ErrorType.RATE_LIMIT_ERROR,
                is_retryable=True,
                max_retries=5,
                initial_delay=2.0,
                max_delay=60.0,
                severity="medium",
                recovery_action="exponential_backoff",
                user_message="API rate limit reached. Waiting before retry...",
            )

        # Validation errors
        if any(
            keyword in error_str
            for keyword in ["invalid", "validation", "format", "parse"]
        ):
            return ErrorInfo(
                error_type=ErrorType.VALIDATION_ERROR,
                is_retryable=False,
                max_retries=0,
                initial_delay=0.0,
                max_delay=0.0,
                severity="low",
                recovery_action="user_feedback",
                user_message="Input validation error. Please check the format of your request.",
            )

        # Tool execution errors
        if any(keyword in error_str for keyword in ["tool", "mcp", "execution"]):
            return ErrorInfo(
                error_type=ErrorType.TOOL_ERROR,
                is_retryable=True,
                max_retries=2,
                initial_delay=0.5,
                max_delay=4.0,
                severity="medium",
                recovery_action="retry_simplified",
                user_message="Tool execution error. Attempting recovery...",
            )

        # LLM errors
        if any(
            keyword in error_str
            for keyword in ["llm", "model", "generation", "completion"]
        ):
            return ErrorInfo(
                error_type=ErrorType.LLM_ERROR,
                is_retryable=True,
                max_retries=2,
                initial_delay=1.0,
                max_delay=8.0,
                severity="high",
                recovery_action="retry_with_fallback",
                user_message="AI service error. Retrying with fallback strategy...",
            )

        # Default: Unknown error
        return ErrorInfo(
            error_type=ErrorType.UNKNOWN_ERROR,
            is_retryable=True,
            max_retries=1,
            initial_delay=1.0,
            max_delay=4.0,
            severity="medium",
            recovery_action="basic_retry",
            user_message="Unexpected error encountered. Attempting recovery...",
        )


class ServiceCircuitBreaker:
    """Circuit breaker implementation for external service calls."""

    def __init__(
        self,
        service_name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
    ):
        self.service_name = service_name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.success_count = 0

    async def call(self, func, *args, **kwargs):
        """Execute a function call with circuit breaker protection."""
        if self.state == "OPEN":
            if time.time() - (self.last_failure_time or 0) > self.recovery_timeout:
                self.state = "HALF_OPEN"
                self.success_count = 0
            else:
                raise Exception(f"Circuit breaker OPEN for {self.service_name}")

        try:
            result = await func(*args, **kwargs)
            self._record_success()
            return result
        except Exception as e:
            self._record_failure()
            raise e

    def _record_success(self):
        if self.state == "HALF_OPEN":
            self.success_count += 1
            if self.success_count >= 2:  # Require 2 successes to close
                self.state = "CLOSED"
                self.failure_count = 0
        elif self.state == "CLOSED":
            self.failure_count = max(
                0, self.failure_count - 1
            )  # Gradually reduce failure count

    def _record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"


class RetryManager:
    """Manages retry logic with exponential backoff and jitter."""

    def __init__(self):
        self.error_classifier = ErrorClassifier()

    async def execute_with_retry(self, func, *args, **kwargs):
        """Execute a function with intelligent retry logic."""
        last_error = None

        for attempt in range(1, 6):  # Max 5 attempts total
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_error = e
                error_info = self.error_classifier.classify(e)

                if not error_info.is_retryable or attempt > error_info.max_retries:
                    print(
                        f"Final failure after {attempt} attempts: {error_info.user_message}"
                    )
                    raise e

                delay = self._calculate_delay(
                    attempt, error_info.initial_delay, error_info.max_delay
                )
                print(f"Attempt {attempt} failed: {error_info.user_message}")
                print(f"Retrying in {delay:.1f}s...")
                await asyncio.sleep(delay)

        # Should never reach here, but just in case
        raise last_error

    def _calculate_delay(
        self, attempt: int, initial_delay: float, max_delay: float
    ) -> float:
        """Calculate delay with exponential backoff and jitter."""
        base_delay = initial_delay * (2 ** (attempt - 1))
        delay = min(base_delay, max_delay)
        # Add jitter (±25%)
        jitter = delay * 0.25 * random.uniform(-1, 1)
        return max(0.1, delay + jitter)


# Global instances for error handling
_circuit_breakers: Dict[str, ServiceCircuitBreaker] = {}
_retry_manager = RetryManager()


def get_circuit_breaker(service_name: str) -> ServiceCircuitBreaker:
    """Get or create a circuit breaker for a service."""
    if service_name not in _circuit_breakers:
        _circuit_breakers[service_name] = ServiceCircuitBreaker(service_name)
    return _circuit_breakers[service_name]


# --- ReAct Models ---


class ReActStep(BaseModel):
    """A single step in the ReAct process, containing thought and action."""

    thought: str = Field(
        min_length=1, description="The agent's reasoning and plan for the next action."
    )
    action: Optional[str] = Field(
        default=None,
        description="The tool call to execute, or null if reasoning is complete.",
    )


# --- Configuration ---
async def load_llm_config():
    """Load and configure the LLM model with circuit breaker protection."""
    load_dotenv()

    async def _load_openai_model():
        """Internal function to load OpenAI model."""
        model_name = os.environ.get("OPENAI_MODEL_NAME", "gpt-4.1-mini")
        print(f"Using OpenAI model: {model_name}")
        return OpenAIModel(model_name=model_name)

    async def _load_gemini_model():
        """Internal function to load Gemini model."""
        model_name = os.environ.get("GEMINI_MODEL_NAME", "gemini-2.5-flash")
        print(f"Using Gemini model: {model_name}")
        return GeminiModel(model_name=model_name)

    # Check for OpenAI API key first
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if openai_api_key and openai_api_key.strip():
        try:
            # Use circuit breaker for OpenAI model loading in non-test environments
            if not os.environ.get("PYTEST_CURRENT_TEST"):
                circuit_breaker = get_circuit_breaker("openai")
                return await circuit_breaker.call(_load_openai_model)
            else:
                # For tests, load directly without circuit breaker
                return await _load_openai_model()
        except Exception as e:
            print(f"OpenAI model loading failed: {e}")
            print("Falling back to Gemini...")

    # Check for Gemini API keys
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if gemini_api_key and gemini_api_key.strip():
        try:
            # Use circuit breaker for Gemini model loading in non-test environments
            if not os.environ.get("PYTEST_CURRENT_TEST"):
                circuit_breaker = get_circuit_breaker("gemini")
                return await circuit_breaker.call(_load_gemini_model)
            else:
                # For tests, load directly without circuit breaker
                return await _load_gemini_model()
        except Exception as e:
            print(f"Gemini model loading failed: {e}")

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


def check_api_keys_config():
    """
    Utility function to check which API keys are configured.
    Returns a dictionary with configuration status.
    """
    load_dotenv()

    config_status = {
        "openai_configured": False,
        "gemini_configured": False,
        "active_provider": None,
        "active_model": None,
    }

    # Check OpenAI
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if openai_api_key and openai_api_key.strip():
        config_status["openai_configured"] = True
        config_status["openai_model"] = os.environ.get(
            "OPENAI_MODEL_NAME", "gpt-4.1-mini"
        )
        if not config_status["active_provider"]:  # First priority
            config_status["active_provider"] = "openai"
            config_status["active_model"] = config_status["openai_model"]

    # Check Gemini
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if gemini_api_key and gemini_api_key.strip():
        config_status["gemini_configured"] = True
        config_status["gemini_model"] = os.environ.get(
            "GEMINI_MODEL_NAME", "gemini-2.5-flash"
        )
        if not config_status["active_provider"]:  # Second priority
            config_status["active_provider"] = "gemini"
            config_status["active_model"] = config_status["gemini_model"]

    return config_status


# --- ReAct System Prompt ---
REACT_SYSTEM_PROMPT = """You are an assistant that uses a set of tools to manage documents. You will break down complex user requests into a sequence of steps using the ReAct (Reason, Act) pattern.

## ReAct Process Overview

You will follow a "Thought, Action, Observation" loop to complete tasks:

1. **Thought**: You reason about what you need to do next based on the user's request and previous observations.
2. **Action**: You choose **one** tool to execute. This should be a valid function call with proper parameters.
3. **Observation**: You will be given the result of the action to inform your next thought.

You must respond with a JSON object that matches this structure:
{
    "thought": "Your reasoning about what to do next",
    "action": "tool_name(param1=\"value1\", param2=\"value2\")"
}

When you have completed the user's request, provide a final thought and set the `action` to `null` or omit it entirely.

## Available Tools

You have access to these document management tools:

**Document Operations:**
- `create_document(document_name="My Book")` - Creates a new document directory
- `delete_document(document_name="My Book")` - Deletes an entire document
- `list_documents()` - Lists all available documents

**Chapter Operations:**
- `create_chapter(document_name="My Book", chapter_name="01-introduction.md", initial_content="# Introduction")` - Creates a new chapter
- `delete_chapter(document_name="My Book", chapter_name="01-introduction.md")` - Deletes a chapter
- `list_chapters(document_name="My Book")` - Lists all chapters in a document
- `write_chapter_content(document_name="My Book", chapter_name="01-intro.md", new_content="# New Content")` - Overwrites chapter content

**Reading Operations:**
- `read_document_summary(document_name="My Book")` - Reads the _SUMMARY.md file for a document. **Use this first before reading content**
- `read_chapter_content(document_name="My Book", chapter_name="01-intro.md")` - Reads a specific chapter
- `read_full_document(document_name="My Book")` - Reads all chapters of a document  
- `read_paragraph_content(document_name="My Book", chapter_name="01-intro.md", paragraph_index_in_chapter=0)` - Reads a specific paragraph

**Content Modification:**
- `append_paragraph_to_chapter(document_name="My Book", chapter_name="01-intro.md", paragraph_content="New paragraph.")` - Adds content to end of chapter
- `replace_paragraph(document_name="My Book", chapter_name="01-intro.md", paragraph_index=0, new_content="Updated text.")` - Replaces a specific paragraph
- `insert_paragraph_before(document_name="My Book", chapter_name="01-intro.md", paragraph_index=1, new_content="New paragraph.")` - Inserts paragraph before specified index
- `insert_paragraph_after(document_name="My Book", chapter_name="01-intro.md", paragraph_index=0, new_content="New paragraph.")` - Inserts paragraph after specified index
- `delete_paragraph(document_name="My Book", chapter_name="01-intro.md", paragraph_index=2)` - Deletes a specific paragraph
- `move_paragraph_before(document_name="My Book", chapter_name="01-intro.md", paragraph_to_move_index=3, target_paragraph_index=1)` - Moves paragraph to new position
- `move_paragraph_to_end(document_name="My Book", chapter_name="01-intro.md", paragraph_to_move_index=0)` - Moves paragraph to end
- `replace_text_in_chapter(document_name="My Book", chapter_name="01-intro.md", text_to_find="old", replacement_text="new")` - Find and replace in chapter
- `replace_text_in_document(document_name="My Book", text_to_find="old_term", replacement_text="new_term")` - Find and replace across document

**Analysis Operations:**
- `get_chapter_statistics(document_name="My Book", chapter_name="01-intro.md")` - Gets word/paragraph counts for a chapter
- `get_document_statistics(document_name="My Book")` - Gets aggregate statistics for entire document
- `find_text_in_chapter(document_name="My Book", chapter_name="01-intro.md", query="search term", case_sensitive=false)` - Searches within a chapter
- `find_text_in_document(document_name="My Book", query="search term", case_sensitive=false)` - Searches across entire document

## Example ReAct Sequence

User Query: "Create a document named 'Project Alpha' and add a chapter to it called 'introduction'."

**Step 1:**
{
    "thought": "The user wants to create a document and then add a chapter. I need to do this in two steps. First, I will create the document.",
    "action": "create_document(document_name=\"Project Alpha\")"
}

**Observation:** {"success": true, "message": "Document 'Project Alpha' created successfully."}

**Step 2:**
{
    "thought": "The document 'Project Alpha' has been created successfully. Now I need to add the 'introduction' chapter to it. The chapter name should be a valid filename with .md extension.",
    "action": "create_chapter(document_name=\"Project Alpha\", chapter_name=\"01-introduction.md\", initial_content=\"# Introduction\")"
}

**Observation:** {"success": true, "message": "Chapter '01-introduction.md' created successfully in document 'Project Alpha'.", "details": {"document_name": "Project Alpha", "chapter_name": "01-introduction.md"}}

**Step 3:**
{
    "thought": "I have successfully created the document 'Project Alpha' and added the 'introduction' chapter with basic markdown content. The task is now complete.",
    "action": null
}

## Important Guidelines

1. **One Action Per Step**: Execute only one tool call per response.
2. **Valid JSON**: Always respond with valid JSON matching the ReActStep structure.
3. **Parameter Formatting**: Use proper string quoting and escaping in action calls.
4. **Chapter Names**: Chapter filenames must end with `.md` and should follow naming conventions like `01-intro.md`, `02-content.md`.
5. **Error Handling**: If a tool returns an error, analyze the error message and adjust your approach in the next thought.
6. **Completion**: When the user's request is fully satisfied, set `action` to `null` to indicate completion.
7. **Step-by-Step**: Break complex requests into logical, sequential steps.
8. **Summary Operations**: 
   - **Explicit Content Requests**: When user explicitly asks to "read the content", "show me the content", "what's in the document" → Read content directly using `read_full_document()` or `read_chapter_content()`
   - **Broad Screening/Editing**: When user gives broad edit commands like "update the document", "modify this section", "improve the writing" → First use `read_document_summary()` to understand structure, then read specific content as needed
   - **General Inquiries**: For topics/questions about documents, check `list_documents` for `has_summary: true` and use `read_document_summary()` first
   - **After Write Operations**: Suggest creating/updating `_SUMMARY.md` files

Remember: Think clearly, act precisely, and observe carefully. Always prioritize summary-driven workflows to provide efficient and comprehensive document management."""

# --- ReAct Execution Loop ---


def parse_action_string(action_str: str) -> tuple[str, dict]:
    """
    Parse an action string to extract the tool name and arguments.

    Args:
        action_str: String like 'create_document(document_name="My Book")'

    Returns:
        tuple of (tool_name, kwargs_dict)

    Raises:
        ValueError: If the action string cannot be parsed
    """
    # Basic regex to extract function name and arguments
    pattern = r"^(\w+)\((.*)\)$"
    match = re.match(pattern, action_str.strip())

    if not match:
        raise ValueError(f"Invalid action format: {action_str}")

    tool_name = match.group(1)
    args_str = match.group(2).strip()

    # If no arguments, return empty dict
    if not args_str:
        return tool_name, {}

    kwargs = {}

    # Use a more sophisticated parsing approach
    import shlex

    try:
        # Try to use shlex for proper quote handling
        lexer = shlex.shlex(args_str, posix=True)
        lexer.whitespace_split = True
        lexer.commenters = ""
        lexer.whitespace = ","

        parts = list(lexer)

        for part in parts:
            part = part.strip()
            if "=" in part:
                key, value = part.split("=", 1)
                key = key.strip()
                value = value.strip()

                # Handle boolean and None values
                if value.lower() == "true":
                    value = True
                elif value.lower() == "false":
                    value = False
                elif value.lower() == "none":
                    value = None
                elif value.isdigit():
                    value = int(value)

                kwargs[key] = value

    except Exception:
        # Fallback to manual parsing
        # Split by comma but be aware of quotes
        in_quotes = False
        quote_char = None
        current_part = ""
        parts = []

        i = 0
        while i < len(args_str):
            char = args_str[i]

            if not in_quotes and char in ['"', "'"]:
                in_quotes = True
                quote_char = char
                current_part += char
            elif in_quotes and char == quote_char:
                in_quotes = False
                quote_char = None
                current_part += char
            elif not in_quotes and char == ",":
                parts.append(current_part.strip())
                current_part = ""
            else:
                current_part += char

            i += 1

        # Add the last part
        if current_part.strip():
            parts.append(current_part.strip())

        # Parse each part
        for part in parts:
            if "=" in part:
                key, value = part.split("=", 1)
                key = key.strip()
                value = value.strip()

                # Remove quotes if present
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]

                # Handle boolean and None values
                if value.lower() == "true":
                    value = True
                elif value.lower() == "false":
                    value = False
                elif value.lower() == "none":
                    value = None
                elif value.isdigit():
                    value = int(value)

                kwargs[key] = value

    return tool_name, kwargs


async def execute_mcp_tool_directly(agent: Agent, tool_name: str, kwargs: dict) -> str:
    """
    Execute an MCP tool efficiently with advanced error handling and retry logic.

    Args:
        agent: The pydantic-ai Agent with MCP servers configured
        tool_name: Name of the tool to execute
        kwargs: Dictionary of keyword arguments for the tool

    Returns:
        String representation of the tool execution result

    Raises:
        Exception: If tool execution fails after all retries
    """

    async def _execute_tool():
        """Internal function for actual tool execution."""
        # OPTIMIZED: Streamlined tool execution with minimal LLM overhead
        kwargs_str = ", ".join(
            f'{k}="{v}"' if isinstance(v, str) else f"{k}={v}"
            for k, v in kwargs.items()
        )

        # Use concise prompt format to minimize token usage
        tool_prompt = f"{tool_name}({kwargs_str})"

        # Execute through the agent with optimized prompt
        result = await agent.run(tool_prompt)

        # Streamlined result extraction
        if hasattr(result, "output") and result.output:
            if hasattr(result.output, "model_dump"):
                return str(result.output.model_dump())
            else:
                return str(result.output)
        else:
            return str(result)

    try:
        # Use circuit breaker and retry for tool execution in non-test environments
        if not os.environ.get("PYTEST_CURRENT_TEST"):
            circuit_breaker = get_circuit_breaker(f"mcp_tool_{tool_name}")
            return await _retry_manager.execute_with_retry(
                lambda: circuit_breaker.call(_execute_tool)
            )
        else:
            # For tests, execute directly without advanced error handling
            return await _execute_tool()

    except Exception as e:
        # Enhanced error reporting with classification in non-test environments
        if not os.environ.get("PYTEST_CURRENT_TEST"):
            error_classifier = ErrorClassifier()
            error_info = error_classifier.classify(e)
            print(f"🚨 Tool execution failed: {error_info.user_message}")
        raise Exception(f"Failed to execute tool {tool_name}: {str(e)}")


# --- Optimized History Context Builder ---
class HistoryContextBuilder:
    """Efficiently builds and manages ReAct loop history context."""

    def __init__(self):
        self.history_parts = []
        self._cached_context = ""
        self._context_dirty = False

    def add_step(self, step_data: dict):
        """Add a step to the history and mark context as dirty."""
        step_text = (
            f"**Previous Step {step_data['step']}:**\n"
            f"Thought: {step_data['thought']}\n"
            f"Action: {step_data['action']}\n"
            f"Observation: {step_data['observation']}"
        )
        self.history_parts.append(step_text)
        self._context_dirty = True

    def get_context(self) -> str:
        """Get the current history context, using cache when possible."""
        if self._context_dirty or not self._cached_context:
            self._cached_context = "\n\n".join(self.history_parts)
            self._context_dirty = False
        return self._cached_context

    def clear(self):
        """Clear the history context."""
        self.history_parts.clear()
        self._cached_context = ""
        self._context_dirty = False


# --- Global Agent Cache for Reuse ---
_agent_cache = {}


async def get_cached_agent(model_type: str, system_prompt: str, mcp_server) -> Agent:
    """
    Get or create a cached agent instance for performance optimization.

    OPTIMIZATION: This caches agent instances to avoid repeated initialization overhead.
    Uses a cache key based on model type and system prompt hash for uniqueness.
    """
    import hashlib

    cache_key = f"{model_type}_{hashlib.md5(system_prompt.encode()).hexdigest()[:8]}"

    if cache_key not in _agent_cache:
        model = await load_llm_config()
        _agent_cache[cache_key] = Agent(
            model=model,
            mcp_servers=[mcp_server],
            system_prompt=system_prompt,
            output_type=ReActStep,
        )

    return _agent_cache[cache_key]


async def run_react_loop(user_query: str, max_steps: int = 10) -> List[Dict[str, Any]]:
    """Main execution loop for the ReAct agent."""
    console = Console()
    history = []

    console.print(
        Panel(
            f"Starting ReAct loop for query: [bold cyan]{user_query}[/bold cyan]\nMaximum steps: {max_steps}",
            title="ReAct Agent",
            expand=False,
        )
    )

    # Connect to MCP server via stdio
    # Use python3 directly to avoid issues with sys.executable in virtual environments
    server_command = ["python3", "-m", "document_mcp.doc_tool_server", "stdio"]
    
    try:
        mcp_server = MCPServerStdio(
            command="python3",
            args=["-m", "document_mcp.doc_tool_server", "stdio"],
        )
    except Exception as e:
        console.print(f"[red]Failed to create MCP server: {e}[/red]")
        return [{"step": 1, "error": f"Failed to create MCP server: {e}"}]

    # Initialize ReAct Agent using the proper Agent class
    try:
        react_agent = await get_cached_agent("react", REACT_SYSTEM_PROMPT, mcp_server)
    except Exception as e:
        console.print(f"[red]Failed to initialize agent: {e}[/red]")
        return [{"step": 1, "error": f"Failed to initialize agent: {e}"}]

    # Set the document root directory for the server-side logic
    # This ensures consistency if the agent is run from a different CWD
    if "DOCUMENT_ROOT_DIR" in os.environ:
        doc_tool_server.DOCS_ROOT_PATH = os.environ["DOCUMENT_ROOT_DIR"]

    try:
        # Start the MCP server
        async with mcp_server:
            # Initialize context and step counter
            context_builder = HistoryContextBuilder()
            step = 0
            
            # Create initial context with user query
            current_context = f"User Query: {user_query}\n\nPlease provide your first thought and action."
            
            while step < max_steps:
                step += 1
                
                # Add history context if we have previous steps
                if step > 1:
                    current_context = f"User Query: {user_query}\n\n{context_builder.get_context()}\n\nPlease provide your next thought and action."
                
                try:
                    # Run the agent to get the next step
                    result = await react_agent.run(current_context)
                    
                    # Extract the ReActStep from the result
                    if hasattr(result, 'output'):
                        react_step = result.output
                    else:
                        react_step = result
                except Exception as e:
                    console.print(f"[red]Agent execution error: {e}[/red]")
                    step_data = {
                        "step": step,
                        "thought": f"Error occurred during agent execution: {str(e)}",
                        "action": None,
                        "observation": f"Error: {str(e)}"
                    }
                    history.append(step_data)
                    break
                
                # Prepare step data
                step_data = {
                    "step": step,
                    "thought": react_step.thought,
                    "action": react_step.action,
                    "observation": None
                }
                
                # If there's an action, execute it
                if react_step.action and react_step.action.strip():
                    try:
                        # Parse and execute the action
                        tool_name, kwargs = parse_action_string(react_step.action)
                        observation = await execute_mcp_tool_directly(react_agent, tool_name, kwargs)
                        step_data["observation"] = observation
                    except Exception as e:
                        step_data["observation"] = f"Error executing action: {str(e)}"
                else:
                    # No action means the agent has completed the task
                    step_data["observation"] = "Task completed."
                    history.append(step_data)
                    break
                
                # Add step to history and context
                history.append(step_data)
                context_builder.add_step(step_data)
                
                # Display step results
                console.print(f"[bold]Step {step}:[/bold]")
                console.print(Panel(f"[yellow]Thought:[/yellow]\n{step_data['thought']}"))
                if step_data.get("action"):
                    console.print(
                        Panel(
                            f"[cyan]Action:[/cyan]\n{step_data['action']}",
                            border_style="cyan",
                        )
                    )
                if step_data.get("observation") is not None:
                    console.print(
                        Panel(
                            f"[magenta]Observation:[/magenta]\n{step_data['observation']}",
                            border_style="magenta",
                        )
                    )
                console.print("-" * 80)

    except Exception as e:
        console.print(Panel(f"[bold red]Error during ReAct loop:[/bold red]\n{e}"))
        # Add error details to history for inspection
        if not history:  # Only add error if no history exists yet
            history.append({"step": 1, "error": str(e)})

    final_thought = history[-1].get("thought", "No final thought.") if history else "No steps completed."
    console.print(
        Panel(
            f"ReAct loop finished.\n[bold]Final Thought:[/bold] {final_thought}",
            title="ReAct Agent Complete",
            border_style="green",
        )
    )
    return history


# --- Main Function and CLI ---
async def main():
    """Main function to run the ReAct agent."""
    parser = argparse.ArgumentParser(description="ReAct Document Management Agent")
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        help="A single query to process. If provided, the agent will run non-interactively.",
    )
    parser.add_argument(
        "--max-steps",
        "-m",
        type=int,
        default=10,
        help="Maximum number of steps (default: 10)",
    )
    parser.add_argument(
        "--interactive", "-i", action="store_true", help="Run in interactive mode"
    )
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Check API key configuration and exit.",
    )

    args = parser.parse_args()

    # Handle configuration check (same as simple agent)
    if args.check_config:
        config = check_api_keys_config()
        print("=== API Configuration Status ===")
        print(f"OpenAI configured: {'✓' if config['openai_configured'] else '✗'}")
        if config["openai_configured"]:
            print(f"  Model: {config.get('openai_model', 'N/A')}")
        print(f"Gemini configured: {'✓' if config['gemini_configured'] else '✗'}")
        if config["gemini_configured"]:
            print(f"  Model: {config.get('gemini_model', 'N/A')}")
        print()
        if config["active_provider"]:
            print(f"Active provider: {config['active_provider'].upper()}")
            print(f"Active model: {config['active_model']}")
        else:
            print("No API keys configured!")
            print(
                "Please set either OPENAI_API_KEY or GEMINI_API_KEY in your .env file."
            )
        return

    # Check API key configuration
    print("Checking API key configuration...")
    config_status = check_api_keys_config()

    if not config_status["active_provider"]:
        print(
            "Configuration error: No valid API key found in environment variables. "
            "Please set one of the following in your .env file:\n"
            "- OPENAI_API_KEY for OpenAI models\n"
            "- GEMINI_API_KEY for Google Gemini models\n"
            "\nOptionally, you can also set:\n"
            "- OPENAI_MODEL_NAME (default: gpt-4.1-mini)\n"
            "- GEMINI_MODEL_NAME (default: gemini-2.5-flash)"
        )
        print(
            "Error: No valid API key found in environment variables. Please set one of the following in your .env file:\n"
            "- OPENAI_API_KEY for OpenAI models\n"
            "- GEMINI_API_KEY for Google Gemini models\n"
            "\nOptionally, you can also set:\n"
            "- OPENAI_MODEL_NAME (default: gpt-4.1-mini)\n"
            "- GEMINI_MODEL_NAME (default: gemini-2.5-flash)"
        )
        return

    print(
        f"Using {config_status['active_provider'].upper()} with model: {config_status['active_model']}"
    )
    print()

    if args.interactive:
        # Interactive mode
        print("ReAct Agent - Interactive Mode")
        print("Type 'exit' or 'quit' to stop")
        print("=" * 60)

        while True:
            try:
                user_query = input("\nEnter your query: ").strip()

                if user_query.lower() in ["exit", "quit"]:
                    print("Goodbye!")
                    break

                if not user_query:
                    print("Please enter a valid query.")
                    continue

                # Run the ReAct loop
                print()
                history = await run_react_loop(user_query, args.max_steps)

                # Display summary
                print("\nExecution Summary:")
                print(f"   Total steps: {len(history)}")
                if history:
                    final_step = history[-1]
                    if final_step["action"] is None:
                        print("   Status: Completed successfully")
                    else:
                        print("   Status: Incomplete (max steps reached)")
                print()

            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
                print()

    elif args.query:
        # Single query mode
        print(f"Processing query: {args.query}")
        print()

        try:
            history = await run_react_loop(args.query, args.max_steps)

            # Display summary
            print("\nExecution Summary:")
            print(f"   Query: {args.query}")
            print(f"   Total steps: {len(history)}")
            if history:
                final_step = history[-1]
                if final_step["action"] is None:
                    print("   Status: Completed successfully")
                else:
                    print("   Status: Incomplete (max steps reached)")

        except Exception as e:
            print(f"Error: {e}")

    else:
        # No query provided, show help
        parser.print_help()
        print("\nExample usage:")
        print(
            '  python src/agents/react_agent/main.py --query "Create a document called My Story"'
        )
        print("  python src/agents/react_agent/main.py --interactive")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Critical error: {e}")
        sys.exit(1)
