#!/usr/bin/env python3
"""Shared CLI utilities for document management agents.

This module provides common command-line argument parsing and configuration
checking functionality used by both simple and ReAct agents.
"""

import argparse
import os
from pathlib import Path

from .config import get_settings


def create_base_parser(description: str) -> argparse.ArgumentParser:
    """Create the base argument parser with common options."""
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "--query",
        "-q",
        type=str,
        help="A single query to process. If provided, the agent will run non-interactively.",
    )

    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Run in interactive mode",
    )

    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Check API key configuration and exit.",
    )

    return parser


def add_react_specific_args(parser: argparse.ArgumentParser) -> None:
    """Add ReAct-specific arguments to the parser."""
    parser.add_argument(
        "--max-steps",
        "-m",
        type=int,
        default=10,
        help="Maximum number of steps (default: 10)",
    )


def handle_config_check() -> None:
    """Handle the --check-config flag by displaying API configuration status."""
    settings = get_settings()
    print("=== API Configuration Status ===")
    print(f"OpenAI configured: {'[OK]' if settings.openai_configured else '[X]'}")
    if settings.openai_configured:
        print(f"  Model: {settings.openai_model_name}")
    print(f"Gemini configured: {'[OK]' if settings.gemini_configured else '[X]'}")
    if settings.gemini_configured:
        print(f"  Model: {settings.gemini_model_name}")
    print()
    if settings.active_provider:
        print(f"Active provider: {settings.active_provider.upper()}")
        print(f"Active model: {settings.active_model}")
    else:
        print("No API keys configured!")
        print("Please set either OPENAI_API_KEY or GEMINI_API_KEY in your .env file.")


def validate_api_configuration() -> bool:
    """Validate API configuration and display errors if invalid. Returns True if valid."""
    settings = get_settings()
    if not settings.active_provider:
        print(
            "Configuration error: No valid API key found in environment variables. "
            "Please set one of the following in your .env file:\n"
            "- OPENAI_API_KEY for OpenAI models\n"
            "- GEMINI_API_KEY for Google Gemini models\n"
            "\nOptionally, you can also set:\n"
            "- OPENAI_MODEL_NAME (default: gpt-4.1-mini)\n"
            "- GEMINI_MODEL_NAME (default: gemini-2.5-flash)"
        )
        return False

    print(f"Using {settings.active_provider.upper()} with model: {settings.active_model}")
    return True


def setup_document_root() -> None:
    """Set the document root directory for the server-side logic."""
    if "DOCUMENT_ROOT_DIR" in os.environ:
        from document_mcp import doc_tool_server

        doc_tool_server.DOCS_ROOT_PATH = Path(os.environ["DOCUMENT_ROOT_DIR"])


def parse_simple_agent_args() -> argparse.Namespace:
    """Parse command-line arguments for the simple agent."""
    parser = create_base_parser("Simple Document Agent")
    return parser.parse_args()


def parse_react_agent_args() -> argparse.Namespace:
    """Parse command-line arguments for the ReAct agent."""
    parser = create_base_parser("ReAct Document Management Agent")
    add_react_specific_args(parser)
    return parser.parse_args()


def parse_planner_agent_args() -> argparse.Namespace:
    """Parse command-line arguments for the planner agent."""
    parser = create_base_parser("Planner Agent - Plan-and-Execute approach")
    return parser.parse_args()
