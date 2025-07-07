#!/usr/bin/env python3
"""
Shared CLI utilities for document management agents.

This module provides common command-line argument parsing and configuration
checking functionality used by both simple and ReAct agents.
"""

import argparse
import os
from pathlib import Path

from .config import check_api_keys_config


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
        print("Please set either OPENAI_API_KEY or GEMINI_API_KEY in your .env file.")


def validate_api_configuration() -> bool:
    """Validate API configuration and display errors if invalid. Returns True if valid."""
    config = check_api_keys_config()
    if not config["active_provider"]:
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

    print(
        f"Using {config['active_provider'].upper()} with model: {config['active_model']}"
    )
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
