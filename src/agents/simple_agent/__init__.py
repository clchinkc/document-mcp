"""Simple Document Management Agent Package.

This package contains the simple agent implementation that provides
single-tool execution with structured output for document management.
"""

from ..shared.agent_factory import register_agent
from .agent import SimpleAgent

# Register the simple agent with the factory
register_agent("simple", SimpleAgent)

__all__ = ["SimpleAgent"]
