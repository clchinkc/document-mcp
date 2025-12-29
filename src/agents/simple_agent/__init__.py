"""Simple Document Management Agent Package.

This package contains the simple agent implementation that provides
single-tool execution with structured output for document management.
"""

from ..shared.agent_factory import register_agent


def __getattr__(name):
    """Lazy load SimpleAgent to avoid importing pydantic_ai.mcp at package load time."""
    if name == "SimpleAgent":
        from .agent import SimpleAgent

        # Register the simple agent with the factory
        register_agent("simple", SimpleAgent)
        return SimpleAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["SimpleAgent"]
