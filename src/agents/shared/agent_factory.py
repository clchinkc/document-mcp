"""Agent factory for creating and managing Document MCP agents.

This module provides a factory pattern for creating agents with consistent
configuration, caching, and lifecycle management.
"""

import asyncio
import hashlib

from document_mcp.config import get_settings
from document_mcp.exceptions import AgentConfigurationError

from .agent_base import AgentBase


class AgentFactory:
    """Factory for creating and managing Document MCP agents."""

    def __init__(self):
        """Initialize the agent factory."""
        self.settings = get_settings()
        self._agent_cache: dict[str, AgentBase] = {}
        self._registered_agents: dict[str, type[AgentBase]] = {}

    def register_agent(self, agent_type: str, agent_class: type[AgentBase]) -> None:
        """Register an agent class with the factory.

        Args:
            agent_type: Unique identifier for the agent type
            agent_class: Agent class to register
        """
        self._registered_agents[agent_type] = agent_class

    def get_available_agents(self) -> list[str]:
        """Get list of available agent types.

        Returns:
            List of registered agent type names
        """
        return list(self._registered_agents.keys())

    def _create_cache_key(self, agent_type: str, **config_params) -> str:
        """Create a cache key for agent instances.

        Args:
            agent_type: Type of agent
            **config_params: Configuration parameters

        Returns:
            Cache key string
        """
        # Include relevant configuration in cache key
        config_data = {
            "agent_type": agent_type,
            "provider": self.settings.active_provider,
            "model": self.settings.active_model,
            "is_test": self.settings.is_test_environment,
            **config_params,
        }

        # Create a hash of the configuration
        config_str = str(sorted(config_data.items()))
        return hashlib.md5(config_str.encode()).hexdigest()

    async def create_agent(self, agent_type: str, use_cache: bool = True, **config_params) -> AgentBase:
        """Create or retrieve an agent instance.

        Args:
            agent_type: Type of agent to create
            use_cache: Whether to use cached instances
            **config_params: Additional configuration parameters

        Returns:
            Agent instance

        Raises:
            AgentConfigurationError: If agent type is not registered or creation fails
        """
        if agent_type not in self._registered_agents:
            raise AgentConfigurationError(
                agent_type,
                f"Agent type '{agent_type}' is not registered. Available types: {list(self._registered_agents.keys())}",
                details={"available_agents": list(self._registered_agents.keys())},
            )

        # Check cache if enabled
        if use_cache:
            cache_key = self._create_cache_key(agent_type, **config_params)
            if cache_key in self._agent_cache:
                return self._agent_cache[cache_key]

        # Create new agent instance
        try:
            agent_class = self._registered_agents[agent_type]
            agent = agent_class(agent_type)

            # Initialize the agent
            await agent.get_llm()  # Ensure LLM is loaded
            agent.validate_configuration()

            # Cache the instance if caching is enabled
            if use_cache:
                cache_key = self._create_cache_key(agent_type, **config_params)
                self._agent_cache[cache_key] = agent

            return agent

        except Exception as e:
            raise AgentConfigurationError(
                agent_type,
                f"Failed to create {agent_type} agent: {str(e)}",
                details={"original_error": str(e), "config_params": config_params},
            ) from e

    async def get_or_create_agent(self, agent_type: str, **config_params) -> AgentBase:
        """Get an existing agent or create a new one.

        This is a convenience method that always uses caching.

        Args:
            agent_type: Type of agent to get/create
            **config_params: Additional configuration parameters

        Returns:
            Agent instance
        """
        return await self.create_agent(agent_type, use_cache=True, **config_params)

    def clear_cache(self, agent_type: str | None = None) -> None:
        """Clear cached agent instances.

        Args:
            agent_type: Specific agent type to clear, or None to clear all
        """
        if agent_type is None:
            self._agent_cache.clear()
        else:
            # Clear cache entries for specific agent type
            keys_to_remove = [
                key for key in self._agent_cache.keys() if self._agent_cache[key].agent_type == agent_type
            ]
            for key in keys_to_remove:
                del self._agent_cache[key]

    def get_cache_stats(self) -> dict[str, int]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        stats = {"total_cached": len(self._agent_cache)}

        # Count by agent type
        for agent in self._agent_cache.values():
            key = f"{agent.agent_type}_count"
            stats[key] = stats.get(key, 0) + 1

        return stats

    async def shutdown(self) -> None:
        """Shutdown the factory and cleanup resources."""
        # Close all cached agents if they have cleanup methods
        cleanup_tasks = []
        for agent in self._agent_cache.values():
            if hasattr(agent, "__aexit__"):
                cleanup_tasks.append(agent.__aexit__(None, None, None))

        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)

        self._agent_cache.clear()


# Global factory instance
_global_factory: AgentFactory | None = None


def get_agent_factory() -> AgentFactory:
    """Get the global agent factory instance.

    Returns:
        Global AgentFactory instance
    """
    global _global_factory
    if _global_factory is None:
        _global_factory = AgentFactory()
    return _global_factory


async def create_agent(agent_type: str, **config_params) -> AgentBase:
    """Convenience function to create an agent using the global factory.

    Args:
        agent_type: Type of agent to create
        **config_params: Additional configuration parameters

    Returns:
        Agent instance
    """
    factory = get_agent_factory()
    return await factory.create_agent(agent_type, **config_params)


def register_agent(agent_type: str, agent_class: type[AgentBase]) -> None:
    """Convenience function to register an agent with the global factory.

    Args:
        agent_type: Unique identifier for the agent type
        agent_class: Agent class to register
    """
    factory = get_agent_factory()
    factory.register_agent(agent_type, agent_class)
