"""Global batch operation registry instance.

This module provides a singleton instance of the BatchOperationRegistry
to avoid circular import issues between the main server and tool modules.
"""

from .registry import BatchOperationRegistry

# Global registry instance
_batch_registry = BatchOperationRegistry()


def get_batch_registry() -> BatchOperationRegistry:
    """Get the global batch operation registry."""
    return _batch_registry
