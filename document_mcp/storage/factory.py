"""Storage Backend Factory.

Auto-selects the appropriate storage backend based on environment:
- GCS: When running on GCP (K_SERVICE set) or GCS_BUCKET is configured
- Local: Default for development and testing
"""

from __future__ import annotations

import os
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import StorageBackend


class StorageType(Enum):
    """Available storage backend types."""

    LOCAL = "local"
    GCS = "gcs"


def detect_environment() -> StorageType:
    """Auto-detect the appropriate storage backend.

    Simple detection logic:
    1. STORAGE_BACKEND env var (explicit override: "gcs" or "local")
    2. GCS_BUCKET env var set -> GCS
    3. Default -> Local

    Returns:
        StorageType enum indicating which backend to use
    """
    # Check for explicit override
    explicit = os.environ.get("STORAGE_BACKEND", "").lower()
    if explicit == "gcs":
        return StorageType.GCS
    elif explicit == "local":
        return StorageType.LOCAL

    # GCS if bucket is configured (works locally or on Cloud Run)
    if os.environ.get("GCS_BUCKET"):
        return StorageType.GCS

    # Default to local filesystem
    return StorageType.LOCAL


def create_storage_backend(
    storage_type: StorageType | None = None,
    **kwargs,
) -> StorageBackend:
    """Create a storage backend instance.

    Args:
        storage_type: Explicit storage type, or None to auto-detect
        **kwargs: Backend-specific configuration

    Returns:
        Configured StorageBackend instance
    """
    if storage_type is None:
        storage_type = detect_environment()

    if storage_type == StorageType.GCS:
        from .gcs import GCSStorageBackend

        return GCSStorageBackend(
            bucket_name=kwargs.get("bucket_name"),
            prefix=kwargs.get("prefix", ""),
        )
    else:
        from .local import LocalStorageBackend

        return LocalStorageBackend(
            root_dir=kwargs.get("root_dir"),
        )


# Singleton instance for the application
_storage_instance: StorageBackend | None = None


def get_storage(**kwargs) -> StorageBackend:
    """Get the global storage backend instance.

    Creates the instance on first call using auto-detection.
    Subsequent calls return the same instance.

    Args:
        **kwargs: Backend-specific configuration (only used on first call)

    Returns:
        The global StorageBackend instance
    """
    global _storage_instance

    if _storage_instance is None:
        _storage_instance = create_storage_backend(**kwargs)

    return _storage_instance


def get_storage_sync(**kwargs) -> StorageBackend:
    """Synchronous alias for get_storage."""
    return get_storage(**kwargs)


def reset_storage() -> None:
    """Reset the global storage instance (for testing)."""
    global _storage_instance
    _storage_instance = None


def get_storage_info() -> dict:
    """Get information about the current storage configuration.

    Returns:
        Dict with storage backend info for debugging/monitoring
    """
    storage = get_storage()
    detected_type = detect_environment()

    return {
        "backend_type": storage.backend_type,
        "detected_type": detected_type.value,
        "root_path": storage.root_path,
        "gcs_bucket": os.environ.get("GCS_BUCKET"),
        "storage_backend_env": os.environ.get("STORAGE_BACKEND", "auto"),
    }
