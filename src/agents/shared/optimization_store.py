"""Versioned optimization storage for DSPy-optimized prompts.

This module provides a clean interface for storing and loading optimized
prompt instructions with full version history support.

Storage Structure:
    prompt_optimizations/
    ├── full/
    │   ├── current.json      # Active optimization (latest or pinned)
    │   ├── manifest.json     # Version history and metadata
    │   └── history/
    │       ├── v001.json
    │       └── v002.json
    ├── compact/
    └── minimal/

Usage:
    from src.agents.shared.optimization_store import OptimizationStore

    store = OptimizationStore()

    # Save new optimization (auto-versions)
    version = store.save("full", instructions, metrics)

    # Load current optimization
    opt = store.load("full")

    # List all versions
    versions = store.list_versions("full")

    # Rollback to specific version
    store.set_current("full", "v001")
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class OptimizationMetrics:
    """Metrics from an optimization run."""

    accuracy: float
    composite: float
    avg_input_tokens: float
    avg_output_tokens: float


@dataclass
class OptimizationVersion:
    """A single optimization version."""

    version: str
    timestamp: str
    optimizer: str
    model: str
    baseline: OptimizationMetrics
    optimized: OptimizationMetrics
    instructions: dict[str, str]

    @property
    def improvement(self) -> float:
        """Composite score improvement."""
        return self.optimized.composite - self.baseline.composite


@dataclass
class Manifest:
    """Manifest tracking all versions for a variant."""

    variant: str
    current_version: str | None
    versions: list[dict]  # List of version metadata
    created_at: str
    updated_at: str


def get_store_root() -> Path:
    """Get the optimization store root directory."""
    # Project root / prompt_optimizations
    return Path(__file__).parent.parent.parent.parent / "prompt_optimizations"


class OptimizationStore:
    """Versioned storage for prompt optimizations."""

    def __init__(self, root: Path | None = None):
        """Initialize the optimization store.

        Args:
            root: Custom root directory (defaults to project's prompt_optimizations/)
        """
        self.root = root or get_store_root()

    def _get_variant_dir(self, variant: str) -> Path:
        """Get directory for a specific variant."""
        return self.root / variant

    def _get_history_dir(self, variant: str) -> Path:
        """Get history directory for a variant."""
        return self._get_variant_dir(variant) / "history"

    def _get_manifest_path(self, variant: str) -> Path:
        """Get manifest file path for a variant."""
        return self._get_variant_dir(variant) / "manifest.json"

    def _get_current_path(self, variant: str) -> Path:
        """Get current optimization file path for a variant."""
        return self._get_variant_dir(variant) / "current.json"

    def _ensure_dirs(self, variant: str) -> None:
        """Ensure all directories exist for a variant."""
        self._get_history_dir(variant).mkdir(parents=True, exist_ok=True)

    def _load_manifest(self, variant: str) -> Manifest:
        """Load or create manifest for a variant."""
        manifest_path = self._get_manifest_path(variant)

        if manifest_path.exists():
            with open(manifest_path) as f:
                data = json.load(f)
                return Manifest(**data)

        # Create new manifest
        now = datetime.now().isoformat()
        return Manifest(
            variant=variant,
            current_version=None,
            versions=[],
            created_at=now,
            updated_at=now,
        )

    def _save_manifest(self, manifest: Manifest) -> None:
        """Save manifest to disk."""
        self._ensure_dirs(manifest.variant)
        manifest_path = self._get_manifest_path(manifest.variant)
        manifest.updated_at = datetime.now().isoformat()

        with open(manifest_path, "w") as f:
            json.dump(asdict(manifest), f, indent=2)

    def _get_next_version(self, manifest: Manifest) -> str:
        """Get the next version number."""
        if not manifest.versions:
            return "v001"

        # Find highest version number
        max_num = 0
        for v in manifest.versions:
            try:
                num = int(v["version"][1:])  # Remove 'v' prefix
                max_num = max(max_num, num)
            except (ValueError, KeyError):
                continue

        return f"v{max_num + 1:03d}"

    def save(
        self,
        variant: str,
        instructions: dict[str, str],
        baseline_metrics: dict,
        optimized_metrics: dict,
        optimizer: str = "copro",
        model: str = "unknown",
        rules: dict | None = None,
        demos: list[dict] | None = None,
    ) -> str:
        """Save a new optimization version.

        Args:
            variant: Prompt variant ("full", "compact", "minimal")
            instructions: Optimized instructions dict (from COPRO/MIPROv2)
            baseline_metrics: Baseline metrics dict
            optimized_metrics: Optimized metrics dict
            optimizer: Optimizer type used
            model: Model used for optimization
            rules: Parsed tool selection rules (categories, disambiguation, error handling)
            demos: Few-shot demonstrations (from BootstrapFewShot)

        Returns:
            Version string (e.g., "v001")
        """
        self._ensure_dirs(variant)
        manifest = self._load_manifest(variant)
        version = self._get_next_version(manifest)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create version data
        version_data = {
            "version": version,
            "timestamp": timestamp,
            "optimizer": optimizer,
            "model": model,
            "baseline": baseline_metrics,
            "optimized": optimized_metrics,
            "instructions": instructions,
            "rules": rules or {},
            "demos": demos or [],
        }

        # Save to history
        history_path = self._get_history_dir(variant) / f"{version}.json"
        with open(history_path, "w") as f:
            json.dump(version_data, f, indent=2)

        # Update manifest
        manifest.versions.append(
            {
                "version": version,
                "timestamp": timestamp,
                "optimizer": optimizer,
                "model": model,
                "accuracy_improvement": optimized_metrics.get("accuracy", 0)
                - baseline_metrics.get("accuracy", 0),
                "composite_improvement": optimized_metrics.get("composite", 0)
                - baseline_metrics.get("composite", 0),
            }
        )
        manifest.current_version = version
        self._save_manifest(manifest)

        # Update current.json
        self._update_current(variant, version_data)

        logger.info(f"Saved optimization {version} for variant '{variant}'")
        return version

    def _update_current(self, variant: str, version_data: dict) -> None:
        """Update the current.json file."""
        current_path = self._get_current_path(variant)
        with open(current_path, "w") as f:
            json.dump(version_data, f, indent=2)

    def load(self, variant: str) -> dict | None:
        """Load the current optimization for a variant.

        Args:
            variant: Prompt variant ("full", "compact", "minimal")

        Returns:
            Optimization data dict or None if not found
        """
        current_path = self._get_current_path(variant)

        if not current_path.exists():
            logger.debug(f"No current optimization for variant '{variant}'")
            return None

        try:
            with open(current_path) as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load optimization for '{variant}': {e}")
            return None

    def load_instructions(self, variant: str) -> dict[str, str]:
        """Load just the instructions for a variant.

        Args:
            variant: Prompt variant

        Returns:
            Instructions dict or empty dict if not found
        """
        data = self.load(variant)
        if data:
            return data.get("instructions", {})
        return {}

    def load_demos(self, variant: str) -> list[dict]:
        """Load few-shot demos for a variant.

        Args:
            variant: Prompt variant

        Returns:
            List of demo dicts or empty list if not found
        """
        data = self.load(variant)
        if data:
            return data.get("demos", [])
        return []

    def load_rules(self, variant: str) -> dict:
        """Load tool selection rules for a variant.

        Args:
            variant: Prompt variant

        Returns:
            Dict with 'categories', 'disambiguation', 'error_handling' keys
        """
        data = self.load(variant)
        if data:
            return data.get("rules", {})
        return {}

    def load_version(self, variant: str, version: str) -> dict | None:
        """Load a specific version.

        Args:
            variant: Prompt variant
            version: Version string (e.g., "v001")

        Returns:
            Version data or None if not found
        """
        history_path = self._get_history_dir(variant) / f"{version}.json"

        if not history_path.exists():
            logger.warning(f"Version {version} not found for variant '{variant}'")
            return None

        with open(history_path) as f:
            return json.load(f)

    def list_versions(self, variant: str) -> list[dict]:
        """List all versions for a variant.

        Args:
            variant: Prompt variant

        Returns:
            List of version metadata dicts
        """
        manifest = self._load_manifest(variant)
        return manifest.versions

    def get_current_version(self, variant: str) -> str | None:
        """Get the current version string for a variant."""
        manifest = self._load_manifest(variant)
        return manifest.current_version

    def set_current(self, variant: str, version: str) -> bool:
        """Set a specific version as current (rollback/pin).

        Args:
            variant: Prompt variant
            version: Version to set as current

        Returns:
            True if successful, False otherwise
        """
        version_data = self.load_version(variant, version)
        if not version_data:
            return False

        # Update current.json
        self._update_current(variant, version_data)

        # Update manifest
        manifest = self._load_manifest(variant)
        manifest.current_version = version
        self._save_manifest(manifest)

        logger.info(f"Set {version} as current for variant '{variant}'")
        return True

    def get_version_history(self, variant: str, limit: int = 10) -> list[dict]:
        """Get recent version history with details.

        Args:
            variant: Prompt variant
            limit: Max versions to return

        Returns:
            List of version summaries (most recent first)
        """
        versions = self.list_versions(variant)
        return list(reversed(versions[-limit:]))


# Singleton instance for easy access
_store: OptimizationStore | None = None


def get_store() -> OptimizationStore:
    """Get the global optimization store instance."""
    global _store
    if _store is None:
        _store = OptimizationStore()
    return _store
