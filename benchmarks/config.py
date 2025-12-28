"""Benchmark Configuration.

Flexible configuration for comparing models, tool sets, and descriptions.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from dataclasses import field
from typing import Any


@dataclass
class ModelConfig:
    """Configuration for a specific LLM model."""

    name: str
    model_id: str
    base_url: str = "https://openrouter.ai/api/v1"
    api_key_env: str = "OPENROUTER_API_KEY"

    @property
    def is_available(self) -> bool:
        """Check if API key is available."""
        key = os.environ.get(self.api_key_env, "").strip()
        return bool(key) and key != "test_key" and not key.startswith("sk-test")


# Standard models for multi-model benchmarking (OpenRouter model IDs)
# Gemini 3 Flash is the default (first in list)
BENCHMARK_MODELS = [
    ModelConfig(name="Gemini 3 Flash", model_id="google/gemini-3-flash-preview"),
    ModelConfig(name="Claude Haiku 4.5", model_id="anthropic/claude-haiku-4.5"),
    ModelConfig(name="GPT-5 Mini", model_id="openai/gpt-5-mini"),
]

# Default model for optimization and benchmarks
DEFAULT_MODEL = BENCHMARK_MODELS[0]  # Gemini 3 Flash


def get_available_models() -> list[ModelConfig]:
    """Get models with valid API keys."""
    return [m for m in BENCHMARK_MODELS if m.is_available]


@dataclass
class ComparisonConfig:
    """Configuration for A/B comparison of two variants."""

    name: str
    variant_a: Any  # Tool set, model, or description set
    variant_b: Any
    variant_a_name: str = "Variant A"
    variant_b_name: str = "Variant B"


@dataclass
class ScoringWeights:
    """Weights for composite score calculation."""

    accuracy: float = 0.6
    input_tokens: float = 0.25
    output_tokens: float = 0.15

    def __post_init__(self):
        total = self.accuracy + self.input_tokens + self.output_tokens
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total}")

    def as_dict(self) -> dict[str, float]:
        return {
            "accuracy_weight": self.accuracy,
            "input_token_weight": self.input_tokens,
            "output_token_weight": self.output_tokens,
        }


@dataclass
class BenchmarkConfig:
    """Flexible configuration for benchmark runs.

    Supports comparing:
    - Different tool sets (atomic vs consolidated)
    - Different models (GPT vs Gemini vs Claude)
    - Different tool descriptions (verbose vs compact)
    - Any combination of the above
    """

    # What to compare
    models: list[ModelConfig] = field(default_factory=list)
    tool_sets: list[str] = field(default_factory=lambda: ["atomic"])

    # Custom tool descriptions (optional override)
    custom_tool_descriptions: dict[str, str] = field(default_factory=dict)

    # Scoring configuration
    weights: ScoringWeights = field(default_factory=ScoringWeights)

    # Token baselines for normalization
    input_token_baseline: float = 2000
    output_token_baseline: float = 500

    # DSPy optimization settings
    num_trials: int = 3
    max_bootstrapped_demos: int = 3
    max_labeled_demos: int = 2

    # Comparison mode
    comparison: ComparisonConfig | None = None

    def __post_init__(self):
        if not self.models:
            self.models = get_available_models()[:1] or BENCHMARK_MODELS[:1]

    def get_models(self) -> list[ModelConfig]:
        """Get models to benchmark."""
        return [m for m in self.models if m.is_available] or self.models

    def get_tool_sets(self) -> list[str]:
        """Get tool sets to benchmark."""
        return self.tool_sets

    @classmethod
    def for_tool_comparison(
        cls,
        tool_set_a: str = "atomic",
        tool_set_b: str = "consolidated",
        model: ModelConfig | None = None,
    ) -> BenchmarkConfig:
        """Create config for comparing two tool sets."""
        return cls(
            models=[model] if model else [],
            tool_sets=[tool_set_a, tool_set_b],
            comparison=ComparisonConfig(
                name="Tool Set Comparison",
                variant_a=tool_set_a,
                variant_b=tool_set_b,
                variant_a_name=f"{tool_set_a.capitalize()} Tools",
                variant_b_name=f"{tool_set_b.capitalize()} Tools",
            ),
        )

    @classmethod
    def for_model_comparison(
        cls,
        model_a: ModelConfig,
        model_b: ModelConfig,
        tool_set: str = "atomic",
    ) -> BenchmarkConfig:
        """Create config for comparing two models."""
        return cls(
            models=[model_a, model_b],
            tool_sets=[tool_set],
            comparison=ComparisonConfig(
                name="Model Comparison",
                variant_a=model_a,
                variant_b=model_b,
                variant_a_name=model_a.name,
                variant_b_name=model_b.name,
            ),
        )

    @classmethod
    def for_description_comparison(
        cls,
        descriptions_a: dict[str, str],
        descriptions_b: dict[str, str],
        name_a: str = "Current",
        name_b: str = "Improved",
        model: ModelConfig | None = None,
    ) -> BenchmarkConfig:
        """Create config for comparing tool descriptions."""
        return cls(
            models=[model] if model else [],
            custom_tool_descriptions=descriptions_a,  # Will switch for variant B
            comparison=ComparisonConfig(
                name="Description Comparison",
                variant_a=descriptions_a,
                variant_b=descriptions_b,
                variant_a_name=name_a,
                variant_b_name=name_b,
            ),
        )

    @classmethod
    def full_matrix(
        cls,
        models: list[ModelConfig] | None = None,
        tool_sets: list[str] | None = None,
    ) -> BenchmarkConfig:
        """Create config for full matrix comparison (all models x all tool sets)."""
        return cls(
            models=models or BENCHMARK_MODELS,
            tool_sets=tool_sets or ["atomic", "consolidated"],
        )
