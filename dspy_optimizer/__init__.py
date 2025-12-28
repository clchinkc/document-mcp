"""DSPy Optimizer for Document MCP tool selection.

This package provides automated prompt optimization using Stanford's DSPy framework.
Optimizes tool selection across ALL 28 MCP tools.
"""

from .optimizer import OptimizationResult
from .optimizer import ToolSelectionOptimizer
from .optimizer import run_multi_model_comparison
from .optimizer import run_optimization

__all__ = [
    "ToolSelectionOptimizer",
    "OptimizationResult",
    "run_optimization",
    "run_multi_model_comparison",
]
