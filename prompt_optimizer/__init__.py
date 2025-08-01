"""Prompt Optimizer Package.

A standalone tool for automated prompt optimization using test-based evaluation.

from .cli import main
from .core import PromptOptimizer
from .evaluation import OptimizationResult
from .evaluation import PerformanceEvaluator

__all__ = ["PromptOptimizer", "PerformanceEvaluator", "OptimizationResult", "main"]
