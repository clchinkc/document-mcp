"""
Prompt Optimizer Package

A simplified, standalone tool for automated prompt optimization using test-based evaluation.
"""

from .core import PromptOptimizer
from .evaluation import PerformanceEvaluator, OptimizationResult
from .cli import main

__all__ = ["PromptOptimizer", "PerformanceEvaluator", "OptimizationResult", "main"]