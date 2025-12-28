"""Benchmark Metrics.

Shared metrics classes for measuring benchmark performance.
"""

from dataclasses import dataclass
from dataclasses import field


@dataclass
class CompositeScore:
    """Breakdown of composite score components."""

    accuracy: float
    input_token_efficiency: float
    output_token_efficiency: float
    total: float

    def __str__(self) -> str:
        return (
            f"Composite: {self.total:.3f} "
            f"(acc={self.accuracy:.1%}, in_eff={self.input_token_efficiency:.2f}, "
            f"out_eff={self.output_token_efficiency:.2f})"
        )


@dataclass
class BenchmarkMetrics:
    """Comprehensive metrics for optimization and comparison."""

    accuracy: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    error_count: int = 0
    num_examples: int = 0
    confusion_matrix: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        """Total tokens (input + output)."""
        return self.input_tokens + self.output_tokens

    @property
    def avg_input_tokens(self) -> float:
        """Average input tokens per query."""
        return self.input_tokens / self.num_examples if self.num_examples else 0

    @property
    def avg_output_tokens(self) -> float:
        """Average output tokens per query."""
        return self.output_tokens / self.num_examples if self.num_examples else 0

    @property
    def avg_total_tokens(self) -> float:
        """Average total tokens per query."""
        return self.total_tokens / self.num_examples if self.num_examples else 0

    @property
    def error_rate(self) -> float:
        return self.error_count / self.num_examples if self.num_examples else 0

    def composite_score(self) -> float:
        """Calculate composite score: accuracy weighted with token efficiency.

        Weights: 60% accuracy, 25% input efficiency, 15% output efficiency

        Token efficiency uses inverse scaling (no reference baselines):
        - efficiency = 1 / (1 + avg_tokens / scale_factor)
        - Lower tokens → efficiency approaches 1.0
        - Higher tokens → efficiency approaches 0.0

        Scale factors control sensitivity, not target values. They're chosen
        so typical LLM responses fall in the useful 0.3-0.7 efficiency range.

        Returns:
            Composite score between 0 and 1
        """
        if self.num_examples == 0:
            return 0.0

        # Input efficiency: inversely proportional to tokens
        # Scale factor 1000 means 1000 tokens gives 0.5 efficiency
        input_eff = 1.0 / (1.0 + self.avg_input_tokens / 1000)

        # Output efficiency: inversely proportional to tokens
        # Scale factor 500 means 500 tokens gives 0.5 efficiency
        output_eff = 1.0 / (1.0 + self.avg_output_tokens / 500)

        return 0.6 * self.accuracy + 0.25 * input_eff + 0.15 * output_eff

    def composite_breakdown(self) -> CompositeScore:
        """Get detailed breakdown of score components."""
        if self.num_examples == 0:
            return CompositeScore(
                accuracy=0.0,
                input_token_efficiency=0.0,
                output_token_efficiency=0.0,
                total=0.0,
            )

        input_eff = 1.0 / (1.0 + self.avg_input_tokens / 1000)
        output_eff = 1.0 / (1.0 + self.avg_output_tokens / 500)

        return CompositeScore(
            accuracy=self.accuracy,
            input_token_efficiency=input_eff,
            output_token_efficiency=output_eff,
            total=self.composite_score(),
        )

    def cost_optimized_score(self) -> float:
        """Score for COST_OPTIMIZED variant: 50% accuracy, 25% input, 25% output.

        Focuses on minimizing token usage while maintaining acceptable accuracy.
        """
        if self.num_examples == 0:
            return 0.0

        input_eff = 1.0 / (1.0 + self.avg_input_tokens / 1000)
        output_eff = 1.0 / (1.0 + self.avg_output_tokens / 500)

        return 0.5 * self.accuracy + 0.25 * input_eff + 0.25 * output_eff

    def accuracy_balanced_score(self) -> float:
        """Score for ACCURACY_BALANCED variant: accuracy per token ratio.

        Metric: Accuracy / (1 + avg_tokens / baseline)
        Rewards high accuracy while penalizing token waste equally in both directions.
        """
        if self.num_examples == 0:
            return 0.0

        # Baseline tokens (typical values)
        baseline_input = 1000
        baseline_output = 500

        # Normalize total tokens to baseline
        avg_total = self.avg_input_tokens + self.avg_output_tokens
        baseline_total = baseline_input + baseline_output
        tokens_ratio = avg_total / baseline_total

        # Accuracy per normalized token
        # accuracy / (1 + excess_tokens)
        # When tokens equal baseline: 1.0 multiplier
        # When tokens = 2x baseline: 0.5 multiplier
        return self.accuracy / (1.0 + max(0, tokens_ratio - 1.0))

    def maximum_accuracy_score(self) -> float:
        """Score for MAXIMUM_ACCURACY variant: pure accuracy.

        Ignores token usage, focuses entirely on tool selection accuracy.
        """
        return self.accuracy

    def add_confusion(self, expected: str, predicted: str) -> None:
        """Track a confusion matrix entry."""
        self.confusion_matrix.setdefault(expected, {})
        self.confusion_matrix[expected][predicted] = self.confusion_matrix[expected].get(predicted, 0) + 1

    def report(self) -> str:
        """Generate a human-readable report."""
        lines = [
            f"Accuracy: {self.accuracy:.1%} ({int(self.accuracy * self.num_examples)}/{self.num_examples})",
            f"Tokens: {self.avg_input_tokens:.0f} in / {self.avg_output_tokens:.0f} out (avg)",
            f"Errors: {self.error_count}",
            f"Composite: {self.composite_score():.3f}",
        ]

        if self.confusion_matrix:
            lines.append("\nConfusion (expected -> predicted):")
            for expected, predictions in sorted(self.confusion_matrix.items()):
                for predicted, count in sorted(predictions.items()):
                    lines.append(f"  {expected} -> {predicted}: {count}")

        return "\n".join(lines)


@dataclass
class ComparisonResult:
    """Result of comparing two benchmark runs."""

    baseline: BenchmarkMetrics
    variant: BenchmarkMetrics
    baseline_name: str = "Baseline"
    variant_name: str = "Variant"

    @property
    def accuracy_change(self) -> float:
        """Change in accuracy (positive = improvement)."""
        return self.variant.accuracy - self.baseline.accuracy

    @property
    def input_token_change(self) -> float:
        """Change in input tokens (negative = improvement)."""
        return self.variant.avg_input_tokens - self.baseline.avg_input_tokens

    @property
    def output_token_change(self) -> float:
        """Change in output tokens (negative = improvement)."""
        return self.variant.avg_output_tokens - self.baseline.avg_output_tokens

    @property
    def composite_improvement(self) -> float:
        """Improvement in composite score (positive = better)."""
        return self.variant.composite_score() - self.baseline.composite_score()

    def report(self) -> str:
        """Generate comparison report."""
        lines = [
            "=" * 60,
            "BENCHMARK COMPARISON",
            "=" * 60,
            "",
            f"{self.baseline_name}:",
            f"  Accuracy: {self.baseline.accuracy:.1%}",
            f"  Tokens: {self.baseline.avg_input_tokens:.0f} in / {self.baseline.avg_output_tokens:.0f} out",
            f"  Composite: {self.baseline.composite_score():.3f}",
            "",
            f"{self.variant_name}:",
            f"  Accuracy: {self.variant.accuracy:.1%}",
            f"  Tokens: {self.variant.avg_input_tokens:.0f} in / {self.variant.avg_output_tokens:.0f} out",
            f"  Composite: {self.variant.composite_score():.3f}",
            "",
            "Changes:",
            f"  Accuracy: {self.accuracy_change:+.1%}",
            f"  Input tokens: {self.input_token_change:+.0f}",
            f"  Output tokens: {self.output_token_change:+.0f}",
            f"  Composite: {self.composite_improvement:+.3f}",
            "=" * 60,
        ]
        return "\n".join(lines)
