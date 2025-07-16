"""Evaluation utilities for agent performance testing with LLM evaluation support.

This module provides specialized utilities for the evaluation test suite,
focusing on performance metrics collection, analysis, and LLM-based quality assessment.
"""

import json
import time
from dataclasses import dataclass
from typing import Any

from tests.e2e.validation_utils import DocumentSystemValidator

# Removed dependency on over-engineered LLM evaluator


@dataclass
class TokenUsageMetrics:
    """Token usage metrics for performance analysis."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __post_init__(self):
        if (
            self.total_tokens == 0
            and self.prompt_tokens > 0
            and self.completion_tokens > 0
        ):
            self.total_tokens = self.prompt_tokens + self.completion_tokens


@dataclass
class ToolCallMetrics:
    """Tool call metrics for performance analysis."""

    tool_name: str
    execution_time: float
    success: bool
    error_message: str | None = None
    result_size: int = 0  # Size of the result in characters


class EvaluationAssertions:
    """Specialized assertions for evaluation tests focusing on structured data and LLM evaluation."""

    @staticmethod
    def assert_details_field_structure(
        response: dict[str, Any], expected_operation: str
    ):
        """Assert that the details field contains expected operation structure."""
        assert "details" in response, "Response must contain 'details' field"

        details = response["details"]
        if isinstance(details, str):
            try:
                details = json.loads(details)
            except json.JSONDecodeError:
                raise AssertionError(f"Details field contains invalid JSON: {details}")

        assert isinstance(details, dict), (
            f"Details must be a dictionary, got {type(details)}"
        )
        assert "operation" in details, "Details must contain 'operation' field"
        assert details["operation"] == expected_operation, (
            f"Expected operation '{expected_operation}', got '{details['operation']}'"
        )
        assert "success" in details, "Details must contain 'success' field"
        assert isinstance(details["success"], bool), "Success field must be boolean"

    @staticmethod
    def assert_performance_within_bounds(
        metrics: "AgentPerformanceMetrics", max_tokens: int, max_time: float
    ):
        """Assert that performance metrics are within acceptable bounds."""
        if metrics.token_usage is not None:
            assert metrics.token_usage <= max_tokens, (
                f"Token usage {metrics.token_usage} exceeds maximum {max_tokens}"
            )

        assert metrics.execution_time <= max_time, (
            f"Execution time {metrics.execution_time:.2f}s exceeds maximum {max_time}s"
        )

    @staticmethod
    def assert_file_system_state(
        validator: DocumentSystemValidator, expected_changes: dict[str, Any]
    ):
        """Assert that file system state matches expected changes."""
        for change_type, change_data in expected_changes.items():
            if change_type == "documents_created":
                for doc_name in change_data:
                    validator.assert_document_exists(doc_name)
            elif change_type == "chapters_created":
                for doc_name, chapters in change_data.items():
                    for chapter_name in chapters:
                        validator.assert_chapter_exists(doc_name, chapter_name)
            elif change_type == "documents_deleted":
                for doc_name in change_data:
                    validator.assert_document_not_exists(doc_name)
            elif change_type == "content_contains":
                for (doc_name, chapter_name), content in change_data.items():
                    validator.assert_chapter_content_contains(
                        doc_name, chapter_name, content
                    )

    @staticmethod
    def assert_token_efficiency(agent_type: str, operation: str, token_count: int):
        """Assert that token usage is efficient for the given operation."""
        # Token usage baselines for different operations
        baselines = {
            "simple": {
                "create_document": 200,
                "create_chapter": 250,
                "list_documents": 150,
                "read_document": 300,
                "update_chapter": 400,
            },
            "react": {
                "create_document": 500,
                "create_chapter": 600,
                "list_documents": 400,
                "read_document": 700,
                "update_chapter": 800,
            },
        }

        if agent_type in baselines and operation in baselines[agent_type]:
            max_tokens = baselines[agent_type][operation]
            assert token_count <= max_tokens, (
                f"{agent_type} agent used {token_count} tokens for {operation}, "
                f"expected <= {max_tokens}"
            )

    @staticmethod
    def assert_llm_evaluation_quality(enhanced_metrics, min_score: float = 0.5):
        """Assert LLM evaluation quality using clean architecture pattern."""
        if enhanced_metrics.llm_evaluation and enhanced_metrics.llm_evaluation.success:
            assert enhanced_metrics.llm_evaluation.score >= min_score, (
                f"LLM quality score {enhanced_metrics.llm_evaluation.score:.2f} below minimum {min_score}"
            )
            assert enhanced_metrics.llm_evaluation.feedback != "", (
                "Should provide feedback"
            )
        # If LLM evaluation is disabled or failed, test still passes (graceful degradation)

    @staticmethod
    def assert_combined_score_threshold(
        performance_score: float,
        quality_score: float,
        combined_score: float,
        expected_threshold: float = 0.6,
        performance_weight: float = 0.7,
        quality_weight: float = 0.3,
    ):
        """Assert that combined score meets threshold and is calculated correctly."""
        # Verify calculation
        expected_combined = (performance_weight * performance_score) + (
            quality_weight * quality_score
        )
        assert abs(combined_score - expected_combined) < 0.01, (
            f"Combined score {combined_score:.3f} doesn't match expected {expected_combined:.3f}"
        )

        # Verify threshold
        assert combined_score >= expected_threshold, (
            f"Combined score {combined_score:.2f} below threshold {expected_threshold}"
        )

    @staticmethod
    def assert_comparative_evaluation_results(
        comparison_results: dict[str, Any],
        expected_agents: list[str],
        require_ranking: bool = True,
    ):
        """Assert that comparative evaluation results are properly structured."""
        assert "individual_evaluations" in comparison_results
        assert "ranked_responses" in comparison_results
        assert "best_response_index" in comparison_results

        individual_evals = comparison_results["individual_evaluations"]
        assert len(individual_evals) == len(expected_agents), (
            f"Expected {len(expected_agents)} evaluations, got {len(individual_evals)}"
        )

        if require_ranking:
            ranked_responses = comparison_results["ranked_responses"]
            assert len(ranked_responses) > 0, "Should have ranked responses"

            # Check ranking order (should be descending by score)
            for i in range(len(ranked_responses) - 1):
                current_score = ranked_responses[i]["overall_score"]
                next_score = ranked_responses[i + 1]["overall_score"]
                assert current_score >= next_score, (
                    "Ranking should be in descending order"
                )


class PerformanceTracker:
    """Context manager for tracking performance metrics during test execution."""

    def __init__(self):
        self.start_time: float | None = None
        self.end_time: float | None = None
        self.tool_calls: list[ToolCallMetrics] = []
        self.token_usage: TokenUsageMetrics | None = None
        self.file_system_snapshot_before: dict | None = None
        self.file_system_snapshot_after: dict | None = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()

    @property
    def execution_time(self) -> float:
        """Get total execution time."""
        if self.start_time is None or self.end_time is None:
            return 0.0
        return self.end_time - self.start_time

    def record_tool_call(
        self,
        tool_name: str,
        execution_time: float,
        success: bool,
        error_message: str | None = None,
        result_size: int = 0,
    ):
        """Record a tool call for performance analysis."""
        self.tool_calls.append(
            ToolCallMetrics(
                tool_name=tool_name,
                execution_time=execution_time,
                success=success,
                error_message=error_message,
                result_size=result_size,
            )
        )

    def set_token_usage(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int | None = None,
    ):
        """Set token usage metrics."""
        self.token_usage = TokenUsageMetrics(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens or (prompt_tokens + completion_tokens),
        )

    def take_file_system_snapshot(
        self, validator: DocumentSystemValidator, snapshot_type: str
    ):
        """Take a snapshot of the file system state."""
        snapshot = {
            "timestamp": time.time(),
            "documents": list(validator.get_document_names()),
            "chapters": {},
        }

        for doc_name in snapshot["documents"]:
            snapshot["chapters"][doc_name] = list(validator.get_chapter_names(doc_name))

        if snapshot_type == "before":
            self.file_system_snapshot_before = snapshot
        elif snapshot_type == "after":
            self.file_system_snapshot_after = snapshot

    def get_file_system_changes(self) -> dict[str, Any]:
        """Get changes in file system state between snapshots."""
        if not self.file_system_snapshot_before or not self.file_system_snapshot_after:
            return {}

        before = self.file_system_snapshot_before
        after = self.file_system_snapshot_after

        changes = {
            "documents_created": list(
                set(after["documents"]) - set(before["documents"])
            ),
            "documents_deleted": list(
                set(before["documents"]) - set(after["documents"])
            ),
            "chapters_created": {},
            "chapters_deleted": {},
        }

        # Track chapter changes
        for doc_name in set(before["documents"]).union(set(after["documents"])):
            before_chapters = set(before["chapters"].get(doc_name, []))
            after_chapters = set(after["chapters"].get(doc_name, []))

            created = list(after_chapters - before_chapters)
            deleted = list(before_chapters - after_chapters)

            if created:
                changes["chapters_created"][doc_name] = created
            if deleted:
                changes["chapters_deleted"][doc_name] = deleted

        return changes


class MockDataGenerator:
    """Generator for consistent mock data across evaluation tests."""

    @staticmethod
    def create_test_document_data(doc_name: str) -> dict[str, Any]:
        """Create consistent test document data."""
        return {
            "document_name": doc_name,
            "chapters": {
                "01-introduction.md": "This is the introduction chapter.",
                "02-methodology.md": "This chapter describes the methodology.",
                "03-results.md": "This chapter presents the results.",
                "04-conclusion.md": "This chapter provides the conclusion.",
            },
            "summary": f"Test document {doc_name} with 4 chapters",
        }

    @staticmethod
    def create_test_chapter_data(doc_name: str, chapter_name: str) -> dict[str, Any]:
        """Create consistent test chapter data."""
        return {
            "document_name": doc_name,
            "chapter_name": chapter_name,
            "content": f"# {chapter_name.replace('.md', '').replace('-', ' ').title()}\n\nThis is test content for {chapter_name}.",
            "word_count": 50,
            "section_count": 1,
        }

    @staticmethod
    def create_performance_test_scenarios() -> list[dict[str, Any]]:
        """Create a set of performance test scenarios."""
        return [
            {
                "name": "simple_document_creation",
                "query": 'Create a new document called "test_doc"',
                "expected_operations": ["create_document"],
                "max_tokens": {"simple": 200, "react": 500},
                "max_time": 5.0,
                "expected_files": ["test_doc/"],
            },
            {
                "name": "complex_document_with_chapters",
                "query": "Create a book about AI with 3 chapters: introduction, methods, and conclusion",
                "expected_operations": [
                    "create_document",
                    "create_chapter",
                    "create_chapter",
                    "create_chapter",
                ],
                "max_tokens": {"simple": 800, "react": 1500},
                "max_time": 15.0,
                "expected_files": [
                    "ai_book/",
                    "ai_book/01-introduction.md",
                    "ai_book/02-methods.md",
                    "ai_book/03-conclusion.md",
                ],
            },
            {
                "name": "document_query_and_analysis",
                "query": "List all documents and show their statistics",
                "expected_operations": ["list_documents", "get_statistics"],
                "max_tokens": {"simple": 300, "react": 600},
                "max_time": 10.0,
                "expected_files": [],  # Query operation, no new files
            },
        ]


def compare_agent_performance(
    metrics_simple: "AgentPerformanceMetrics", metrics_react: "AgentPerformanceMetrics"
) -> dict[str, Any]:
    """Compare performance between simple and react agents."""
    comparison = {
        "token_efficiency": {
            "simple": metrics_simple.token_usage or 0,
            "react": metrics_react.token_usage or 0,
            "ratio": (metrics_react.token_usage or 0)
            / max(metrics_simple.token_usage or 1, 1),
        },
        "speed_comparison": {
            "simple": metrics_simple.execution_time,
            "react": metrics_react.execution_time,
            "ratio": metrics_react.execution_time
            / max(metrics_simple.execution_time, 0.001),
        },
        "success_rate": {
            "simple": metrics_simple.success,
            "react": metrics_react.success,
            "both_successful": metrics_simple.success and metrics_react.success,
        },
        "tool_usage": {
            "simple": metrics_simple.tool_calls_count,
            "react": metrics_react.tool_calls_count,
            "difference": metrics_react.tool_calls_count
            - metrics_simple.tool_calls_count,
        },
    }

    return comparison


def generate_performance_summary(
    metrics_list: list["AgentPerformanceMetrics"], include_llm_evaluation: bool = False
) -> dict[str, Any]:
    """Generate a comprehensive performance summary with optional LLM evaluation metrics."""
    successful_tests = [m for m in metrics_list if m.success]
    failed_tests = [m for m in metrics_list if not m.success]

    summary = {
        "total_tests": len(metrics_list),
        "successful_tests": len(successful_tests),
        "failed_tests": len(failed_tests),
        "success_rate": (
            len(successful_tests) / len(metrics_list) if metrics_list else 0
        ),
        "average_execution_time": (
            sum(m.execution_time for m in metrics_list) / len(metrics_list)
            if metrics_list
            else 0
        ),
        "average_token_usage": (
            sum(m.token_usage or 0 for m in metrics_list) / len(metrics_list)
            if metrics_list
            else 0
        ),
        "total_token_usage": sum(m.token_usage or 0 for m in metrics_list),
        "total_execution_time": sum(m.execution_time for m in metrics_list),
        "performance_buckets": {
            "fast": len([m for m in metrics_list if m.execution_time < 1.0]),
            "medium": len([m for m in metrics_list if 1.0 <= m.execution_time < 5.0]),
            "slow": len([m for m in metrics_list if m.execution_time >= 5.0]),
        },
        "token_efficiency_buckets": {
            "efficient": len([m for m in metrics_list if (m.token_usage or 0) < 200]),
            "moderate": len(
                [m for m in metrics_list if 200 <= (m.token_usage or 0) < 500]
            ),
            "heavy": len([m for m in metrics_list if (m.token_usage or 0) >= 500]),
        },
    }

    # Add LLM evaluation metrics if requested and available (clean architecture)
    if include_llm_evaluation:
        # Extract LLM evaluation metrics from enhanced metrics (using clean architecture)
        llm_evaluations = []
        for m in metrics_list:
            if (
                hasattr(m, "llm_evaluation")
                and m.llm_evaluation
                and m.llm_evaluation.success
            ):
                llm_evaluations.append(m.llm_evaluation)

        if llm_evaluations:
            summary["llm_evaluation"] = {
                "total_evaluations": len(llm_evaluations),
                "successful_evaluations": len(llm_evaluations),
                "average_quality_score": sum(e.score for e in llm_evaluations)
                / len(llm_evaluations),
                "quality_distribution": {
                    "high": len([e for e in llm_evaluations if e.score >= 0.8]),
                    "medium": len([e for e in llm_evaluations if 0.6 <= e.score < 0.8]),
                    "low": len([e for e in llm_evaluations if e.score < 0.6]),
                },
            }

            # Add combined score metrics if available (using clean architecture)
            combined_scores = []
            for m in metrics_list:
                if hasattr(m, "combined_score") and m.combined_score > 0:
                    combined_scores.append(m.combined_score)

            if combined_scores:
                summary["combined_scoring"] = {
                    "average_combined_score": sum(combined_scores)
                    / len(combined_scores),
                    "combined_distribution": {
                        "excellent": len([s for s in combined_scores if s >= 0.8]),
                        "good": len([s for s in combined_scores if 0.6 <= s < 0.8]),
                        "needs_improvement": len(
                            [s for s in combined_scores if s < 0.6]
                        ),
                    },
                }

    return summary


# Complex LLM evaluation functions were removed in favor of clean architecture.
# For LLM evaluation capabilities, use:
#   from tests.evaluation.llm_evaluation_layer import enhance_test_metrics
#   enhanced = await enhance_test_metrics(performance_metrics, query, response)
#   if enhanced.llm_evaluation:
#       quality_score = enhanced.llm_evaluation.score
#       feedback = enhanced.llm_evaluation.feedback
