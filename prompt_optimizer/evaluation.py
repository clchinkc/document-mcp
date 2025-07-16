"""
Performance evaluation system for prompt optimization using real agent benchmarks.

This module integrates with the clean evaluation architecture to provide 
comprehensive prompt performance scoring based on actual task completion,
enhanced with simple LLM-based qualitative assessment.
"""

import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

# Import clean evaluation infrastructure
from tests.evaluation.config import get_test_scenarios
from tests.evaluation.llm_evaluation_layer import enhance_test_metrics, EnhancedTestMetrics, LLMEvaluation
from src.agents.simple_agent.main import initialize_agent_and_mcp_server, process_single_user_query_with_metrics
from src.agents.react_agent.main import run_react_agent_with_metrics


@dataclass
class OptimizationResult:
    """Result from prompt optimization evaluation using clean architecture."""
    keep_improvement: bool
    reason: str
    test_passed: bool
    token_change: int
    test_count: int
    performance_score: float = 0.0
    scenario_results: Dict[str, Dict] = None
    benchmark_comparison: Dict[str, float] = None
    # Simple LLM evaluation (optional)
    llm_evaluation_score: float = 0.0
    llm_evaluation_feedback: str = ""
    
    def __post_init__(self):
        if self.scenario_results is None:
            self.scenario_results = {}
        if self.benchmark_comparison is None:
            self.benchmark_comparison = {}


@dataclass 
class BenchmarkResult:
    """Individual benchmark test result using clean architecture."""
    scenario_name: str
    success: bool
    execution_time: float
    token_usage: Optional[int] = None
    tool_calls_count: int = 0
    performance_score: float = 0.0
    # Simple LLM evaluation (optional)
    llm_evaluation: Optional[LLMEvaluation] = None
    quality_score: float = 0.0


class PerformanceEvaluator:
    """Clean evaluator using real agent performance benchmarks with simple LLM assessment."""
    
    def __init__(self, project_root: Path, enable_llm_evaluation: bool = True):
        self.project_root = project_root
        self.enable_llm_evaluation = enable_llm_evaluation
        
        if self.enable_llm_evaluation:
            print("✅ LLM evaluation enabled (using clean architecture)")
        else:
            print("ℹ️  LLM evaluation disabled")
        
    async def run_performance_benchmarks(self, agent_type: str) -> List[BenchmarkResult]:
        """Run performance benchmarks using clean architecture with optional LLM evaluation."""
        print(f"🏃 Running performance benchmarks for {agent_type} agent...")
        
        # Get 5 key scenarios for optimization speed
        scenarios = get_test_scenarios()
        benchmark_scenarios = [
            s for s in scenarios 
            if s.get("category") in ["basic", "intermediate", "query"]
        ][:5]
        
        results = []
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            import os
            old_doc_root = os.environ.get("DOCUMENT_ROOT_DIR")
            os.environ["DOCUMENT_ROOT_DIR"] = str(tmp_dir)
            
            try:
                for scenario in benchmark_scenarios:
                    try:
                        print(f"  📋 Running scenario: {scenario['name']}")
                        
                        # Run agent using clean architecture
                        if agent_type == "simple":
                            agent, mcp_server = await initialize_agent_and_mcp_server()
                            async with agent.run_mcp_servers():
                                response, performance_metrics = await process_single_user_query_with_metrics(
                                    agent, scenario["query"]
                                )
                        elif agent_type == "react":
                            history, performance_metrics = await run_react_agent_with_metrics(
                                scenario["query"], max_steps=5
                            )
                            response = {"summary": str(history[-1]) if history else "No response"}
                        else:
                            raise ValueError(f"Unknown agent type: {agent_type}")
                        
                        # Calculate simple performance index
                        performance_index = self._calculate_performance_index(performance_metrics)
                        
                        # Enhance with LLM evaluation if enabled
                        llm_evaluation = None
                        quality_score = 0.0
                        
                        if self.enable_llm_evaluation and performance_metrics.success:
                            try:
                                response_summary = response.get("summary", "") if isinstance(response, dict) else str(response)
                                enhanced_metrics = await enhance_test_metrics(
                                    performance_metrics, scenario["query"], response_summary
                                )
                                
                                if enhanced_metrics.llm_evaluation and enhanced_metrics.llm_evaluation.success:
                                    llm_evaluation = enhanced_metrics.llm_evaluation
                                    quality_score = llm_evaluation.score
                                    print(f"    🎯 LLM Quality Score: {quality_score:.2f}")
                                
                            except Exception as e:
                                print(f"    ⚠️ LLM evaluation failed: {e}")
                        
                        result = BenchmarkResult(
                            scenario_name=scenario["name"],
                            success=performance_metrics.success,
                            execution_time=performance_metrics.execution_time,
                            token_usage=performance_metrics.token_usage,
                            tool_calls_count=performance_metrics.tool_calls_count,
                            performance_score=performance_index,
                            llm_evaluation=llm_evaluation,
                            quality_score=quality_score
                        )
                        
                        results.append(result)
                        
                        # Log results
                        status = "✅" if result.success else "❌"
                        quality_info = f", Quality: {quality_score:.2f}" if quality_score > 0 else ""
                        print(f"    {status} Score: {result.performance_score:.2f}, Time: {result.execution_time:.2f}s{quality_info}")
                            
                    except Exception as e:
                        print(f"    ❌ Scenario failed: {e}")
                        results.append(BenchmarkResult(scenario_name=scenario["name"], success=False, execution_time=0.0))
            
            finally:
                # Restore environment
                if old_doc_root:
                    os.environ["DOCUMENT_ROOT_DIR"] = old_doc_root
                else:
                    os.environ.pop("DOCUMENT_ROOT_DIR", None)
        
        return results
    
    
    def _calculate_performance_index(self, metrics) -> float:
        """Calculate simple performance index: success / (time + tokens/100)."""
        if not metrics.success:
            return 0.0
            
        # Simple efficiency index: success rate divided by resource usage
        time_factor = max(0.1, metrics.execution_time)  # Avoid division by zero
        token_factor = (metrics.token_usage or 0) / 100  # Normalize tokens
        
        # Performance index: higher is better
        return 1.0 / (time_factor + token_factor + 0.1)  # Small constant to prevent infinite values
    
    def _calculate_combined_score(self, performance_index: float, quality_score: float) -> float:
        """Calculate combined score from performance and quality metrics."""
        if not self.enable_llm_evaluation or quality_score == 0.0:
            return performance_index
        
        # Weighted combination: 70% performance, 30% quality
        return (0.7 * performance_index) + (0.3 * quality_score)
    
    def _store_baseline_metrics(self, benchmark_results: List[BenchmarkResult]) -> dict:
        """Store baseline metrics for comparison, including LLM evaluation."""
        if not benchmark_results:
            return {}
            
        successful_results = [r for r in benchmark_results if r.success]
        if not successful_results:
            return {}
        
        # Calculate traditional metrics
        metrics = {
            "success_rate": len(successful_results) / len(benchmark_results),
            "avg_time": sum(r.execution_time for r in successful_results) / len(successful_results),
            "avg_tokens": sum(r.token_usage or 0 for r in successful_results) / len(successful_results),
            "avg_performance_index": sum(r.performance_score for r in successful_results) / len(successful_results)
        }
        
        # Add LLM evaluation metrics if available
        if self.enable_llm_evaluation:
            quality_scores = [r.quality_score for r in successful_results if r.quality_score > 0]
            if quality_scores:
                metrics["avg_quality_score"] = sum(quality_scores) / len(quality_scores)
                # Calculate combined score for optimization decision
                combined_scores = [
                    self._calculate_combined_score(r.performance_score, r.quality_score)
                    for r in successful_results
                ]
                metrics["avg_combined_score"] = sum(combined_scores) / len(combined_scores)
            else:
                metrics["avg_quality_score"] = 0.0
                metrics["avg_combined_score"] = metrics["avg_performance_index"]
        
        return metrics
    
    def run_unit_tests(self) -> dict:
        """Run unit/integration/e2e tests to ensure functionality."""
        print("🧪 Running functionality tests...")
        
        cmd = [sys.executable, "-m", "pytest", "tests/unit/", "tests/integration/", "tests/e2e/", "--tb=short"]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, 
                                    cwd=self.project_root, timeout=600)
            
            passed = result.returncode == 0
            test_count = self._count_tests(result.stdout)
            
            if not passed:
                print(f"❌ Tests failed ({test_count} tests)")
                print("Error:", result.stderr[-500:])
            
            return {"passed": passed, "test_count": test_count, "duration": time.time()}
            
        except subprocess.TimeoutExpired:
            print("⏰ Tests timed out after 10 minutes")
            return {"passed": False, "test_count": 0, "duration": 600}
        except Exception as e:
            print(f"❌ Error running tests: {e}")
            return {"passed": False, "test_count": 0, "duration": 0}
    
    def _count_tests(self, output: str) -> int:
        """Extract test count from pytest output."""
        for line in reversed(output.split('\n')):
            if 'passed' in line:
                words = line.split()
                for i, word in enumerate(words):
                    if word == 'passed' and i > 0 and words[i-1].isdigit():
                        return int(words[i-1])
        return 0
    
    async def evaluate_change(self, agent_type: str, baseline_tokens: int, 
                            improved_tokens: int, baseline_metrics: dict = None) -> OptimizationResult:
        """Comprehensive evaluation using functionality tests, performance benchmarks, and LLM assessment."""
        
        token_change = baseline_tokens - improved_tokens  # Positive = reduction
        
        # Step 1: Run functionality tests (critical requirement)
        print("\n📋 Step 1: Functionality Validation")
        functionality_results = self.run_unit_tests()
        
        if not functionality_results["passed"]:
            return OptimizationResult(
                keep_improvement=False,
                reason="Functionality tests failed - prompt breaks core system",
                test_passed=False,
                token_change=token_change,
                test_count=functionality_results["test_count"],
                performance_score=0.0,
                scenario_results={},
                benchmark_comparison={},
                llm_evaluation_score=0.0,
                llm_evaluation_feedback="Tests failed - no LLM evaluation performed"
            )
        
        # Step 2: Run performance benchmarks with LLM evaluation
        print(f"\n🏆 Step 2: Performance Benchmarking{'with LLM Evaluation' if self.enable_llm_evaluation else ''}")
        benchmark_results = await self.run_performance_benchmarks(agent_type)
        
        # Enhanced performance evaluation with LLM metrics
        current_metrics = self._store_baseline_metrics(benchmark_results)
        success_rate = current_metrics.get("success_rate", 0.0)
        avg_performance_index = current_metrics.get("avg_performance_index", 0.0)
        avg_quality_score = current_metrics.get("avg_quality_score", 0.0)
        avg_combined_score = current_metrics.get("avg_combined_score", avg_performance_index)
        avg_time = current_metrics.get("avg_time", 0.0)
        avg_tokens = current_metrics.get("avg_tokens", 0.0)
        
        # Generate LLM evaluation feedback
        llm_feedback = self._generate_llm_evaluation_feedback(benchmark_results)
        
        # Create enhanced scenario results summary
        scenario_results = {
            r.scenario_name: {
                "success": r.success,
                "performance_index": r.performance_score,
                "execution_time": r.execution_time,
                "token_usage": r.token_usage,
                "quality_score": r.quality_score,
                "combined_score": self._calculate_combined_score(r.performance_score, r.quality_score),
                "llm_feedback": r.llm_evaluation.feedback if r.llm_evaluation else ""
            }
            for r in benchmark_results
        }
        
        # Enhanced decision logic: consider both performance and quality
        if success_rate < 1.0:  # All evaluation tests must pass
            reason = f"Evaluation tests failed ({success_rate:.1%} success) - rejecting changes"
            keep = False
        elif not baseline_metrics:  # No baseline to compare against
            if self.enable_llm_evaluation:
                reason = f"No baseline - accepting if all tests pass (combined score: {avg_combined_score:.2f})"
            else:
                reason = f"No baseline - accepting if all tests pass (performance index: {avg_performance_index:.2f})"
            keep = True
        else:  # Compare to baseline using appropriate metric
            if self.enable_llm_evaluation and "avg_combined_score" in baseline_metrics:
                baseline_score = baseline_metrics.get("avg_combined_score", 0.0)
                current_score = avg_combined_score
                metric_name = "combined score"
            else:
                baseline_score = baseline_metrics.get("avg_performance_index", 0.0)
                current_score = avg_performance_index
                metric_name = "performance index"
            
            if current_score > baseline_score:
                reason = f"{metric_name.title()} improved: {current_score:.2f} > {baseline_score:.2f} (baseline)"
                keep = True
            else:
                reason = f"{metric_name.title()} declined: {current_score:.2f} <= {baseline_score:.2f} (baseline)"
                keep = False
        
        # Enhanced performance comparison metrics
        benchmark_comparison = {
            "success_rate": success_rate,
            "avg_performance_index": avg_performance_index,
            "avg_time": avg_time,
            "avg_tokens": avg_tokens,
            "baseline_comparison": baseline_metrics
        }
        
        # Add LLM evaluation metrics if available
        if self.enable_llm_evaluation:
            benchmark_comparison.update({
                "avg_quality_score": avg_quality_score,
                "avg_combined_score": avg_combined_score,
                "llm_evaluation_enabled": True
            })
        
        # Enhanced reporting
        print("\n📊 PERFORMANCE EVALUATION:")
        print(f"  Success Rate: {success_rate:.1%}")
        print(f"  Avg Performance Index: {avg_performance_index:.2f}")
        print(f"  Avg Time: {avg_time:.2f}s")
        print(f"  Avg Tokens: {avg_tokens:.0f}")
        print(f"  Token Change: {token_change:+d}")
        
        if self.enable_llm_evaluation:
            print(f"  Avg Quality Score: {avg_quality_score:.2f}")
            print(f"  Avg Combined Score: {avg_combined_score:.2f}")
        
        if baseline_metrics:
            if self.enable_llm_evaluation and "avg_combined_score" in baseline_metrics:
                baseline_score = baseline_metrics.get("avg_combined_score", 0.0)
                current_score = avg_combined_score
                print(f"  Baseline Combined Score: {baseline_score:.2f}")
                improvement = current_score - baseline_score
                print(f"  Combined Score Improvement: {improvement:+.2f}")
            else:
                baseline_index = baseline_metrics.get("avg_performance_index", 0.0)
                print(f"  Baseline Performance Index: {baseline_index:.2f}")
                improvement = avg_performance_index - baseline_index
                print(f"  Performance Improvement: {improvement:+.2f}")
        
        return OptimizationResult(
            keep_improvement=keep,
            reason=reason,
            test_passed=functionality_results["passed"],
            token_change=token_change,
            test_count=functionality_results["test_count"],
            performance_score=avg_performance_index,
            scenario_results=scenario_results,
            benchmark_comparison=benchmark_comparison,
            llm_evaluation_score=avg_quality_score,
            llm_evaluation_feedback=llm_feedback
        )
    
    def _generate_llm_evaluation_feedback(self, benchmark_results: List[BenchmarkResult]) -> str:
        """Generate consolidated LLM evaluation feedback."""
        if not self.enable_llm_evaluation:
            return "LLM evaluation not enabled"
        
        feedback_parts = []
        successful_evaluations = [r for r in benchmark_results if r.llm_evaluation and r.success]
        
        if not successful_evaluations:
            return "No successful LLM evaluations available"
        
        # Extract common themes from feedback
        quality_scores = [r.quality_score for r in successful_evaluations]
        avg_quality = sum(quality_scores) / len(quality_scores)
        
        feedback_parts.append(f"Average Quality Score: {avg_quality:.2f}")
        
        # Collect feedback from LLM evaluations
        feedback_entries = []
        for result in successful_evaluations:
            if result.llm_evaluation and result.llm_evaluation.feedback:
                feedback_entries.append(result.llm_evaluation.feedback)
        
        # Add feedback summary
        if feedback_entries:
            feedback_parts.append("Sample feedback:")
            for i, feedback in enumerate(feedback_entries[:3]):  # Show first 3
                feedback_parts.append(f"  - {feedback[:100]}{'...' if len(feedback) > 100 else ''}")
            
            if len(feedback_entries) > 3:
                feedback_parts.append(f"  ... and {len(feedback_entries) - 3} more evaluations")
        
        return "\n".join(feedback_parts)


