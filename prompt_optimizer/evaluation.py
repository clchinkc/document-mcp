"""
Performance evaluation system for prompt optimization using real agent benchmarks.

This module integrates with the existing evaluation test suite to provide 
comprehensive prompt performance scoring based on actual task completion.
"""

import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

# Import evaluation infrastructure
from tests.evaluation.test_agent_performance import AgentTestRunner
from tests.evaluation.config import get_test_scenarios


@dataclass
class OptimizationResult:
    """Result from prompt optimization evaluation."""
    keep_improvement: bool
    reason: str
    test_passed: bool
    token_change: int
    test_count: int
    performance_score: float = 0.0
    scenario_results: Dict[str, Dict] = None
    benchmark_comparison: Dict[str, float] = None
    
    def __post_init__(self):
        if self.scenario_results is None:
            self.scenario_results = {}
        if self.benchmark_comparison is None:
            self.benchmark_comparison = {}


@dataclass 
class BenchmarkResult:
    """Individual benchmark test result."""
    scenario_name: str
    success: bool
    execution_time: float
    token_usage: Optional[int] = None
    tool_calls_count: int = 0
    within_thresholds: bool = False
    performance_score: float = 0.0


class PerformanceEvaluator:
    """Advanced evaluator using real agent performance benchmarks."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        
    async def run_performance_benchmarks(self, agent_type: str) -> List[BenchmarkResult]:
        """Run performance benchmarks on the specified agent type."""
        print(f"🏃 Running performance benchmarks for {agent_type} agent...")
        
        # Get 5 key scenarios for optimization speed
        scenarios = get_test_scenarios()
        benchmark_scenarios = [
            s for s in scenarios 
            if s.get("category") in ["basic", "intermediate", "query"]
        ][:5]
        
        results = []
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            docs_root = Path(tmp_dir)
            runner = AgentTestRunner(docs_root)
            
            for scenario in benchmark_scenarios:
                try:
                    print(f"  📋 Running scenario: {scenario['name']}")
                    
                    # Run the scenario with real agent
                    metrics = await runner.run_agent_test(agent_type, scenario["query"])
                    
                    # Calculate simple performance index
                    performance_index = self._calculate_performance_index(metrics)
                    
                    result = BenchmarkResult(
                        scenario_name=scenario["name"],
                        success=metrics.success,
                        execution_time=metrics.execution_time,
                        token_usage=metrics.token_usage,
                        tool_calls_count=metrics.tool_calls_count,
                        within_thresholds=True,  # Not used in simple approach
                        performance_score=performance_index
                    )
                    
                    results.append(result)
                    
                    # Log results
                    status = "✅" if result.success else "❌"
                    print(f"    {status} Score: {result.performance_score:.2f}, Time: {result.execution_time:.2f}s")
                        
                except Exception as e:
                    print(f"    ❌ Scenario failed: {e}")
                    results.append(BenchmarkResult(scenario_name=scenario["name"], success=False))
        
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
    
    def _store_baseline_metrics(self, benchmark_results: List[BenchmarkResult]) -> dict:
        """Store baseline metrics for comparison."""
        if not benchmark_results:
            return {}
            
        successful_results = [r for r in benchmark_results if r.success]
        if not successful_results:
            return {}
            
        return {
            "success_rate": len(successful_results) / len(benchmark_results),
            "avg_time": sum(r.execution_time for r in successful_results) / len(successful_results),
            "avg_tokens": sum(r.token_usage or 0 for r in successful_results) / len(successful_results),
            "avg_performance_index": sum(r.performance_score for r in successful_results) / len(successful_results)
        }
    
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
        """Comprehensive evaluation using both functionality and performance tests."""
        
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
                benchmark_comparison={}
            )
        
        # Step 2: Run performance benchmarks
        print("\n🏆 Step 2: Performance Benchmarking")
        benchmark_results = await self.run_performance_benchmarks(agent_type)
        
        # Simple relative performance evaluation
        current_metrics = self._store_baseline_metrics(benchmark_results)
        success_rate = current_metrics.get("success_rate", 0.0)
        avg_performance_index = current_metrics.get("avg_performance_index", 0.0)
        avg_time = current_metrics.get("avg_time", 0.0)
        avg_tokens = current_metrics.get("avg_tokens", 0.0)
        
        # Create scenario results summary
        scenario_results = {
            r.scenario_name: {
                "success": r.success,
                "performance_index": r.performance_score,
                "execution_time": r.execution_time,
                "token_usage": r.token_usage,
            }
            for r in benchmark_results
        }
        
        # Simple decision logic: all tests must pass, then compare to baseline
        if success_rate < 1.0:  # All evaluation tests must pass
            reason = f"Evaluation tests failed ({success_rate:.1%} success) - rejecting changes"
            keep = False
        elif not baseline_metrics:  # No baseline to compare against
            reason = f"No baseline - accepting if all tests pass (performance index: {avg_performance_index:.2f})"
            keep = True
        else:  # Compare to baseline
            baseline_index = baseline_metrics.get("avg_performance_index", 0.0)
            if avg_performance_index > baseline_index:
                reason = f"Performance improved: {avg_performance_index:.2f} > {baseline_index:.2f} (baseline)"
                keep = True
            else:
                reason = f"Performance declined: {avg_performance_index:.2f} <= {baseline_index:.2f} (baseline)"
                keep = False
        
        # Performance comparison metrics
        benchmark_comparison = {
            "success_rate": success_rate,
            "avg_performance_index": avg_performance_index,
            "avg_time": avg_time,
            "avg_tokens": avg_tokens,
            "baseline_comparison": baseline_metrics
        }
        
        print("\n📊 PERFORMANCE EVALUATION:")
        print(f"  Success Rate: {success_rate:.1%}")
        print(f"  Avg Performance Index: {avg_performance_index:.2f}")
        print(f"  Avg Time: {avg_time:.2f}s")
        print(f"  Avg Tokens: {avg_tokens:.0f}")
        print(f"  Token Change: {token_change:+d}")
        if baseline_metrics:
            baseline_index = baseline_metrics.get("avg_performance_index", 0.0)
            print(f"  Baseline Index: {baseline_index:.2f}")
            improvement = avg_performance_index - baseline_index
            print(f"  Improvement: {improvement:+.2f}")
        
        return OptimizationResult(
            keep_improvement=keep,
            reason=reason,
            test_passed=functionality_results["passed"],
            token_change=token_change,
            test_count=functionality_results["test_count"],
            performance_score=avg_performance_index,
            scenario_results=scenario_results,
            benchmark_comparison=benchmark_comparison
        )


