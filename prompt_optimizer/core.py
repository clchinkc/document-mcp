"""Core prompt optimization logic.

This module contains the main PromptOptimizer class with focused functionality.
"""

import os
import shutil
import sys
import time
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# Add src to path for agent imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from .evaluation import OptimizationResult
from .evaluation import PerformanceEvaluator


class PromptOptimizer:
    """Prompt optimizer that focuses on test-based validation."""

    def __init__(self):
        """Initialize the prompt optimizer."""
        self.project_root = Path(__file__).parent.parent
        self.backup_dir = Path("prompt_backups")
        self.backup_dir.mkdir(exist_ok=True)
        self.evaluator = PerformanceEvaluator(self.project_root)

        self.agent_files = {
            "simple": "src/agents/simple_agent/prompts.py",
            "react": "src/agents/react_agent/prompts.py",
        }

    def get_current_prompt(self, agent_type: str) -> str:
        """Get the current prompt content for an agent."""
        if agent_type == "simple":
            from src.agents.simple_agent.prompts import get_simple_agent_system_prompt

            return get_simple_agent_system_prompt()
        elif agent_type == "react":
            from src.agents.react_agent.prompts import get_react_system_prompt

            return get_react_system_prompt()
        else:
            raise ValueError(f"Unknown agent type: {agent_type}. Available: simple, react")

    def backup_prompt(self, agent_type: str) -> str:
        """Backup current prompt file."""
        source_file = Path(self.agent_files[agent_type])
        backup_file = self.backup_dir / f"{agent_type}_prompt_backup_{int(time.time())}.py"

        if source_file.exists():
            shutil.copy2(source_file, backup_file)
            return str(backup_file)
        return ""

    def restore_prompt(self, agent_type: str, backup_path: str) -> bool:
        """Restore prompt from backup."""
        try:
            source_file = Path(self.agent_files[agent_type])
            backup_file = Path(backup_path)

            if backup_file.exists():
                shutil.copy2(backup_file, source_file)
                return True
            return False
        except Exception as e:
            print(f"[ERROR] Error restoring prompt: {e}")
            return False

    def apply_improved_prompt(self, agent_type: str, improved_prompt: str) -> bool:
        """Apply the improved prompt to the agent file."""
        file_path = Path(self.agent_files[agent_type])

        # Template mapping for agent types
        templates = {
            "simple": "get_simple_agent_system_prompt",
            "react": "get_react_system_prompt",
        }

        if agent_type not in templates:
            return False

        file_content = f'''from ..shared.tool_descriptions import get_tool_descriptions_for_agent

def {templates[agent_type]}() -> str:
    """Generate the {agent_type} agent system prompt with dynamic tool descriptions."""
    tool_descriptions = get_tool_descriptions_for_agent("{agent_type}")

    return f"""{improved_prompt}"""
'''

        try:
            with open(file_path, "w") as f:
                f.write(file_content)
            return True
        except Exception as e:
            print(f"[ERROR] Error applying prompt: {e}")
            return False

    async def get_llm_optimization(self, agent_type: str, current_prompt: str) -> str:
        """Get an improved prompt from LLM."""
        from pydantic_ai import Agent

        from src.agents.shared.config import load_llm_config

        optimization_request = f"""You are helping optimize a {agent_type} agent prompt. Make MINIMAL, SAFE improvements only.

CRITICAL REQUIREMENTS:
- Make it more concise and efficient
- Keep ALL existing instructions
- Keep the {{tool_descriptions}} placeholder EXACTLY as is
- Do NOT remove any tool usage instructions
- Do NOT change the core logic or behavior
- Make only small improvements for clarity, precision, and structure

Current prompt:
{current_prompt}

Return ONLY the improved prompt text - no explanations or markdown:"""

        print(f"🤖 Requesting LLM optimization for {agent_type} agent...")

        try:
            import asyncio

            llm = await load_llm_config()
            optimizer_agent = Agent(llm, output_type=str)

            result = await asyncio.wait_for(optimizer_agent.run(optimization_request), timeout=120)

            improved_prompt = result.output.strip()
            return improved_prompt

        except asyncio.TimeoutError:
            raise Exception("LLM request timed out")
        except Exception as e:
            if "api" in str(e).lower() or "key" in str(e).lower():
                print("[INFO] Hint: Check your API keys (OPENAI_API_KEY or GEMINI_API_KEY)")
            raise Exception(f"LLM optimization failed: {e}")

    async def get_baseline_metrics(self, agent_type: str) -> dict:
        """Get baseline performance metrics for comparison."""
        print(f"[DATA] Getting baseline metrics for {agent_type} agent...")

        # Run evaluation benchmarks to get baseline
        benchmark_results = await self.evaluator.run_performance_benchmarks(agent_type)
        baseline_metrics = self.evaluator._store_baseline_metrics(benchmark_results)

        if not baseline_metrics:
            print("[WARN] No baseline metrics available")

        return baseline_metrics

    async def optimize_agent(self, agent_type: str) -> OptimizationResult:
        """Optimize a single agent's prompt using relative improvement."""
        print(f"\n[START] OPTIMIZING {agent_type.upper()} AGENT")
        print("=" * 50)

        # Get current prompt and metrics
        current_prompt = self.get_current_prompt(agent_type)
        baseline_tokens = len(current_prompt) // 4  # Rough token estimate

        print(f"[DATA] Baseline: ~{baseline_tokens} tokens")

        # Get baseline performance metrics
        baseline_metrics = await self.get_baseline_metrics(agent_type)

        # Backup current version
        backup_path = self.backup_prompt(agent_type)

        try:
            # Get improved prompt from LLM
            improved_prompt = await self.get_llm_optimization(agent_type, current_prompt)
            improved_tokens = len(improved_prompt) // 4

            print(f"📏 Improved: ~{improved_tokens} tokens ({improved_tokens - baseline_tokens:+d})")

            # Apply the improvement
            if not self.apply_improved_prompt(agent_type, improved_prompt):
                return OptimizationResult(
                    keep_improvement=False,
                    reason="Failed to apply improved prompt",
                    test_passed=False,
                    token_change=0,
                    test_count=0,
                )

            # Run evaluation with baseline comparison
            result = await self.evaluator.evaluate_change(
                agent_type, baseline_tokens, improved_tokens, baseline_metrics
            )

            print("\n🧠 EVALUATION:")
            print(f"  Performance Index: {result.performance_score:.2f}")
            print(f"  Token Change: {result.token_change:+d}")
            print(f"  Tests: {'[OK]' if result.test_passed else '[FAIL]'}")

            if result.scenario_results:
                successful = sum(1 for r in result.scenario_results.values() if r["success"])
                total = len(result.scenario_results)
                print(f"  Benchmarks: {successful}/{total}")

            if result.keep_improvement:
                print(f"\n[OK] KEEPING: {result.reason}")
            else:
                print(f"\n[FAIL] REJECTING: {result.reason}")
                if backup_path:
                    self.restore_prompt(agent_type, backup_path)

            return result

        except Exception as e:
            print(f"[ERROR] Optimization failed: {e}")
            if backup_path:
                self.restore_prompt(agent_type, backup_path)

            return OptimizationResult(
                keep_improvement=False,
                reason=f"Optimization failed: {e}",
                test_passed=False,
                token_change=0,
                test_count=0,
            )
