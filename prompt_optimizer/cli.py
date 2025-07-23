"""Simple CLI interface for the prompt optimizer."""

import asyncio
import sys
import time

from .core import PromptOptimizer


def print_help():
    """Print usage information."""
    print("🚀 Prompt Optimizer")
    print("=" * 20)
    print("\nUSAGE: python3 -m prompt_optimizer [AGENT]")
    print("\nAGENTS: simple, react, planner, all")
    print("\nEXAMPLES:")
    print("  python3 -m prompt_optimizer simple")
    print("  python3 -m prompt_optimizer all")


def print_summary(results: dict, duration: float):
    """Print optimization summary."""
    print(f"\n{'=' * 40}")
    print("📊 SUMMARY")
    print("=" * 40)

    improved = sum(1 for r in results.values() if r.keep_improvement)
    total_tokens = sum(r.token_change for r in results.values() if r.keep_improvement)

    for agent, result in results.items():
        status = "✅" if result.keep_improvement else "❌"
        print(f"{agent:>8}: {status} {result.reason}")
        if result.keep_improvement:
            print(f"          Tokens: {result.token_change:+d}")

    print(f"\nResults: {improved}/{len(results)} improved")
    print(f"Tokens saved: {total_tokens:+d}")
    print(f"Duration: {duration:.1f}s")


async def main():
    """Main CLI entry point."""
    # Parse arguments
    if len(sys.argv) < 2 or sys.argv[1] in ["-h", "--help"]:
        print_help()
        return

    print("🚀 Prompt Optimizer")
    print("=" * 20)

    agent_arg = sys.argv[1].lower()

    # Determine agents to optimize
    if agent_arg == "all":
        agents = ["simple", "react", "planner"]
    elif agent_arg in ["simple", "react", "planner"]:
        agents = [agent_arg]
    else:
        print(f"❌ Unknown agent: {agent_arg}")
        print_help()
        return

    # Run optimization
    optimizer = PromptOptimizer()
    results = {}
    start_time = time.time()

    for agent in agents:
        try:
            results[agent] = await optimizer.optimize_agent(agent)
        except Exception as e:
            print(f"❌ Failed to optimize {agent}: {e}")
            from .evaluation import OptimizationResult

            results[agent] = OptimizationResult(
                keep_improvement=False,
                reason=f"Failed: {e}",
                test_passed=False,
                token_change=0,
                test_count=0,
            )

    duration = time.time() - start_time
    print_summary(results, duration)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Error: {e}")
        sys.exit(1)
