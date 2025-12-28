"""CLI entry point for DSPy optimizer using MIPROv2."""

import argparse
import sys


def main():
    """Run DSPy optimizer with specified configuration."""
    # Import here to avoid RuntimeWarning with sys.modules
    from .optimizer import run_multi_model_comparison
    from .optimizer import run_optimization
    from .optimizer import run_variant_comparison

    parser = argparse.ArgumentParser(
        description="DSPy Tool Selection Optimizer (MIPROv2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings (MIPROv2 light mode, full variant)
  python -m dspy_optimizer

  # Use medium or heavy optimization intensity
  python -m dspy_optimizer --mode medium
  python -m dspy_optimizer --mode heavy

  # Compare all prompt variants
  python -m dspy_optimizer --compare-variants

  # Compare all models
  python -m dspy_optimizer --compare-models

  # Write optimized prompts back to source files
  python -m dspy_optimizer --write-back
""",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model ID (default: uses DEFAULT_MODEL from config)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="light",
        choices=["light", "medium", "heavy"],
        help="MIPROv2 optimization intensity (default: light)",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="full",
        choices=["compact", "full", "minimal"],
        help="Prompt variant (default: full)",
    )
    parser.add_argument(
        "--compare-variants",
        action="store_true",
        help="Compare all prompt variants",
    )
    parser.add_argument(
        "--compare-models",
        action="store_true",
        help="Compare all models",
    )
    parser.add_argument(
        "--write-back",
        action="store_true",
        help="Write optimized prompts back to source files",
    )

    args = parser.parse_args()

    try:
        if args.compare_variants:
            results = run_variant_comparison(model=args.model, auto_mode=args.mode)
            print(f"\n✓ Compared {len(results)} variants")
        elif args.compare_models:
            results = run_multi_model_comparison(auto_mode=args.mode)
            print(f"\n✓ Compared {len(results)} models")
        else:
            result = run_optimization(
                model=args.model,
                variant=args.variant,
                auto_mode=args.mode,
                write_back=args.write_back,
            )
            if result.improvement > 0:
                print(f"\n✓ Composite score improved by {result.improvement:+.3f}")
            else:
                print(f"\n→ No improvement ({result.improvement:+.3f})")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Optimization failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
