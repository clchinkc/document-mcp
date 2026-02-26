#!/usr/bin/env python3
"""
Flaky Test Detection Tool

Runs integration tests multiple times and identifies flaky tests that fail inconsistently.
Useful for local development and CI/CD validation.

Usage:
    python3 scripts/development/flaky_test_detector.py [--runs N] [--test-path PATH]
"""

import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import click


def run_pytest(test_path: str, run_number: int) -> tuple[int, str]:
    """Run pytest and return exit code and output."""
    cmd = [
        "uv",
        "run",
        "pytest",
        test_path,
        "-v",
        "--tb=short",
        "--strict-markers",
        "--disable-warnings",
        "-q",
    ]

    result = subprocess.run(
        cmd, capture_output=True, text=True
    )
    return result.returncode, result.stdout + result.stderr


def parse_pytest_output(output: str) -> dict[str, str]:
    """Parse pytest output to extract test results."""
    results = {}
    for line in output.split("\n"):
        if " PASSED" in line or " FAILED" in line:
            # Format: tests/integration/test_file.py::test_name PASSED/FAILED
            parts = line.split(" ")
            for i, part in enumerate(parts):
                if part in ["PASSED", "FAILED"]:
                    test_name = parts[0].strip()
                    if test_name and "::" in test_name:
                        results[test_name] = part.lower()
                    break
    return results


def analyze_results(
    all_results: dict[str, list[str]], total_runs: int
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """Analyze test results to identify flaky tests."""
    flaky_tests = []
    stable_passed = []
    stable_failed = []

    for test_name, results in sorted(all_results.items()):
        passed_count = sum(1 for r in results if r == "passed")
        failed_count = sum(1 for r in results if r == "failed")
        total = passed_count + failed_count

        if total == 0:
            continue

        failure_rate = (failed_count / total) * 100

        # Flaky: Sometimes passes, sometimes fails
        if 0 < failed_count < total:
            flaky_tests.append(
                {
                    "name": test_name,
                    "passed": passed_count,
                    "failed": failed_count,
                    "total_runs": total,
                    "failure_rate": failure_rate,
                    "pattern": "".join(["✓" if r == "passed" else "✗" for r in results]),
                }
            )
        elif failed_count == 0:
            stable_passed.append(test_name)
        else:
            stable_failed.append(test_name)

    # Sort flaky tests by failure rate (descending)
    flaky_tests.sort(key=lambda x: x["failure_rate"], reverse=True)

    return flaky_tests, stable_passed, stable_failed


def print_report(
    flaky_tests: list[dict[str, Any]],
    stable_passed: list[str],
    stable_failed: list[str],
    total_runs: int,
) -> None:
    """Print human-readable report."""
    print("\n" + "=" * 80)
    print("FLAKY TEST DETECTION REPORT")
    print("=" * 80 + "\n")

    print(f"Total Runs: {total_runs}")
    print(f"Unique Tests: {len(stable_passed) + len(stable_failed) + len(flaky_tests)}")
    print(f"  ✓ Stable (Passed): {len(stable_passed)}")
    print(f"  ✗ Stable (Failed): {len(stable_failed)}")
    print(f"  ⚠ Flaky: {len(flaky_tests)}\n")

    if flaky_tests:
        print("FLAKY TESTS DETECTED:")
        print("-" * 80)
        print(f"{'Test Name':<60} {'Rate':<8} {'Pattern':<15}")
        print("-" * 80)

        for test in flaky_tests:
            short_name = test["name"]
            if len(short_name) > 60:
                short_name = "..." + short_name[-57:]

            print(
                f"{short_name:<60} {test['failure_rate']:>6.1f}%  {test['pattern']:<15}"
            )

        print(
            "\nRecommendations:\n"
            "  1. Review test setup and teardown for race conditions\n"
            "  2. Check for dependency on external services/APIs\n"
            "  3. Look for non-deterministic behavior\n"
            "  4. Check system resource availability\n"
            "  5. Review test isolation and cleanup"
        )
    else:
        print("✅ No flaky tests detected! All tests are stable.")

    if stable_failed:
        print("\n" + "=" * 80)
        print("CONSISTENTLY FAILING TESTS (All runs failed):")
        print("=" * 80)
        for test in stable_failed[:10]:
            print(f"  ✗ {test}")
        if len(stable_failed) > 10:
            print(f"  ... and {len(stable_failed) - 10} more")

    print("\n" + "=" * 80 + "\n")


def save_json_report(
    flaky_tests: list[dict[str, Any]],
    stable_passed: list[str],
    stable_failed: list[str],
    total_runs: int,
    output_path: str,
) -> None:
    """Save results to JSON file for CI/CD integration."""
    report = {
        "total_runs": total_runs,
        "summary": {
            "total_unique_tests": len(stable_passed) + len(stable_failed) + len(flaky_tests),
            "stable_passed": len(stable_passed),
            "stable_failed": len(stable_failed),
            "flaky_tests_count": len(flaky_tests),
        },
        "flaky_tests": flaky_tests,
        "stable_passed_tests": stable_passed,
        "stable_failed_tests": stable_failed,
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"✓ JSON report saved to {output_path}")


@click.command()
@click.option(
    "--runs",
    default=5,
    help="Number of test runs to perform (default: 5)",
    type=int,
)
@click.option(
    "--test-path",
    default="tests/integration/",
    help="Path to integration tests (default: tests/integration/)",
    type=str,
)
@click.option(
    "--save-json",
    default=None,
    help="Save JSON report to this file",
    type=str,
)
def main(runs: int, test_path: str, save_json: str | None) -> None:
    """Run integration tests multiple times to detect flaky tests."""
    if runs < 2:
        click.echo("Error: --runs must be at least 2", err=True)
        sys.exit(1)

    click.echo(f"Running integration tests {runs} times...\n")

    all_results: dict[str, list[str]] = defaultdict(list)

    for run_num in range(1, runs + 1):
        click.echo(f"Run {run_num}/{runs}...")

        exit_code, output = run_pytest(test_path, run_num)

        # Parse results
        results = parse_pytest_output(output)

        # Track all test results
        for test_name, status in results.items():
            all_results[test_name].append(status)

        click.echo(f"  Completed: {len(results)} tests processed")

    # Analyze results
    flaky_tests, stable_passed, stable_failed = analyze_results(all_results, runs)

    # Print report
    print_report(flaky_tests, stable_passed, stable_failed, runs)

    # Save JSON report if requested
    if save_json:
        save_json_report(flaky_tests, stable_passed, stable_failed, runs, save_json)

    # Exit with error if flaky or failed tests detected
    if flaky_tests or stable_failed:
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
