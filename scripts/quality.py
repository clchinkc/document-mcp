#!/usr/bin/env python3
"""
Document MCP Code Quality Manager

A comprehensive script for managing code quality across the Document MCP project.
Integrates formatting, linting, and type checking.

Note: For running tests, use scripts/run_pytest.py
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


class CodeQualityManager:
    """Manages code quality tools for the Document MCP project."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.root_dir = Path(__file__).parent.parent
        self.target_dirs = ["src/", "document_mcp/", "tests/", "scripts/"]

    def _run_command(self, cmd: List[str], description: str) -> bool:
        """Run a command and return success status."""
        if self.verbose:
            print(f"🔧 {description}")
            print(f"   Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd, cwd=self.root_dir, capture_output=not self.verbose, text=True
            )

            if result.returncode == 0:
                if self.verbose:
                    print(f"✅ {description} - SUCCESS")
                return True
            else:
                print(f"❌ {description} - FAILED")
                if not self.verbose and result.stdout:
                    print(result.stdout)
                if not self.verbose and result.stderr:
                    print(result.stderr)
                return False

        except FileNotFoundError:
            print(
                f"❌ {description} - Command not found. Is '{cmd[0]}' installed and in your PATH?"
            )
            print(
                "   Please ensure all development dependencies from 'requirements.txt' are installed."
            )
            return False

    def format_code(self) -> bool:
        """Format code with black and isort."""
        print("🎨 Formatting code...")

        # Run black
        black_success = self._run_command(
            [sys.executable, "-m", "black"] + self.target_dirs, "Black code formatting"
        )

        # Run isort
        isort_success = self._run_command(
            [sys.executable, "-m", "isort"] + self.target_dirs,
            "Import sorting with isort",
        )

        return black_success and isort_success

    def fix_code(self) -> bool:
        """Apply automatic fixes with autoflake."""
        print("🔧 Applying automatic fixes...")

        # Find all Python files
        python_files = []
        for pattern in ["**/*.py"]:
            for target_dir in self.target_dirs:
                python_files.extend(
                    str(p) for p in self.root_dir.glob(f"{target_dir}{pattern}")
                )

        if not python_files:
            print("📁 No Python files found")
            return True

        cmd = [
            sys.executable,
            "-m",
            "autoflake",
            "--remove-all-unused-imports",
            "--remove-unused-variables",
            "--remove-duplicate-keys",
            "--in-place",
        ] + python_files

        return self._run_command(cmd, "Fixing with autoflake")

    def lint_code(self) -> bool:
        """Run flake8 linting."""
        print("🔍 Running flake8 linting...")

        return self._run_command(
            [sys.executable, "-m", "flake8"]
            + self.target_dirs
            + ["--count", "--statistics"],
            "Flake8 linting",
        )

    def type_check(self) -> bool:
        """Run mypy type checking."""
        print("🔎 Running mypy type checking...")

        return self._run_command(
            [sys.executable, "-m", "mypy"] + self.target_dirs, "MyPy type checking"
        )

    def check_all(self) -> bool:
        """Run all code quality checks (linting and type checking only)."""
        print("🚀 Running code quality checks...")

        results = []

        # Linting
        results.append(self.lint_code())

        # Type checking
        results.append(self.type_check())

        success = all(results)

        if success:
            print("🎉 All code quality checks passed!")
        else:
            print("⚠️  Some code quality checks failed")

        print("\nNote: Run 'python scripts/run_pytest.py' to execute tests")

        return success

    def full_quality_pass(self) -> bool:
        """Run complete quality pass: fix, format, then check."""
        print("🔥 Running FULL quality pass...")

        steps = [
            ("Automatic fixes", self.fix_code),
            ("Code formatting", self.format_code),
            ("Code quality checks", self.check_all),
        ]

        for description, step_func in steps:
            print(f"\n{'='*50}")
            print(f"STEP: {description}")
            print("=" * 50)

            if not step_func():
                print(f"❌ Failed at step: {description}")
                return False

        print("\n🎉 FULL quality pass completed successfully!")
        print("Note: Run 'python scripts/run_pytest.py' to execute tests")
        return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Document MCP Code Quality Manager")
    parser.add_argument(
        "command",
        choices=["check", "fix", "format", "lint", "typecheck", "full"],
        help="Quality command to run",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed output"
    )

    args = parser.parse_args()

    manager = CodeQualityManager(verbose=args.verbose)

    # Command mapping
    commands = {
        "check": manager.check_all,
        "fix": manager.fix_code,
        "format": manager.format_code,
        "lint": manager.lint_code,
        "typecheck": manager.type_check,
        "full": manager.full_quality_pass,
    }

    # Execute command
    success = commands[args.command]()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
