#!/usr/bin/env python3
"""Document MCP Code Quality Manager.

A comprehensive script for managing code quality across the Document MCP project.
Integrates formatting, linting, and type checking using uv and ruff.

Note: For running tests, use scripts/run_pytest.py
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


class CodeQualityManager:
    """Manages code quality tools for the Document MCP project using uv and ruff."""

    def __init__(self, verbose: bool = False):
        """Initialize the quality checker."""
        self.verbose = verbose
        self.root_dir = Path(__file__).parent.parent
        self.target_dirs = ["src/", "document_mcp/", "tests/", "scripts/"]
        self.venv_path = self.root_dir / "venv"

    def _check_venv(self) -> bool:
        """Check if venv exists and is activated."""
        if not self.venv_path.exists():
            print("❌ Virtual environment 'venv' not found. Please create it first:")
            print("   python3 -m venv venv")
            print("   source venv/bin/activate  # On Windows: venv\\Scripts\\activate")
            return False

        # Check if we're in the virtual environment
        if sys.prefix == sys.base_prefix:
            print("❌ Virtual environment not activated. Please activate it first:")
            print("   source venv/bin/activate  # On Windows: venv\\Scripts\\activate")
            return False

        return True

    def _run_command(self, cmd: list[str], description: str) -> bool:
        """Run a command and return success status."""
        if not self._check_venv():
            return False

        if self.verbose:
            print(f"🔧 {description}")
            print(f"   Command: {' '.join(cmd)}")

        # Set environment variable to tell uv to use the active environment
        env = os.environ.copy()
        env["UV_PROJECT_ENVIRONMENT"] = str(self.venv_path)

        try:
            result = subprocess.run(
                cmd,
                cwd=self.root_dir,
                capture_output=not self.verbose,
                text=True,
                env=env,
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
            print(f"❌ {description} - Tool not found. Installing...")
            self._install_missing_tools()
            return False

    def _install_missing_tools(self):
        """Install missing quality tools using uv in the virtual environment."""
        if not self._check_venv():
            return

        tools = ["ruff", "mypy"]
        print(f"📦 Installing missing tools with uv: {', '.join(tools)}")

        # Set environment variable to tell uv to use the active environment
        env = os.environ.copy()
        env["UV_PROJECT_ENVIRONMENT"] = str(self.venv_path)

        subprocess.run(["uv", "add", "--dev"] + tools, cwd=self.root_dir, env=env)

    def format_code(self) -> bool:
        """Format code with ruff."""
        print("🎨 Formatting code...")

        # Run ruff format (replaces black and isort)
        format_success = self._run_command(
            ["uv", "run", "ruff", "format"] + self.target_dirs, "Ruff code formatting"
        )

        # Run ruff check with --fix for import sorting and other fixable issues
        fix_success = self._run_command(
            ["uv", "run", "ruff", "check", "--fix"] + self.target_dirs,
            "Ruff automatic fixes",
        )

        return format_success and fix_success

    def fix_code(self) -> bool:
        """Apply automatic fixes with ruff."""
        print("🔧 Applying automatic fixes...")

        # Ruff can fix many issues automatically
        return self._run_command(
            ["uv", "run", "ruff", "check", "--fix"] + self.target_dirs,
            "Ruff automatic fixes",
        )

    def lint_code(self) -> bool:
        """Run ruff linting."""
        print("🔍 Running ruff linting...")

        return self._run_command(
            ["uv", "run", "ruff", "check"] + self.target_dirs, "Ruff linting"
        )

    def type_check(self) -> bool:
        """Run mypy type checking."""
        print("🔎 Running mypy type checking...")

        return self._run_command(
            ["uv", "run", "mypy"] + self.target_dirs, "MyPy type checking"
        )

    def validate_docstrings(self) -> bool:
        """Run ruff docstring validation (replaces pydocstyle)."""
        print("📚 Running ruff docstring validation...")

        return self._run_command(
            ["uv", "run", "ruff", "check", "--select=D"] + self.target_dirs,
            "Ruff docstring validation",
        )

    def check_all(self) -> bool:
        """Run all code quality checks (linting, type checking, and docstring validation)."""
        print("🚀 Running code quality checks...")

        results = []

        # Linting (includes docstring validation)
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
            print(f"\n{'=' * 50}")
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
        choices=["check", "fix", "format", "lint", "typecheck", "docstring", "full"],
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
        "docstring": manager.validate_docstrings,
        "full": manager.full_quality_pass,
    }

    # Execute command
    success = commands[args.command]()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
