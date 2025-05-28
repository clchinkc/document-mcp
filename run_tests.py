#!/usr/bin/env python3
"""
Test runner for the document-mcp project.
Runs tests for both the MCP server package and the example agent.
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, cwd=None):
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    if cwd:
        print(f"Working directory: {cwd}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, cwd=cwd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        return False

def main():
    """Run all tests."""
    project_root = Path(__file__).parent
    package_dir = project_root / "document_mcp"
    
    success = True
    
    # Test the MCP server package
    print("Testing MCP Server Package...")
    if not run_command([
        sys.executable, "-m", "pytest", 
        "test_doc_tool_server.py", "-v"
    ], cwd=package_dir):
        success = False
    
    # Test the example agent
    print("\nTesting Example Agent...")
    if not run_command([
        sys.executable, "-m", "pytest", 
        "example_agent/test_agent.py", "-v"
    ], cwd=project_root):
        success = False
    
    # Summary
    print(f"\n{'='*60}")
    if success:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
    print('='*60)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 