#!/usr/bin/env python3
"""
Simple test runner for Document MCP.
"""
import subprocess
import sys
import os

def run_tests():
    """Run tests with proper configuration."""
    print("Document MCP - Test Runner")
    print("=" * 40)
    
    # Set environment for testing
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    
    # Run all tests together with simplified options
    print("\n🧪 Running All Tests...")
    result = subprocess.run([
        sys.executable, "-m", "pytest", 
        "-v", "--tb=short"
    ], env=env)
    
    # Summary
    print("\n" + "=" * 40)
    if result.returncode == 0:
        print("✅ All tests passed!")
        return 0
    else:
        print("❌ Some tests failed.")
        return 1

if __name__ == "__main__":
    exit_code = run_tests()
    sys.exit(exit_code) 