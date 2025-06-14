#!/usr/bin/env python3
"""
Test runner for Document MCP - runs all tests in virtual environment.
"""
import subprocess
import sys
import os
from pathlib import Path

def ensure_virtual_env():
    """Ensure we're running in a virtual environment."""
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        # Not in virtual environment
        venv_path = Path(".venv/bin/python")
        if venv_path.exists():
            print("Activating virtual environment...")
            # Re-run this script with the virtual environment Python
            os.execv(str(venv_path), [str(venv_path)] + sys.argv)
        else:
            print("⚠️  Virtual environment not found. Please create one with:")
            print("   python -m venv .venv")
            print("   source .venv/bin/activate")
            print("   pip install -r requirements.txt")
            sys.exit(1)

def check_environment():
    """Check if environment is properly set up."""
    # Check .env file
    env_file = Path(".env")
    if not env_file.exists():
        print("❌ .env file not found. Please create one with your API keys.")
        return False
    
    env_content = env_file.read_text()
    if "GOOGLE_API_KEY" not in env_content and "GEMINI_API_KEY" not in env_content:
        print("❌ API key not found in .env. Please add GOOGLE_API_KEY or GEMINI_API_KEY.")
        return False
    
    print("✅ Environment configured correctly")
    return True

def run_tests():
    """Run all tests with proper environment setup."""
    print("Document MCP - Comprehensive Test Runner")
    print("=" * 50)
    
    # Ensure virtual environment
    ensure_virtual_env()
    
    # Check environment
    if not check_environment():
        print("Please fix environment issues before running tests.")
        return 1
    
    # Set up environment variables for testing
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd())
    
    # Run server-side tests
    print("\n🔧 Running MCP Server Tests...")
    server_result = subprocess.run([
        sys.executable, "-m", "pytest", 
        "document_mcp/test_doc_tool_server.py",
        "-v", "--tb=short"
    ], env=env)
    
    # Run agent tests
    print("\n🤖 Running Agent Integration Tests...")
    agent_result = subprocess.run([
        sys.executable, "-m", "pytest", 
        "example_agent/test_agent.py",
        "-v", "--tb=short", "-n", "auto"
    ], env=env, cwd=Path.cwd())
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    server_status = "✅ PASSED" if server_result.returncode == 0 else "❌ FAILED"
    agent_status = "✅ PASSED" if agent_result.returncode == 0 else "❌ FAILED"
    
    print(f"MCP Server Tests: {server_status}")
    print(f"Agent Tests: {agent_status}")
    
    if server_result.returncode == 0 and agent_result.returncode == 0:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  Some tests failed. Check output above.")
        return 1

if __name__ == "__main__":
    exit_code = run_tests()
    sys.exit(exit_code) 