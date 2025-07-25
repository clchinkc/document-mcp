#!/bin/bash

# Installation Test Script for document-mcp
# Tests pip installation in a clean virtual environment
# Verifies that all major modules can be imported without errors

set -e  # Exit on any error

echo "🧪 Starting document-mcp Installation Test"
echo "=========================================="

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Create a temporary directory for testing
TEST_DIR=$(mktemp -d)
echo "📁 Test directory: $TEST_DIR"

# Cleanup function
cleanup() {
    echo "🧹 Cleaning up test environment..."
    rm -rf "$TEST_DIR"
}
trap cleanup EXIT

cd "$TEST_DIR"

# Create a clean virtual environment
echo "🐍 Creating clean virtual environment..."
python3 -m venv test_env
source test_env/bin/activate

# Upgrade pip to latest version
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install the package from the project root
echo "📦 Installing document-mcp from source..."
pip install "$PROJECT_ROOT"

# Test console script installation
echo "🔧 Testing console script installation..."
if ! command -v document-mcp &> /dev/null; then
    echo "❌ ERROR: document-mcp console script not found"
    exit 1
fi

echo "✅ Console scripts installed successfully"

# Test major module imports
echo "📚 Testing module imports..."

python3 << 'EOF'
import sys
import traceback

modules_to_test = [
    'document_mcp',
    'document_mcp.doc_tool_server',
    'document_mcp.models',
    'document_mcp.tools',
    'document_mcp.tools.document_tools',
    'document_mcp.tools.chapter_tools', 
    'document_mcp.tools.paragraph_tools',
    'document_mcp.tools.content_tools',
    'document_mcp.tools.safety_tools',
    'document_mcp.tools.batch_tools',
    'document_mcp.utils',
    'document_mcp.utils.file_operations',
    'document_mcp.utils.validation',
    'document_mcp.logger_config',
    'document_mcp.metrics_config',
]

failed_imports = []

for module in modules_to_test:
    try:
        __import__(module)
        print(f"✅ {module}")
    except Exception as e:
        print(f"❌ {module}: {str(e)}")
        failed_imports.append((module, str(e)))

if failed_imports:
    print(f"\n❌ {len(failed_imports)} module(s) failed to import:")
    for module, error in failed_imports:
        print(f"  - {module}: {error}")
    sys.exit(1)
else:
    print(f"\n✅ All {len(modules_to_test)} modules imported successfully")

EOF

# Test basic functionality
echo "🔍 Testing basic functionality..."

# Test MCP server help (should not crash)
python3 -c "
import subprocess
import sys
try:
    result = subprocess.run([sys.executable, '-m', 'document_mcp.doc_tool_server', '--help'], 
                          capture_output=True, text=True, timeout=10)
    if result.returncode != 0:
        print('❌ MCP server help failed')
        print(result.stderr)
        sys.exit(1)
    else:
        print('✅ MCP server help works')
except Exception as e:
    print(f'❌ MCP server test failed: {e}')
    sys.exit(1)
"

# Test console scripts
echo "🔧 Testing console scripts..."

# Test document-mcp help
if ! document-mcp --help > /dev/null 2>&1; then
    echo "❌ ERROR: document-mcp console script failed"
    exit 1
fi

echo "✅ Console scripts work correctly"

# Test package metadata
echo "📋 Testing package metadata..."
python3 << 'EOF'
import document_mcp
import importlib.metadata

try:
    # Check version is accessible
    version = document_mcp.__version__
    print(f"✅ Package version: {version}")
    
    # Check package distribution info using modern importlib.metadata
    dist = importlib.metadata.distribution('document-mcp')
    print(f"✅ Package name: {dist.metadata['Name']}")
    print(f"✅ Package version: {dist.version}")
    
except Exception as e:
    print(f"❌ Package metadata error: {e}")
    import sys
    sys.exit(1)
EOF

echo ""
echo "🎉 Installation Test PASSED!"
echo "✅ Package installs correctly"
echo "✅ All major modules import successfully"
echo "✅ Console scripts work"
echo "✅ Basic functionality verified"
echo "✅ Package metadata accessible"
echo ""
echo "📦 The document-mcp package is ready for production release!"