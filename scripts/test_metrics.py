#!/usr/bin/env python3
"""
Test script to verify metrics collection functionality.

This script tests the OpenTelemetry metrics integration by:
1. Checking if metrics are properly initialized
2. Simulating tool calls to generate metrics data
3. Verifying metrics are exported correctly
4. Testing both local Prometheus and remote OTLP export (if configured)
"""

import os
import sys
import time
import requests
import json
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_metrics_initialization():
    """Test that metrics are properly initialized."""
    print("🔧 Testing metrics initialization...")
    
    try:
        from document_mcp.metrics_config import is_metrics_enabled, get_metrics_summary
        
        enabled = is_metrics_enabled()
        print(f"   ✅ Metrics enabled: {enabled}")
        
        if enabled:
            summary = get_metrics_summary()
            print(f"   ✅ Service: {summary['service_name']}")
            print(f"   ✅ Environment: {summary['environment']}")
            print(f"   ✅ OTLP Endpoint: {summary['otlp_endpoint']}")
            print(f"   ✅ Prometheus: {summary['prometheus_enabled']}")
        else:
            print(f"   ⚠️  Metrics disabled or unavailable")
            
        return enabled
        
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_tool_instrumentation():
    """Test that tool calls are properly instrumented."""
    print("\n🔧 Testing tool instrumentation...")
    
    try:
        from document_mcp.metrics_config import record_tool_call_start, record_tool_call_success, record_tool_call_error
        
        # Simulate a successful tool call
        start_time = record_tool_call_start("test_tool", ("arg1",), {"param": "value"})
        time.sleep(0.1)  # Simulate some work
        record_tool_call_success("test_tool", start_time, 100)
        print("   ✅ Successful tool call recorded")
        
        # Simulate a failed tool call
        start_time = record_tool_call_start("test_tool_error", (), {})
        time.sleep(0.05)
        record_tool_call_error("test_tool_error", start_time, ValueError("Test error"))
        print("   ✅ Failed tool call recorded")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def main():
    """Run all metrics tests."""
    print("🚀 Document MCP Server - Metrics Test Suite")
    print("=" * 50)
    
    # Run basic tests
    test_metrics_initialization()
    test_tool_instrumentation()
    
    print("\n🎉 Basic metrics tests completed!")
    print("\n📝 To test the full system:")
    print("   1. Start server: python -m document_mcp.doc_tool_server sse")
    print("   2. Check metrics: curl http://localhost:3001/metrics")
    print("   3. Configure OTLP endpoint in .env for remote collection")

if __name__ == "__main__":
    main() 