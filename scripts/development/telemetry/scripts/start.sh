#!/bin/bash
# Automatic Document MCP Telemetry Startup Script
# This script automatically starts all required services and pushes metrics to Grafana Cloud

set -e

echo "🚀 Document MCP Automatic Telemetry Startup"
echo "=========================================="

# Get the script directory and main project directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Check if we have the prometheus config
PROMETHEUS_CONFIG="$SCRIPT_DIR/../config/prometheus.yml"
if [[ ! -f "$PROMETHEUS_CONFIG" ]]; then
    echo "❌ prometheus.yml not found at $PROMETHEUS_CONFIG"
    exit 1
fi

# Check if Prometheus is installed
if ! command -v prometheus &> /dev/null; then
    echo "❌ Prometheus not found. Please install it first:"
    echo "   brew install prometheus"
    exit 1
fi

# Check if Python/uv is available
if ! command -v uv &> /dev/null; then
    echo "❌ uv not found. Please install it first."
    exit 1
fi

echo "✅ Prerequisites check passed"

# Kill any existing services on our ports
echo "🧹 Cleaning up existing services..."
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
lsof -ti:9090 | xargs kill -9 2>/dev/null || true
pkill -f prometheus 2>/dev/null || true
pkill -f auto_telemetry_service 2>/dev/null || true

sleep 2

echo "🎯 Starting Automatic Telemetry Service..."

# Make the Python script executable
chmod +x "$SCRIPT_DIR/../services/auto_service.py"

# Run the service from the script directory
cd "$SCRIPT_DIR"
if [[ "$1" == "test" ]]; then
    # Test mode - run for specified duration
    duration=${2:-120}
    echo "🧪 Running in test mode for ${duration} seconds..."
    python3 ../services/auto_service.py test $duration
else
    # Production mode - run continuously
    echo "🔄 Running in continuous mode..."
    echo "📊 Metrics will be generated every 5-15 seconds"
    echo "📤 Data pushed to Grafana Cloud every 5 seconds"
    echo "🛑 Press Ctrl+C to stop"
    echo ""
    python3 ../services/auto_service.py
fi