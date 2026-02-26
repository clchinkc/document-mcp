#!/bin/bash
#
# Flaky Test Detection Wrapper
# Convenient shell wrapper for running flaky test detection
#
# Usage:
#   ./scripts/development/detect_flaky_tests.sh [--runs N] [--test-path PATH] [--save-json FILE]
#

# Get the directory this script is in
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT" || exit 1

# Run the Python flaky test detector
python3 "$SCRIPT_DIR/flaky_test_detector.py" "$@"
