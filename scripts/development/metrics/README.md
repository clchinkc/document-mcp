# 📊 Production Metrics Testing

## Overview
This directory contains tools for testing the **production metrics system** built into Document MCP.

## Quick Test
```bash
# From project root:
python3 scripts/development/metrics/test_production.py
```

## What It Does
- ✅ **Tests production metrics**: Validates `document_mcp/metrics_config.py`
- ✅ **Tests Grafana connectivity**: Verifies connection to Grafana Cloud
- ✅ **Tests local endpoint**: Validates `localhost:8000/metrics`
- ✅ **Records test metrics**: Generates sample MCP tool calls

## Expected Output
```
🧪 Testing Document MCP metrics collection...
📊 Initializing metrics...
✅ Automatic telemetry: automatic_grafana_cloud
📈 Recording test metrics...
🔍 Testing local metrics endpoint...
✅ Local metrics endpoint working: 1234 bytes
🌐 Testing Grafana Cloud connectivity...
✅ Grafana Cloud endpoint reachable: 200
✅ Metrics test complete!
```

## Troubleshooting
- **No metrics endpoint**: Check if metrics server started successfully
- **Grafana Cloud errors**: Check network connectivity and credentials
- **Import errors**: Run from project root directory

## Related Testing
- **Development telemetry**: See `../telemetry/README.md` for testing the development infrastructure
- **E2E testing**: See `tests/e2e/` for full system integration tests