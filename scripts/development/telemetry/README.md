# 🚀 Document MCP - Automatic Telemetry System

## Quick Start - Push Metrics to Grafana Cloud

### 🎯 One-Command Start
```bash
# From project root:
# Start automatic telemetry (runs continuously)
scripts/development/telemetry/scripts/start.sh

# Test mode (runs for 2 minutes)
scripts/development/telemetry/scripts/start.sh test 120
```

### ✅ What Happens Automatically
1. **Metrics Server** starts on `localhost:8000`
2. **Prometheus** starts and scrapes every 5 seconds
3. **Continuous Metrics** generated every 5-15 seconds
4. **Data Pushed** to Grafana Cloud every 5 seconds
5. **Realistic MCP Tool Usage** simulated automatically

### 📊 Generated Metrics
- **Tool Calls**: `mcp_tool_calls_total` (success/error by tool)
- **Duration**: `mcp_tool_duration_seconds` (execution times)
- **Concurrency**: `mcp_concurrent_operations` (active operations)
- **Server Info**: `mcp_server_startup_total` (startup events)

### 🔍 View in Grafana Cloud
**URL**: `https://stack-1326187-integration-document-mcp.grafana.net/`

**Queries**:
```promql
{job="document-mcp"}                           # All metrics
mcp_tool_calls_total                          # Tool usage (integers)
rate(mcp_tool_calls_total[5m])                # Usage rate (per second)
sum by (tool_name) (mcp_tool_calls_total)     # Usage by tool
increase(mcp_tool_calls_total[1m])            # Recent increase
```

### 🛠️ MCP Tools Simulated
- `create_document`, `list_documents`, `get_document`
- `add_chapter`, `edit_chapter`, `delete_chapter`  
- `add_paragraph`, `edit_paragraph`, `delete_paragraph`
- `search_content`, `replace_content`, `get_stats`
- `create_snapshot`, `restore_snapshot`, `batch_operations`

### ⚙️ Configuration
- **Scrape Interval**: 5 seconds (in `prometheus.yml`)
- **Metric Generation**: 5-15 second intervals
- **Success Rate**: 90% success, 10% errors
- **Ports**: 8000 (metrics), 9090 (Prometheus UI)

### 🔧 Manual Control
```bash
# Stop all services
pkill prometheus
pkill -f auto_telemetry_service

# Check status
curl http://localhost:8000/metrics | grep mcp_
curl http://localhost:9090/api/v1/targets

# Restart just Prometheus
prometheus --config.file=scripts/development/telemetry/config/prometheus.yml
```

### 📈 Performance
- **Resource Usage**: Minimal (<50MB RAM, <5% CPU)
- **Network**: ~1KB every 5 seconds to Grafana Cloud
- **Metrics Volume**: 15+ metric series continuously
- **Data Points**: 100+ per minute

## 🎉 Success Verification

**You know it's working when:**
1. ✅ Script shows "All services started successfully!"
2. ✅ Metrics endpoint responds: `curl localhost:8000/metrics`
3. ✅ Prometheus targets show UP: `localhost:9090/targets`
4. ✅ Grafana Cloud shows `job="document-mcp"` data

**Troubleshooting:**
- If port 8000 busy: System will try 8001, 8002 automatically
- If Prometheus fails: Check `prometheus.yml` exists
- If no data in Grafana: Wait 2-3 minutes for first data
- Network issues: Connection resets are normal, Prometheus retries automatically
- If mcp_tool_calls_total shows 0: Restart telemetry service - metrics now include initial burst for immediate visibility

**Fixed Issues:**
- ✅ **mcp_tool_calls_total Metric**: Fixed issue where this metric was stuck at 0
- ✅ **Initial Metrics Burst**: Added immediate metric generation on startup for visibility
- ✅ **Debug Endpoint**: Added `/debug` endpoint for troubleshooting metrics
- ✅ **Enhanced Error Handling**: Better error reporting in metric generation

**Quick Tests:**
```bash
# From project root:
# Test development telemetry infrastructure
python3 scripts/development/telemetry/scripts/test.py

# Test production metrics system  
python3 scripts/development/metrics/test_production.py

# Should show: "SUCCESS: mcp_tool_calls_total metrics are being generated!"
```

---
**🌐 Your metrics are now automatically flowing to Grafana Cloud!**