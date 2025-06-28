# Deploying and Monitoring the Document MCP Server

## Minimal, Open-Source, Self-Hosted Metrics: Prometheus Only

For most open-source projects—especially those with a small or moderate user base and a strong need for privacy and data control—the simplest and most robust solution is to use **Prometheus** for metrics collection. This approach:

- **Keeps all metrics data on your own server** (no third-party cloud/SaaS)
- **Minimizes implementation complexity** (just two containers: your app + Prometheus)
- **Requires no account, API keys, or cloud billing**
- **Is fully open source** and easy to run locally or on any VPS
- **Lets you persist metrics data** as long as you want, with full control

### Why Prometheus?
- **No user data leaves your infrastructure**—all metrics are scraped and stored by your own Prometheus instance.
- **No vendor lock-in**—you can export or migrate your data at any time.
- **Simple setup**—just a few lines of Docker Compose.
- **Great for privacy**—ideal for open-source projects and privacy-conscious deployments.

---

## Quick Start: Local Prometheus Monitoring

### 1. Prerequisites
- Docker and Docker Compose installed
- Your `.env` file in the project root (see below for minimal config)

### 2. Minimal `.env` Example
```env
MCP_METRICS_ENABLED=true
OTEL_SERVICE_NAME=document-mcp-server
OTEL_SERVICE_VERSION=0.0.1
DEPLOYMENT_ENVIRONMENT=production
# No OTLP endpoint needed for Prometheus-only
```

### 3. prometheus.yml
```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'document-mcp-server'
    static_configs:
      - targets: ['doc-mcp-server:3001']
```

### 4. docker-compose.yml
```yaml
version: '3.8'

services:
  doc-mcp-server:
    build: .
    container_name: doc-mcp-server
    restart: unless-stopped
    ports:
      - "3001:3001"
    volumes:
      - ./documents_storage_prod:/app/.documents_storage
    env_file:
      - .env

  prometheus:
    image: prom/prometheus:v2.45.0
    container_name: prometheus
    restart: unless-stopped
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - ./prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'

volumes:
  documents_storage_prod:
  prometheus_data:
```

### 5. Start Everything
```bash
mkdir -p ./documents_storage_prod ./prometheus_data
# Build and start both services
docker-compose up -d
```

### 6. Access Metrics
- **Prometheus UI**: [http://localhost:9090](http://localhost:9090)
- **Application Health**: [http://localhost:3001/health](http://localhost:3001/health)
- **Metrics Endpoint**: [http://localhost:3001/metrics](http://localhost:3001/metrics)

### 7. Persisting Metrics Data
- The `./prometheus_data` volume ensures all metrics are stored on disk and survive container restarts.
- You can back up, move, or analyze this data as you wish.

### 8. Privacy & Data Control
- **No metrics or user data ever leaves your server.**
- You have full control over retention, backups, and access.

---

## Part 1: Understanding the Metrics System

Before deploying, it's important to understand the built-in monitoring capabilities. The Document MCP server includes built-in metrics collection that tracks:

- **Tool Usage**: Count of each MCP tool invocation
- **Performance**: Execution time for each tool
- **Errors**: Failed tool calls with error types
- **Concurrency**: Number of simultaneous operations
- **Data Volume**: Size of arguments and responses
- **Environment Info**: Server version, hostname, environment

### Metrics Architecture

```mermaid
graph LR
    A[MCP Server] --> B[OpenTelemetry SDK]
    B --> C[Local Prometheus Endpoint]
    B --> D[Remote OTLP Exporter]
    C --> E[Local Monitoring]
    D --> F[Grafana Cloud]
    D --> G[Datadog]
    D --> H[New Relic]
    D --> I[Custom Collector]
```

### Quick Start: Configuration

1.  **Install Dependencies**: The required OpenTelemetry dependencies are already included in `pyproject.toml`.

2.  **Environment Configuration**: Create a `.env` file in your project root with the following configuration:

    ```bash
    # Enable metrics collection
    MCP_METRICS_ENABLED=true

    # Service identification
    OTEL_SERVICE_NAME=document-mcp-server
    OTEL_SERVICE_VERSION=0.0.1
    DEPLOYMENT_ENVIRONMENT=production

    # Remote metrics endpoint (choose one of the options below)
    OTEL_EXPORTER_OTLP_ENDPOINT=https://your-metrics-service:4317
    OTEL_EXPORTER_OTLP_HEADERS=Authorization=Bearer your-auth-token
    ```

3.  **Verify Configuration**: Once the server is running, you can verify the metrics setup at these endpoints:
    -   `http://<server-host>:3001/metrics` (Prometheus format)
    -   `http://<server-host>:3001/metrics/summary` (JSON summary)


### Configuring Remote Monitoring Services

To export metrics to a managed service, configure the `OTEL_EXPORTER_OTLP_ENDPOINT` and `OTEL_EXPORTER_OTLP_HEADERS` in your `.env` file.

#### Grafana Cloud

```bash
# Get your Grafana Cloud OTLP endpoint and API key from:
# https://grafana.com/docs/grafana-cloud/send-data/otlp/

OTEL_EXPORTER_OTLP_ENDPOINT=https://otlp-gateway-prod-us-central-0.grafana.net/otlp
OTEL_EXPORTER_OTLP_HEADERS=Authorization=Basic YOUR_GRAFANA_CLOUD_API_KEY
```

#### Datadog

```bash
# Get your Datadog API key from:
# https://docs.datadoghq.com/getting_started/site/

OTEL_EXPORTER_OTLP_ENDPOINT=https://api.datadoghq.com:4317
OTEL_EXPORTER_OTLP_HEADERS=DD-API-KEY=your-datadog-api-key
```

#### New Relic

```bash
# Get your New Relic license key from:
# https://docs.newrelic.com/docs/apis/intro-apis/new-relic-api-keys/

OTEL_EXPORTER_OTLP_ENDPOINT=https://otlp.nr-data.net:4317
OTEL_EXPORTER_OTLP_HEADERS=api-key=your-new-relic-license-key
```

#### Self-Hosted OpenTelemetry Collector

```bash
# If you're running your own OTLP collector
OTEL_EXPORTER_OTLP_ENDPOINT=http://your-collector:4317
OTEL_EXPORTER_OTLP_HEADERS=Authorization=Bearer your-auth-token
```

### Available Metrics and Queries

#### Key Metrics

- **Tool Usage**:
  - `mcp_tool_calls_total`: Counter of tool invocations.
  - `mcp_tool_duration_seconds`: Histogram of tool execution times.
  - `mcp_tool_errors_total`: Counter of errors.
  - `mcp_tool_argument_sizes_bytes`: Histogram of argument payload sizes.
- **System Metrics**:
  - `mcp_concurrent_operations`: Gauge of simultaneous operations.
  - `mcp_server_info`: Counter with server metadata.

#### Key Labels for Filtering

- `tool_name`: The name of the MCP tool.
- `status`: "success" or "error".
- `error_type`: The type of exception for error metrics.
- `environment`: "development", "staging", or "production".

#### Example PromQL Queries

```promql
# Rate of calls for each tool over the last 5 minutes
sum by (tool_name) (rate(mcp_tool_calls_total[5m]))

# P95 response time by tool
histogram_quantile(0.95, sum(rate(mcp_tool_duration_seconds_bucket[5m])) by (le, tool_name))

# Error rate per tool
sum by (tool_name) (rate(mcp_tool_errors_total[5m])) / sum by (tool_name) (rate(mcp_tool_calls_total[5m]))
```

---

## Part 2: Deployment Examples

Now that you understand how monitoring is configured, here's how to deploy the application.

### Prerequisites

- **Docker**: Ensure Docker is installed on your deployment server.
- **`.env` file**: Have your configured `.env` file ready in the project root.
- **Dockerfile**: Ensure the `Dockerfile` from Example 1 below exists in your project root.

### Example 1: Deploying with Docker

This is the most straightforward method for deploying the server as a containerized application.

#### Step 1: Create a Dockerfile

```Dockerfile
# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy dependency definition files
COPY pyproject.toml poetry.lock* ./

# Install poetry
RUN pip install poetry

# Install project dependencies
RUN poetry install --no-root --no-interaction

# Copy the application source code into the container
COPY document_mcp/ ./document_mcp/

# Set default environment variables for production
ENV MCP_METRICS_ENABLED=true
ENV DEPLOYMENT_ENVIRONMENT=production
ENV OTEL_SERVICE_NAME=document-mcp-server
ENV PYTHONUNBUFFERED=1

# Expose the port the server will run on
EXPOSE 3001

# The command to run when the container starts
CMD ["poetry", "run", "python", "-m", "document_mcp.doc_tool_server", "sse", "--host", "0.0.0.0", "--port", "3001"]
```

#### Step 2: Build the Docker Image

```bash
docker build -t document-mcp-server .
```

#### Step 3: Run the Docker Container

```bash
# Create a directory on the host to store documents
mkdir -p ./documents_storage_prod

# Run the container
docker run -d \
  --name doc-mcp-server \
  -p 3001:3001 \
  -v $(pwd)/documents_storage_prod:/app/.documents_storage \
  --env-file .env \
  document-mcp-server
```
- **Critical**: The `-v` flag mounts a volume to persist your documents outside the container.
- The `--env-file` flag loads your metrics configuration.

---

### Example 2: Deploying with Docker Compose and Prometheus

For a more robust setup, `docker-compose` can manage both your application server and a Prometheus instance for local monitoring.

#### Step 1: Create `prometheus.yml`

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'document-mcp-server'
    static_configs:
      - targets: ['doc-mcp-server:3001']
```

#### Step 2: Create `docker-compose.yml`

```yaml
# docker-compose.yml
version: '3.8'

services:
  doc-mcp-server:
    build: .
    container_name: doc-mcp-server
    restart: unless-stopped
    ports:
      - "3001:3001"
    volumes:
      - ./documents_storage_prod:/app/.documents_storage
    env_file:
      - .env

  prometheus:
    image: prom/prometheus:v2.45.0
    container_name: prometheus
    restart: unless-stopped
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'

volumes:
  documents_storage_prod:

```

#### Step 3: Run with Docker Compose

```bash
mkdir -p ./documents_storage_prod
docker-compose up -d
```

#### Step 4: Verify the Services
- **Prometheus UI**: `http://localhost:9090`
- **Application Health**: `http://localhost:3001/health`

---

## Part 3: Troubleshooting and Advanced Topics

### Local Development

For local development and testing, you can disable remote export in your `.env` file:

```bash
# Disable remote export by leaving the endpoint blank
OTEL_EXPORTER_OTLP_ENDPOINT=""
# Set environment to development
DEPLOYMENT_ENVIRONMENT=development
```

### Security Considerations

1.  **API Keys**: Never commit API keys or sensitive credentials to version control. Use `.env` or another secrets management system.
2.  **Network**: Ensure OTLP endpoints use HTTPS (`https://...`) in production.
3.  **Data Privacy**: Metrics do not include document content, only metadata.

### Troubleshooting

- **Metrics Not Appearing**:
  1.  Run `pip install -e .` to ensure all dependencies are installed.
  2.  Verify your `.env` configuration at `http://localhost:3001/metrics/summary`.
  3.  Check server logs for OpenTelemetry initialization messages.
- **Remote Export Issues**:
  1.  Double-check your endpoint URL and authentication headers.
  2.  Verify network connectivity from your server to the monitoring service.
  3.  Confirm the export interval (default: 30 seconds).

### Advanced Configuration

You can customize metrics further in `document_mcp/metrics_config.py`:
- **Custom Attributes**: Add new attributes (e.g., customer ID, request ID) to the resource in `get_resource()`.
- **Export Intervals**: Modify the `export_interval_millis` in the `PeriodicExportingMetricReader` to change export frequency. 