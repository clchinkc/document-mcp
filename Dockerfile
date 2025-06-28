# Python base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install poetry
RUN pip install poetry

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# Install dependencies
# Using --no-root to install only dependencies, not the project itself
RUN poetry install --no-root --no-interaction --no-ansi

# Copy the application code
COPY document_mcp/ ./document_mcp/

# Set environment variables for production
ENV MCP_METRICS_ENABLED=true
ENV DEPLOYMENT_ENVIRONMENT=production
ENV OTEL_SERVICE_NAME=document-mcp-server

# Expose the port the server runs on
EXPOSE 3001

# Command to run the application
CMD ["poetry", "run", "python", "-m", "document_mcp.doc_tool_server", "sse", "--host", "0.0.0.0", "--port", "3001"] 