FROM python:3.13-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# - document-mcp from PyPI (the core MCP package)
# - Server dependencies for OAuth 2.1
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    "document-mcp>=0.0.4" \
    "fastapi>=0.115.0" \
    "uvicorn[standard]>=0.32.0" \
    "google-auth>=2.34.0" \
    "google-cloud-firestore>=2.19.0" \
    "redis>=5.0.0" \
    "requests>=2.31.0"

# Copy application code
COPY server.py .

# Create storage directory with proper permissions
RUN mkdir -p /data/documents_storage && \
    chmod 755 /data/documents_storage

# Environment variables
ENV DOCUMENTS_STORAGE_PATH=/data/documents_storage
ENV PORT=8080
ENV LOG_LEVEL=info

# Health check - increased start-period for cold start reliability
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Expose port
EXPOSE ${PORT}

# Run server (use shell form to allow PORT env var expansion)
CMD uvicorn server:app --host 0.0.0.0 --port ${PORT} --log-level info 