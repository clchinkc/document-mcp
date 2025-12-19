FROM python:3.13-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml .

# Install Python dependencies (document-mcp core + server extras)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir ".[server]"

# Copy application code
COPY server.py .

# Create storage directory with proper permissions
RUN mkdir -p /data/documents_storage && \
    chmod 755 /data/documents_storage

# Environment variables
ENV DOCUMENTS_STORAGE_PATH=/data/documents_storage
ENV PORT=8080
ENV LOG_LEVEL=info

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Expose port
EXPOSE ${PORT}

# Run server (use shell form to allow PORT env var expansion)
CMD uvicorn server:app --host 0.0.0.0 --port ${PORT} --log-level info 