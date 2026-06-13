# =============================================================================
# Dockerfile — e-Nose Industrial Gateway Service
# Multi-stage build: builder + slim production image
#
# Stage 1 (builder): installs all Python deps + compiles any C extensions
# Stage 2 (runtime): copies only the installed packages and app code
#
# Build:   docker build -t enose-gateway:latest .
# Run:     docker run --env-file .env -p 8000:8000 enose-gateway:latest
# =============================================================================

# ---- Stage 1: Builder -------------------------------------------------------
FROM python:3.11-slim AS builder

# System build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment to isolate dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install Python dependencies first (Docker layer cache)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r /tmp/requirements.txt

# ---- Stage 2: Runtime -------------------------------------------------------
FROM python:3.11-slim AS runtime

LABEL maintainer="e-Nose R&D Team"
LABEL description="Industrial e-Nose Gateway — MQTT subscriber + FastAPI REST + reconciliation engine"
LABEL version="1.0.0"

# Non-root user for security
RUN groupadd --gid 1001 enose \
 && useradd  --uid 1001 --gid enose --shell /bin/bash --create-home enose

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# System runtime dependencies only (no compilers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Working directory
WORKDIR /app

# Copy application code
COPY gateway/app.py         ./gateway/app.py
COPY gateway/dashboard.py   ./gateway/dashboard.py

# Ownership
RUN chown -R enose:enose /app

USER enose

# Health check: FastAPI /api/v1/health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Expose FastAPI port
EXPOSE 8000

# Uvicorn with 2 workers, auto-reload disabled in production
CMD ["uvicorn", "gateway.app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "info", \
     "--access-log"]
