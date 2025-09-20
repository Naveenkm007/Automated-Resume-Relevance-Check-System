# Multi-stage Dockerfile for Resume Relevance Check API

# Stage 1: Base Python environment
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set work directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Stage 2: Production image
FROM base as production

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p uploads/jds uploads/resumes tmp logs && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Stage 3: Development image
FROM base as development

# Install development dependencies
RUN pip install pytest pytest-cov black isort mypy

# Copy application code
COPY . .

# Create directories with proper permissions
RUN mkdir -p uploads/jds uploads/resumes tmp logs && \
    chmod 755 uploads uploads/jds uploads/resumes tmp logs

# Development command
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
