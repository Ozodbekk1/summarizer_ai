# Stage 1: Build
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Create wheels for all dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip wheel --no-cache-dir --wheel-dir /app/wheels -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies (libgomp is needed for llama-cpp-python)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from wheels
COPY --from=builder /app/wheels /wheels
RUN pip install --no-cache-dir /wheels/* && \
    rm -rf /wheels

# Copy application files
COPY api.py main.py summarize.py summarizer_ai_model.gguf ./

# Set default environment variables
ENV API_HOST=0.0.0.0
ENV API_PORT=8000
ENV MODEL_PATH=summarizer_ai_model.gguf
ENV API_SECRET_KEY=change_me_in_production

EXPOSE 8000

# Run the application
CMD ["python", "main.py"]
