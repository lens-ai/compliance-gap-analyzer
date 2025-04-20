FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs temp vector_store data

# Set environment variables
ENV PYTHONPATH=/app

# Expose ports
EXPOSE 5000 8050

# Run command (can be overridden in docker-compose.yml)
CMD ["python", "app/main.py", "--api"]
