# Use a stable Python runtime for the application
FROM python:3.12-slim

# Application working directory
WORKDIR /app

# Prevent Python from creating .pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# Send Python logs directly to the container output
ENV PYTHONUNBUFFERED=1

# Install system dependencies needed by some Python packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency file first for Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy backend
COPY app ./app

# Copy frontend
COPY UI ./UI

# FastAPI listens on port 8000
EXPOSE 8000

# Start the application
CMD ["python", "-m", "uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]