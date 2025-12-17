# syntax=docker/dockerfile:1

FROM python:3.11-slim

# Prevent Python from writing .pyc, and ensure logs flush
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_PORT=8080 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# System deps (kept minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Install Python deps first for better Docker layer caching
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# Copy the app
COPY . /app

# Fly expects the app to listen on 8080 by default
EXPOSE 8080

# Streamlit entrypoint
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
