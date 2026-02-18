FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (ffmpeg, tesseract, libsndfile for audio processing, gcc for webrtcvad)
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    ffmpeg \
    tesseract-ocr \
    tesseract-ocr-eng \
    libsndfile1-dev \
    gcc \
    libc6-dev \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first (avoids pulling ~2GB CUDA libs)
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy server code
COPY server.py .

# Create documents directory
RUN mkdir -p /documents

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    MCP_TRANSPORT=sse \
    MCP_HOST=0.0.0.0 \
    MCP_PORT=8000 \
    DOCUMENTS_PATH=/documents

# Expose port for SSE transport
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=5)" || exit 1

# Run the server
CMD ["python", "server.py"]
