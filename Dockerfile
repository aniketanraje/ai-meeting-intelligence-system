# -----------------------------------------------
# AI Meeting Notes Analyzer — Dockerfile
# Lightweight prototype containerization.
# NOT for production/enterprise deployment.
# -----------------------------------------------

FROM python:3.11-slim

# System dependencies for Whisper (ffmpeg) and audio processing
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and install Python dependencies first (layer cache optimization)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY . .

# Create required runtime directories
RUN mkdir -p outputs db data/sample_transcripts

# Expose Streamlit default port
EXPOSE 8501

# Streamlit config: disable telemetry, set server options
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Launch application
CMD ["streamlit", "run", "app/streamlit_app.py"]
