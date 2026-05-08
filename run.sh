#!/bin/bash

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$PROJECT_ROOT/.env"

echo "=========================================="
echo " AI Meeting Intelligence System Launcher"
echo "=========================================="

echo "[INFO] Project root: $PROJECT_ROOT"

# -------------------------------------------------------------------
# Python version check
# -------------------------------------------------------------------

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')

if [[ "$PYTHON_VERSION" != "3.12" ]]; then
    echo "[WARN] Recommended Python version is 3.12"
    echo "[INFO] Detected Python version: $PYTHON_VERSION"
fi

# -------------------------------------------------------------------
# Environment file validation
# -------------------------------------------------------------------

if [[ ! -f "$ENV_FILE" ]]; then
    echo "[ERROR] Missing .env file."
    echo "[INFO] Create a .env file before running the application."
    exit 1
fi

echo "[INFO] .env file detected."

# -------------------------------------------------------------------
# ffmpeg validation
# -------------------------------------------------------------------

if ! command -v ffmpeg &> /dev/null; then
    echo "[WARN] ffmpeg is not installed."
    echo "[WARN] Audio transcription may fail."
else
    echo "[INFO] ffmpeg detected."
fi

# -------------------------------------------------------------------
# Create virtual environment if missing
# -------------------------------------------------------------------

if [[ ! -d "$PROJECT_ROOT/venv" ]]; then
    echo "[INFO] Creating virtual environment..."
    python3 -m venv "$PROJECT_ROOT/venv"
else
    echo "[INFO] Existing virtual environment detected."
fi

# -------------------------------------------------------------------
# Activate virtual environment
# -------------------------------------------------------------------

echo "[INFO] Activating virtual environment..."
source "$PROJECT_ROOT/venv/bin/activate"

# -------------------------------------------------------------------
# Upgrade pip
# -------------------------------------------------------------------

echo "[INFO] Upgrading pip..."
pip install --upgrade pip

# -------------------------------------------------------------------
# Install dependencies
# -------------------------------------------------------------------

echo "[INFO] Installing project dependencies..."
pip install -r "$PROJECT_ROOT/requirements.txt"

# -------------------------------------------------------------------
# Startup validation
# -------------------------------------------------------------------

echo "[INFO] Running startup validation..."

python -c "import streamlit, langgraph, groq" \
    && echo "[INFO] Core dependencies validated successfully." \
    || { echo '[ERROR] Dependency validation failed.'; exit 1; }

# -------------------------------------------------------------------
# Launch application
# -------------------------------------------------------------------

echo "[INFO] Launching Streamlit application..."

PYTHONPATH="$PROJECT_ROOT" streamlit run "$PROJECT_ROOT/app/streamlit_app.py"