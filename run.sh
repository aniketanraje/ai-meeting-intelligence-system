#!/bin/bash
# -----------------------------------------------
# AI Meeting Notes Analyzer — Local Run Script
# Usage: bash run.sh
# -----------------------------------------------

set -e  # Exit immediately on error

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_ROOT/venv"
ENV_FILE="$PROJECT_ROOT/.env"
REQUIREMENTS="$PROJECT_ROOT/requirements.txt"

echo "-----------------------------------------------"
echo " AI Meeting Notes Analyzer"
echo "-----------------------------------------------"

# Step 1: Verify .env exists
if [ ! -f "$ENV_FILE" ]; then
    echo "[WARN] .env file not found."
    echo "[INFO] Copying .env.example → .env"
    cp "$PROJECT_ROOT/.env.example" "$ENV_FILE"
    echo "[ACTION REQUIRED] Open .env and set your API keys before proceeding."
    exit 1
fi

# Step 2: Create virtual environment if missing
if [ ! -d "$VENV_DIR" ]; then
    echo "[INFO] Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    echo "[INFO] Virtual environment created at $VENV_DIR"
fi

# Step 3: Activate virtual environment
echo "[INFO] Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Step 4: Install/upgrade dependencies
echo "[INFO] Installing dependencies..."
pip install --upgrade pip --quiet
pip install -r "$REQUIREMENTS" --quiet
echo "[INFO] Dependencies installed."

# Step 5: Initialize required runtime directories
mkdir -p "$PROJECT_ROOT/outputs"
mkdir -p "$PROJECT_ROOT/db"
mkdir -p "$PROJECT_ROOT/data/sample_transcripts"
echo "[INFO] Runtime directories verified."

# Step 6: Launch Streamlit application
echo "[INFO] Launching application..."
echo "-----------------------------------------------"
echo " Open browser → http://localhost:8501"
echo "-----------------------------------------------"

streamlit run "$PROJECT_ROOT/app/streamlit_app.py"
