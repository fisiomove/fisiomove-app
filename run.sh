#!/usr/bin/env bash
set -e

VENV_DIR=".venv"

# Create virtual environment if missing
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Upgrade pip (safe + recommended)
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Activate virtual environment
source "$VENV_DIR/bin/activate"

streamlit run main.py