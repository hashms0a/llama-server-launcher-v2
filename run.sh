#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"

# Check if virtual environment exists; create one if not
if [ ! -d "$VENV_DIR" ]; then
    echo "Virtual environment not found. Creating one..."
    python3 -m venv "$VENV_DIR"
fi

# Activate the virtual environment
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# The pip bundled into the venv can ship a broken version comparator that
# wrongly treats Python 3.10 as older than 3.5, causing it to hide every
# package released after mid-2019. --ignore-requires-python lets us bootstrap
# a healthy pip past that faulty check; once upgraded, the problem is gone.
python3 -m pip install --no-cache-dir --ignore-requires-python --upgrade pip setuptools wheel
echo "Using $(pip --version)"

# Install dependencies from requirements.txt
if [ -f "$SCRIPT_DIR/requirements.txt" ]; then
    pip install --no-cache-dir -r "$SCRIPT_DIR/requirements.txt"
fi

# Run the application
echo "Starting Llama Server Launcher..."
python3 "$SCRIPT_DIR/llama_server_launcher_v14.py"
