#!/bin/bash
# Run the Photo Triage Agent backend server

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$(dirname "$SCRIPT_DIR")"

cd "$BACKEND_DIR"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed."
    echo "Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Sync dependencies
echo "Syncing dependencies..."
uv sync

# Start server
echo "Starting Photo Triage Agent..."
echo "Open http://localhost:8000 in your browser"
echo ""

uv run uvicorn src.main:app --host 127.0.0.1 --port 8000 --reload
