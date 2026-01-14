#!/bin/bash
# Run tests for Photo Triage Agent backend

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$(dirname "$SCRIPT_DIR")"

cd "$BACKEND_DIR"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed."
    exit 1
fi

echo "Running Photo Triage Agent tests..."
echo ""

# Run tests
uv run pytest tests/ -v "$@"
