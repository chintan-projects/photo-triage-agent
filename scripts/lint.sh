#!/bin/bash
set -e

cd "$(dirname "$0")/../backend"

echo "🔍 Linting backend code..."
echo ""

# Run ruff for linting and formatting
uv run ruff check src/ tests/ --fix
uv run ruff format src/ tests/

echo ""
echo "✓ Linting complete"
