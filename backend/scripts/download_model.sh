#!/bin/bash
# Download the LFM model for Photo Triage Agent

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$BACKEND_DIR/models"
MODEL_DIR="$MODELS_DIR/LFM2.5-VL-1.6B-GGUF"

# Model files
MODEL_URL="https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B-GGUF/resolve/main/LFM2.5-VL-1.6B-Q8_0.gguf"
MMPROJ_URL="https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B-GGUF/resolve/main/mmproj-LFM2.5-VL-1.6b-Q8_0.gguf"

echo "Photo Triage Agent - Model Download"
echo "===================================="
echo ""

# Create directories
mkdir -p "$MODEL_DIR"

# Check if model already exists
if [ -f "$MODEL_DIR/LFM2.5-VL-1.6B-Q8_0.gguf" ] && [ -f "$MODEL_DIR/mmproj-LFM2.5-VL-1.6b-Q8_0.gguf" ]; then
    echo "Model files already exist. Skipping download."
    echo "Location: $MODEL_DIR"
    exit 0
fi

# Download model
echo "Downloading LFM model (~1.7GB)..."
echo "This may take a few minutes depending on your internet connection."
echo ""

if command -v curl &> /dev/null; then
    curl -L -o "$MODEL_DIR/LFM2.5-VL-1.6B-Q8_0.gguf" "$MODEL_URL" --progress-bar
    curl -L -o "$MODEL_DIR/mmproj-LFM2.5-VL-1.6b-Q8_0.gguf" "$MMPROJ_URL" --progress-bar
elif command -v wget &> /dev/null; then
    wget -O "$MODEL_DIR/LFM2.5-VL-1.6B-Q8_0.gguf" "$MODEL_URL"
    wget -O "$MODEL_DIR/mmproj-LFM2.5-VL-1.6b-Q8_0.gguf" "$MMPROJ_URL"
else
    echo "Error: Neither curl nor wget is installed."
    echo "Please install curl or wget and try again."
    exit 1
fi

echo ""
echo "Download complete!"
echo "Model location: $MODEL_DIR"
echo ""
echo "Next steps:"
echo "1. Make sure llama.cpp is installed with llama-mtmd-cli"
echo "2. Run: ./scripts/run_backend.sh"
