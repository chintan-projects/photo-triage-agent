#!/bin/bash
set -e

echo "📦 Downloading LFM2.5-VL-1.6B model..."

MODEL_DIR="backend/models"
mkdir -p $MODEL_DIR

# Check if huggingface-cli is installed
if ! command -v huggingface-cli &> /dev/null; then
    echo "Installing huggingface-cli..."
    pip install huggingface_hub
fi

# Download the Q4_0 quantized model (smaller, ~696MB)
huggingface-cli download LiquidAI/LFM2.5-VL-1.6B-GGUF \
    --local-dir $MODEL_DIR \
    --include "*Q4_0*"

echo "✓ Model downloaded to $MODEL_DIR"
echo ""
echo "Available models:"
ls -lh $MODEL_DIR/*.gguf 2>/dev/null || echo "No .gguf files found"
