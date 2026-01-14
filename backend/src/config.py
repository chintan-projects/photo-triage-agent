"""Configuration for the Photo Triage backend."""
import os
from pathlib import Path

# Base paths
BACKEND_DIR = Path(__file__).parent.parent
MODELS_DIR = BACKEND_DIR / "models"

# LFM Model paths (official HuggingFace GGUF)
LFM_MODEL_DIR = MODELS_DIR / "LFM2.5-VL-1.6B-GGUF"
LFM_MODEL_PATH = LFM_MODEL_DIR / "LFM2.5-VL-1.6B-Q8_0.gguf"
LFM_MMPROJ_PATH = LFM_MODEL_DIR / "mmproj-LFM2.5-VL-1.6b-Q8_0.gguf"

# Model settings
MODEL_N_CTX = int(os.getenv("MODEL_N_CTX", "2048"))
MODEL_N_GPU_LAYERS = int(os.getenv("MODEL_N_GPU_LAYERS", "-1"))
MODEL_VERBOSE = os.getenv("MODEL_VERBOSE", "false").lower() == "true"

# API settings
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))

# Database
DATABASE_PATH = BACKEND_DIR / "data" / "photo_triage.db"

# CLI executable
LFM_CLI_PATH = os.getenv("LFM_CLI_PATH", "llama-mtmd-cli")


def get_model_config() -> dict:
    """Get model configuration as a dictionary.

    Returns:
        Dict with model_path, mmproj_path, cli_path, n_ctx, n_gpu_layers.
    """
    return {
        "model_path": str(LFM_MODEL_PATH),
        "mmproj_path": str(LFM_MMPROJ_PATH),
        "cli_path": LFM_CLI_PATH,
        "n_ctx": MODEL_N_CTX,
        "n_gpu_layers": MODEL_N_GPU_LAYERS,
    }


def models_available() -> bool:
    """Check if model files exist."""
    return LFM_MODEL_PATH.exists() and LFM_MMPROJ_PATH.exists()
