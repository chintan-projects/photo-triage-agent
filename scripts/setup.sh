#!/bin/bash
set -e

echo "🚀 Setting up Photo Triage Agent..."

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check prerequisites
echo -e "${YELLOW}Checking prerequisites...${NC}"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python 3 is required but not installed.${NC}"
    echo "Install with: brew install python@3.11"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "  ✓ Python $PYTHON_VERSION"

# Check uv
if ! command -v uv &> /dev/null; then
    echo -e "${YELLOW}Installing uv package manager...${NC}"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
fi
echo "  ✓ uv installed"

# Check Xcode command line tools
if ! xcode-select -p &> /dev/null; then
    echo -e "${YELLOW}Installing Xcode Command Line Tools...${NC}"
    xcode-select --install
    echo "Please complete the Xcode installation and run this script again."
    exit 1
fi
echo "  ✓ Xcode CLI tools"

# Create project structure
echo -e "${YELLOW}Creating project structure...${NC}"

mkdir -p backend/src/{api,analyzers,classifiers,utils,database}
mkdir -p backend/tests/{analyzers,classifiers,api}
mkdir -p backend/models
mkdir -p scripts
mkdir -p docs

# Create essential files
touch backend/src/__init__.py
touch backend/src/api/__init__.py
touch backend/src/analyzers/__init__.py
touch backend/src/classifiers/__init__.py
touch backend/src/utils/__init__.py
touch backend/src/database/__init__.py
touch backend/tests/__init__.py

echo "  ✓ Project structure created"

# Setup Python backend
echo -e "${YELLOW}Setting up Python backend...${NC}"

cd backend

# Create pyproject.toml if not exists
if [ ! -f "pyproject.toml" ]; then
cat > pyproject.toml << 'EOF'
[project]
name = "photo-triage-backend"
version = "0.1.0"
description = "Local-first photo analysis backend"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.109.0",
    "uvicorn[standard]>=0.27.0",
    "pydantic>=2.5.0",
    "structlog>=24.1.0",
    "pillow>=10.2.0",
    "imagehash>=4.3.1",
    "opencv-python-headless>=4.9.0",
    "llama-cpp-python>=0.2.50",
    "send2trash>=1.8.0",
    "aiosqlite>=0.19.0",
    "python-multipart>=0.0.6",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "pytest-cov>=4.1.0",
    "httpx>=0.26.0",
    "ruff>=0.2.0",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP"]
EOF
fi

# Install dependencies
echo "  Installing Python dependencies..."
uv sync
uv sync --extra dev

cd ..
echo "  ✓ Backend setup complete"

# Create .gitignore
echo -e "${YELLOW}Creating .gitignore...${NC}"

cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
.venv/
venv/
.env
*.egg-info/
dist/
build/

# Models (large files)
backend/models/*.gguf
backend/models/*.bin

# IDE
.idea/
.vscode/
*.swp
*.swo
.DS_Store

# Testing
.coverage
htmlcov/
.pytest_cache/

# Database
*.db
*.sqlite


# Logs
*.log
logs/
EOF

echo "  ✓ .gitignore created"

# Download model script
echo -e "${YELLOW}Creating model download script...${NC}"

cat > scripts/download_model.sh << 'EOF'
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
EOF

chmod +x scripts/download_model.sh

echo "  ✓ Model download script created"

# Create run scripts
echo -e "${YELLOW}Creating run scripts...${NC}"

cat > scripts/run_backend.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")/../backend"
uv run uvicorn src.main:app --reload --host 127.0.0.1 --port 8000
EOF

chmod +x scripts/run_backend.sh

cat > scripts/test_backend.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")/../backend"
uv run pytest "${@:---v}"
EOF

chmod +x scripts/test_backend.sh

echo "  ✓ Run scripts created"

# Create initial backend files
echo -e "${YELLOW}Creating initial backend files...${NC}"

# Main FastAPI app
cat > backend/src/main.py << 'EOF'
"""Photo Triage Agent - Backend API"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import structlog

from .api.routes import router

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
)

logger = structlog.get_logger()

app = FastAPI(
    title="Photo Triage Agent",
    description="Local-first photo analysis API",
    version="0.1.0"
)

# CORS for local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

@app.on_event("startup")
async def startup():
    logger.info("server_starting", version="0.1.0")

@app.on_event("shutdown")
async def shutdown():
    logger.info("server_stopping")
EOF

# API routes
cat > backend/src/api/routes.py << 'EOF'
"""API route definitions"""
from fastapi import APIRouter, UploadFile, File, HTTPException
from .schemas import HealthResponse, AnalyzeResponse
import structlog

logger = structlog.get_logger()
router = APIRouter()

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the API is running and model is loaded."""
    return HealthResponse(
        status="healthy",
        model_loaded=False,  # TODO: Check actual model status
        version="0.1.0"
    )

@router.post("/analyze/single", response_model=AnalyzeResponse)
async def analyze_single(image: UploadFile = File(...)):
    """Analyze a single image."""
    logger.info("analyze_single_request", filename=image.filename)
    
    # TODO: Implement actual analysis
    return AnalyzeResponse(
        success=True,
        data={
            "filename": image.filename,
            "category": "unknown",
            "confidence": 0.0,
        }
    )

@router.post("/analyze/folder")
async def analyze_folder(folder_path: str):
    """Start analysis job for a folder."""
    logger.info("analyze_folder_request", folder=folder_path)
    
    # TODO: Implement folder analysis
    return {
        "success": True,
        "job_id": "placeholder-job-id",
        "message": f"Analysis started for {folder_path}"
    }
EOF

# API schemas
cat > backend/src/api/schemas.py << 'EOF'
"""Pydantic schemas for API requests and responses"""
from pydantic import BaseModel
from typing import Any

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str

class AnalyzeResponse(BaseModel):
    success: bool
    data: dict[str, Any] | None = None
    error: str | None = None
EOF

# Base analyzer
cat > backend/src/analyzers/base.py << 'EOF'
"""Base classes for analyzers"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar('T')

@dataclass
class Result(Generic[T]):
    """Result type for operations that can fail."""
    success: bool
    value: T | None = None
    error: str | None = None
    
    @classmethod
    def ok(cls, value: T) -> "Result[T]":
        return cls(success=True, value=value)
    
    @classmethod
    def fail(cls, error: str) -> "Result[T]":
        return cls(success=False, error=error)

@dataclass
class AnalysisResult:
    """Base result from any analyzer."""
    analyzer_name: str
    confidence: float
    metadata: dict

class BaseAnalyzer(ABC):
    """Abstract base class for all analyzers."""
    
    name: str = "base"
    version: str = "1.0.0"
    
    @abstractmethod
    def analyze(self, image_path: str) -> Result[AnalysisResult]:
        """Analyze a single image."""
        pass
    
    def analyze_batch(self, paths: list[str]) -> list[Result[AnalysisResult]]:
        """Analyze multiple images. Override for efficiency."""
        return [self.analyze(p) for p in paths]
EOF

# Base model provider
cat > backend/src/classifiers/base.py << 'EOF'
"""Base classes for model providers"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

class ImageCategory(Enum):
    PEOPLE = "people"
    LANDSCAPE = "landscape"
    FOOD = "food"
    DOCUMENT = "document"
    SCREENSHOT = "screenshot"
    MEME = "meme"
    OBJECT = "object"
    ANIMAL = "animal"
    OTHER = "other"

@dataclass
class ClassificationResult:
    """Result from image classification."""
    category: ImageCategory
    confidence: float
    contains_faces: bool
    is_screenshot: bool
    is_meme: bool
    description: str

class ModelProvider(ABC):
    """Abstract interface for vision model providers.
    
    Implement this to add new models. The orchestrator doesn't
    care which model is used—just that it returns ClassificationResult.
    """
    
    name: str = "base"
    
    @abstractmethod
    def classify(self, image_path: str) -> ClassificationResult:
        """Classify a single image."""
        pass
    
    @abstractmethod
    def classify_batch(self, image_paths: list[str]) -> list[ClassificationResult]:
        """Classify multiple images."""
        pass
    
    @abstractmethod
    def health_check(self) -> bool:
        """Verify model is loaded and ready."""
        pass
EOF

# Image utilities
cat > backend/src/utils/image.py << 'EOF'
"""Image loading and processing utilities"""
from PIL import Image
from pathlib import Path
import structlog

logger = structlog.get_logger()

SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".heic", ".webp"}

def load_image(path: str, mode: str = "RGB") -> Image.Image:
    """Load and normalize an image.
    
    Args:
        path: Path to the image file
        mode: Color mode to convert to (default RGB)
        
    Returns:
        PIL Image object
        
    Raises:
        FileNotFoundError: If image doesn't exist
        ValueError: If format not supported
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    
    if path.suffix.lower() not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format: {path.suffix}")
    
    img = Image.open(path)
    
    if img.mode != mode:
        img = img.convert(mode)
    
    return img

def get_image_dimensions(path: str) -> tuple[int, int]:
    """Get image dimensions without loading full image."""
    with Image.open(path) as img:
        return img.size

def is_supported_format(path: str) -> bool:
    """Check if file format is supported."""
    return Path(path).suffix.lower() in SUPPORTED_FORMATS
EOF

echo "  ✓ Initial backend files created"

# Create initial test
cat > backend/tests/conftest.py << 'EOF'
"""Pytest configuration and fixtures"""
import pytest
from pathlib import Path

@pytest.fixture
def sample_images_dir():
    """Directory containing test images."""
    return Path(__file__).parent / "fixtures" / "images"

@pytest.fixture
def temp_output_dir(tmp_path):
    """Temporary directory for test outputs."""
    return tmp_path / "output"
EOF

cat > backend/tests/test_health.py << 'EOF'
"""Basic API health tests"""
import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
EOF

echo "  ✓ Initial tests created"

# Final message
echo ""
echo -e "${GREEN}✅ Setup complete!${NC}"
echo ""
echo "Next steps:"
echo "  1. Download the model:"
echo "     ./scripts/download_model.sh"
echo ""
echo "  2. Start the backend:"
echo "     ./scripts/run_backend.sh"
echo ""
echo "  3. Run tests:"
echo "     ./scripts/test_backend.sh"
echo ""
echo "Happy coding!"
