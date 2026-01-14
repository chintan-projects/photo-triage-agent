# Photo Triage Agent

Organize your photo library with AI. Locally. Privately.

## Features

- **AI Classification** - Categorize photos using local LFM model (people, landscape, food, etc.)
- **Duplicate Detection** - Find exact and near-duplicate photos using perceptual hashing
- **Quality Analysis** - Detect blurry photos you might want to delete
- **Screenshot Detection** - Identify screenshots and memes for easy cleanup
- **Natural Language Search** - Ask questions like "Find photos from my trip to Japan"
- **Safe Cleanup** - Move to trash with full undo support

All processing happens locally - your photos never leave your machine.

## Quick Start

### Prerequisites

- macOS 14+ (Apple Silicon recommended)
- Python 3.11+
- 4GB RAM minimum (8GB recommended)
- llama.cpp with llama-mtmd-cli (for AI classification)

### Installation

```bash
# Clone the repository
git clone https://github.com/user/photo-triage-agent
cd photo-triage-agent/backend

# Install dependencies
uv sync

# Download the LFM model (one-time)
./scripts/download_model.sh
```

### Running the Server

```bash
# Start the backend server
./scripts/run_backend.sh

# Or manually:
uv run uvicorn src.main:app --host 127.0.0.1 --port 8000
```

Open http://localhost:8000 in your browser.

## Usage

### Web Dashboard

1. Open http://localhost:8000
2. Enter the path to your photo folder
3. Click "Start Analysis"
4. Watch real-time progress with metrics
5. Browse results, review duplicates, and clean up

### iCloud Photos Location

For iCloud Photos on macOS:
```
~/Pictures/Photos Library.photoslibrary/originals/
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/analyze/folder` | POST | Start folder analysis |
| `/analyze/status/{job_id}` | GET | Check job status |
| `/analyze/results/{job_id}` | GET | Get analysis results |
| `/photos` | GET | List photos with filters |
| `/duplicates` | GET | Get duplicate groups |
| `/chat` | POST | Natural language search |
| `/actions/trash` | POST | Move photos to trash |
| `/actions/undo` | POST | Undo an action |

### Example: Start Analysis via API

```bash
curl -X POST http://localhost:8000/analyze/folder \
  -H "Content-Type: application/json" \
  -d '{"folder_path": "~/Pictures", "skip_lfm": false}'
```

### Example: Natural Language Search

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Find all sunset photos"}'
```

## Architecture

For detailed architecture documentation, see **[docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md)**.

Covers:
- System architecture with diagrams
- Database schema (ER diagram)
- API flow sequences
- Key design decisions
- How to extend the system

## Development

### Running Tests

```bash
uv run pytest tests/ -v
```

### Code Style

```bash
uv run ruff check src/
uv run ruff format src/
```

### Project Structure

```
backend/
├── src/
│   ├── analyzers/       # Blur, screenshot, hash detection
│   ├── classifiers/     # LFM model providers
│   ├── database/        # SQLite schema and repositories
│   ├── services/        # Job, metrics, search services
│   ├── api/             # FastAPI routes and schemas
│   ├── web/             # Dashboard templates
│   ├── orchestrator.py  # Folder analysis coordinator
│   └── main.py          # Application entry point
├── tests/               # Test suite
├── models/              # LFM model files (after download)
└── data/                # SQLite database
```

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `API_HOST` | 127.0.0.1 | Server host |
| `API_PORT` | 8000 | Server port |
| `MODEL_N_CTX` | 2048 | Model context size |
| `MODEL_N_GPU_LAYERS` | -1 | GPU layers (-1 = all) |

## Performance

Typical analysis speed on Apple Silicon:
- **With LFM**: ~0.3-0.5 photos/second (2-3s per photo)
- **Quick mode**: ~10-20 photos/second (blur, hash, screenshot only)

Memory usage: ~3-5GB during LFM inference

## Privacy

- All processing is done locally on your machine
- No cloud APIs are used by default
- Photos are never uploaded anywhere
- Database is stored locally in `data/photo_triage.db`

## Architecture Notes

### Current Implementation

The system uses **LFM2.5-VL** (Large Foundation Model with Vision-Language) for both:
1. **Image Classification** - Analyzing photos to determine category, detect faces, etc.
2. **Search Reasoning** - Understanding natural language queries and matching photos

This simplifies deployment (one model) but means search queries (~2-3s) are slower than optimal.

### Search Architecture

```
User Query: "Find photos from my trip to Japan"
                    │
                    ▼
┌─────────────────────────────────────────────┐
│ 1. Keyword Pre-Filter (fast, <50ms)         │
│    - Index lookup: "japan", "trip"          │
│    - Returns ~100 candidate photo IDs       │
└─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│ 2. LLM Reasoning (with timeout, ~2-3s)      │
│    - Send candidates + query to LFM         │
│    - LFM scores each photo by relevance     │
│    - Returns top 20 matches with reasons    │
└─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│ 3. Fallback (if LLM fails/times out)        │
│    - Pure keyword search on candidates      │
│    - Lower confidence score (0.3)           │
└─────────────────────────────────────────────┘
```

### Production Recommendations

For production deployments with large photo libraries (10K+ photos), consider:

| Task | Current | Recommended | Latency |
|------|---------|-------------|---------|
| Image Classification | LFM2.5-VL | LFM2.5-VL | ~2-3s/photo |
| Search Reasoning | LFM2.5-VL | LFM-1B (text-only) | ~0.3s/query |

**Why separate models?**

- Search doesn't need vision - it works on pre-computed descriptions
- Text-only models are 5-10x faster for reasoning tasks
- The ~0.3s latency enables real-time conversational search

**Migration path:**

1. Add a second model provider for text-only inference
2. Use LFM-1B (or similar) for search reasoning
3. Keep LFM2.5-VL for image classification only

### Caching Strategy

The search agent implements aggressive caching:

1. **Photo Context Cache** - Descriptions and keyword index (5 min TTL)
2. **Library Hash** - Invalidates cache when photos added/removed
3. **Keyword Index** - O(1) lookup for pre-filtering

This reduces repeated context building from ~500ms to <1ms.

### Future Enhancements

**Native Mac App (Post-V1)**

For a more integrated macOS experience, a native SwiftUI app could be built that:
- Integrates with Photos.app directly
- Provides menu bar quick access
- Uses native file dialogs and drag-drop
- Leverages macOS system features (Spotlight integration, Quick Look)

The current web dashboard approach was chosen for V1 to:
- Minimize dependencies (no Xcode required)
- Enable cross-platform potential
- Simplify deployment and testing

A SwiftUI implementation would follow the same API contract, making migration straightforward.

## License

MIT License - See LICENSE file for details.
