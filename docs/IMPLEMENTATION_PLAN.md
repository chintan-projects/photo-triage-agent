# Photo Triage Agent - Implementation Status

> Last updated: January 2025

## Project Status: V1 Complete

The core Photo Triage Agent is fully functional with:
- AI-powered photo classification using LFM2.5-VL
- Duplicate detection via perceptual hashing
- Blur and screenshot detection
- Natural language search with conversational refinement
- Web dashboard for review and actions
- Safe file operations with undo support

---

## What's Implemented

### Core Infrastructure

| Component | Status | Files |
|-----------|--------|-------|
| FastAPI Server | Done | `src/main.py` |
| Configuration | Done | `src/config.py` |
| Structured Logging | Done | Uses `structlog` |
| Database (SQLite) | Done | `src/database/schema.py`, `repository.py` |

### AI Classification

| Component | Status | Files |
|-----------|--------|-------|
| LFM CLI Provider | Done | `src/classifiers/lfm_cli_provider.py` |
| LFM Python Provider | Done | `src/classifiers/lfm_provider.py` |
| Mock Provider (testing) | Done | `src/classifiers/mock_provider.py` |
| Model Service | Done | `src/services/model_service.py` |

**Model**: LFM2.5-VL-1.6B (Q8_0 quantization) via `llama-mtmd-cli`

### Image Analyzers

| Analyzer | Status | Method | Files |
|----------|--------|--------|-------|
| Blur Detection | Done | Laplacian variance | `src/analyzers/blur.py` |
| Screenshot Detection | Done | EXIF + heuristics | `src/analyzers/screenshot.py` |
| Perceptual Hasher | Done | pHash (imagehash) | `src/analyzers/hasher.py` |
| Analyzer Registry | Done | Dynamic registration | `src/analyzers/__init__.py` |

### Services

| Service | Status | Purpose | Files |
|---------|--------|---------|-------|
| Job Service | Done | Async job management | `src/services/job_service.py` |
| Search Agent | Done | NL search with LFM | `src/services/search_agent.py` |
| Conversation Service | Done | Chat state persistence | `src/services/conversation.py` |
| Metrics Service | Done | Performance tracking | `src/services/metrics.py` |
| Upload Service | Done | Temporary uploads | `src/services/upload_service.py` |

### Orchestration

| Component | Status | Files |
|-----------|--------|-------|
| Folder Orchestrator | Done | `src/orchestrator.py` |
| Batch Processing | Done | Progress callbacks, resumable |
| Duplicate Grouping | Done | Creates groups from pHash |

### API Endpoints

| Endpoint | Method | Status |
|----------|--------|--------|
| `/health` | GET | Done |
| `/analyze/single` | POST | Done |
| `/analyze/folder` | POST | Done |
| `/analyze/status/{job_id}` | GET | Done |
| `/analyze/results/{job_id}` | GET | Done |
| `/photos` | GET | Done |
| `/duplicates` | GET | Done |
| `/chat` | POST | Done |
| `/chat/refine` | POST | Done |
| `/chat/explain` | POST | Done |
| `/actions/trash` | POST | Done |
| `/actions/undo` | POST | Done |
| `/upload/session` | POST | Done |
| `/upload/photos/{session_id}` | POST | Done |
| `/thumb/{photo_id}` | GET | Done |

### Web Dashboard

| Page | Status | Template |
|------|--------|----------|
| Home (Start Analysis) | Done | `templates/index.html` |
| Progress | Done | `templates/progress.html` |
| Results | Done | `templates/results.html` |
| Dashboard (Browse) | Done | `templates/dashboard.html` |
| Duplicates Review | Done | `templates/duplicates.html` |
| Chat/Search | Done | `templates/chat.html` |

### Database Schema

| Table | Status | Purpose |
|-------|--------|---------|
| `photos` | Done | Photo metadata + hashes |
| `analysis_results` | Done | Per-analyzer results |
| `duplicate_groups` | Done | Duplicate groupings |
| `duplicate_members` | Done | Photos in groups |
| `jobs` | Done | Async job tracking |
| `job_metrics` | Done | Performance metrics |
| `action_history` | Done | Undo stack |
| `conversations` | Done | Chat sessions |
| `conversation_messages` | Done | Chat history |

### Tests

| Area | Count | Status |
|------|-------|--------|
| Analyzers | 21 | Passing |
| Classifiers | 12 | Passing |
| Database | 18 | Passing |
| Orchestrator | 12 | Passing |
| API Health | 1 | Passing |
| **Total** | **77+** | **All passing** |

---

## Future Enhancements (Post-V1)

### High Priority

| Feature | Description | Complexity |
|---------|-------------|------------|
| Separate Search Model | Use LFM-1B (text-only) for faster search (~0.3s vs 2-3s) | Medium |
| Semantic Duplicates | LFM-powered "same scene, different angle" detection | Medium |
| EXIF Location Search | "Photos from Paris" using GPS coordinates | Low |

### Medium Priority

| Feature | Description | Complexity |
|---------|-------------|------------|
| Native Mac App | SwiftUI frontend with Photos.app integration | High |
| CLI Tool | `photo-triage analyze ~/Pictures` command | Low |
| Auto-Organize | Create folders by category/date/event | Medium |
| Batch Export | Export selected photos to folder | Low |

### Nice to Have

| Feature | Description | Complexity |
|---------|-------------|------------|
| Face Recognition | Group by person (on-device) | High |
| RAW Support | Analyze RAW formats (CR2, NEF, ARW) | Medium |
| Video Thumbnails | Extract and analyze video frames | Medium |
| iCloud Integration | Direct Photos.app library access | High |
| Sync Service | Watch folder for new photos | Medium |

---

## Known Limitations

### Current Limitations

1. **Search requires LFM analysis**: Photos analyzed with `skip_lfm=true` have limited search capability (filename-only matching)

2. **Single-threaded LFM**: Only one classification at a time (subprocess-based)

3. **Memory usage**: LFM inference uses 3-5GB RAM

4. **No HEIC thumbnails**: Thumbnails require PIL-compatible format

### Architecture Constraints

1. **SQLite single-writer**: No concurrent write access from multiple processes

2. **Local-only**: No cloud sync or multi-device support

3. **macOS primary**: Tested on macOS, Linux should work but untested

---

## Configuration Reference

```bash
# Environment variables
API_HOST=127.0.0.1          # Server bind address
API_PORT=8000               # Server port
MODEL_N_CTX=2048            # LFM context size
MODEL_N_GPU_LAYERS=-1       # GPU layers (-1 = all)
LFM_CLI_PATH=llama-mtmd-cli # Path to llama.cpp CLI
```

---

## Quick Start

```bash
# 1. Install dependencies
cd backend && uv sync

# 2. Download model (one-time)
./scripts/download_model.sh

# 3. Start server
./scripts/run_backend.sh

# 4. Open browser
open http://localhost:8000

# 5. Run tests
uv run pytest tests/ -v
```

---

## File Structure

```
backend/
├── src/
│   ├── main.py              # App entry point
│   ├── config.py            # Configuration
│   ├── orchestrator.py      # Batch processing
│   ├── api/
│   │   ├── routes.py        # HTTP endpoints
│   │   └── schemas.py       # Request/response models
│   ├── database/
│   │   ├── schema.py        # Table definitions
│   │   └── repository.py    # CRUD operations
│   ├── analyzers/
│   │   ├── blur.py          # Blur detection
│   │   ├── screenshot.py    # Screenshot detection
│   │   └── hasher.py        # Perceptual hashing
│   ├── classifiers/
│   │   ├── lfm_cli_provider.py  # LFM via CLI
│   │   └── mock_provider.py     # Testing mock
│   ├── services/
│   │   ├── job_service.py   # Job management
│   │   ├── search_agent.py  # NL search
│   │   └── conversation.py  # Chat state
│   └── web/
│       ├── templates/       # Jinja2 HTML
│       └── static/          # CSS, JS
├── tests/                   # Test suite
├── models/                  # LFM model files
└── data/                    # SQLite database
```

---

## Related Documentation

- **[README.md](../backend/README.md)** - User-facing documentation
- **[ARCHITECTURE.md](./ARCHITECTURE.md)** - Technical architecture with diagrams
- **[CLAUDE.md](../CLAUDE.md)** - Coding guidelines and patterns
