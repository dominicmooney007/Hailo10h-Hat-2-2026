# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**hailo-apps** is a production-ready infrastructure for deploying AI applications (computer vision, GenAI) on Hailo hardware accelerators (Hailo-8, Hailo-8L, Hailo-10H). Python 3.10+, MIT license, version 25.12.0.

## Build & Development Commands

```bash
# Environment setup (must be sourced, not executed)
source setup_env.sh

# Install package in editable mode
pip install -e .              # core only
pip install -e ".[dev]"       # with dev tools (ruff, pytest, pre-commit)
pip install -e ".[gen-ai]"    # with GenAI dependencies
pip install -e ".[ocr]"       # with OCR dependencies
pip install -e ".[speech-rec]" # with speech recognition dependencies

# Compile C++ post-processing shared libraries (Meson build)
hailo-compile-postprocess

# Download model resources for detected architecture
hailo-download-resources
```

## Linting & Formatting

Uses **ruff** (configured in `pyproject.toml`). Line length: 100. Target: Python 3.10.

```bash
ruff check --fix .            # lint with auto-fix
ruff format .                 # format
```

Pre-commit hooks run ruff automatically on `git commit` (configured in `.pre-commit-config.yaml`). The hooks activate the `venv_hailo_apps` virtualenv internally.

Key ruff rules: E, F, I (isort), B (bugbear), UP (pyupgrade), C4, W, RUF. Ignored: E402 (gi.require_version ordering), F403/F405 (star imports used in GStreamer code).

## Testing

```bash
# Full test suite (downloads resources, runs all 3 tiers)
./run_tests.sh

# Individual test tiers
./run_tests.sh --sanity       # environment validation (imports, Python version, GStreamer, HailoRT)
./run_tests.sh --install      # resource validation (models, videos, SO files)
./run_tests.sh --pipelines    # functional pipeline tests
./run_tests.sh --no-download  # skip resource download step

# Run specific test files directly
python -m pytest tests/test_sanity_check.py -v
python -m pytest tests/test_installation.py -v
python -m pytest tests/test_runner.py -v
python -m pytest tests/test_hef_utils.py -v
python -m pytest tests/test_face_recon.py -v
```

Test configuration lives in `tests/test_control.yaml` and `hailo_apps/config/test_definition_config.yaml`.

## Architecture

### Three Application Types

1. **Pipeline Apps** (`hailo_apps/python/pipeline_apps/`) — Production GStreamer-based video processing (detection, pose, segmentation, face recognition, CLIP, OCR, depth, tiling, multi-camera, ReID). Each app is a CLI entry point registered in `pyproject.toml [project.scripts]` (e.g., `hailo-detect`, `hailo-pose`).

2. **Standalone Apps** (`hailo_apps/python/standalone_apps/`, `hailo_apps/cpp/`) — Learning/prototype apps using HailoRT directly, no TAPPAS required.

3. **GenAI Apps** (`hailo_apps/python/gen_ai_apps/`) — Hailo-10H generative AI: voice assistants, VLM chat, voice-to-action agents, Whisper speech recognition.

### Typical Pipeline App Structure
```
hailo_apps/python/pipeline_apps/<app_name>/
├── <app_name>.py              # CLI entry point with argparse
├── <app_name>_pipeline.py     # GStreamer pipeline class (extends GStreamerApp)
└── README.md
```

### Core Library (`hailo_apps/python/core/`)

Shared infrastructure used by all application types:

- **`gstreamer/`** — `GStreamerApp` base class (`gstreamer_app.py`), pipeline builder helpers (`gstreamer_helper_pipelines.py`), common GStreamer utilities
- **`common/`** — `hailo_inference.py` (HailoRT wrapper), `parser.py` (CLI args), `camera_utils.py` (input detection), `defines.py` (constants, arch URLs), `hef_utils.py` (model file utils), `buffer_utils.py` (GStreamer buffer ops), `hailo_logger.py` (loguru-based logging), `db_handler.py` (LanceDB), `core.py` (env loading)
- **`tracker/`** — ByteTrack implementation with Kalman filter

### C++ Post-Processing (`hailo_apps/postprocess/cpp/`)

20+ shared libraries (.so) built with **Meson** for performance-critical inference post-processing (YOLO variants, depth, CLIP, ArcFace, OCR, SSD, SCRFD). Compiled outputs go to `resources/so/`.

### Configuration System (`hailo_apps/config/`)

- `config.yaml` — Runtime settings, device/architecture auto-detection, resource paths
- `resources_config.yaml` — Model/video/image definitions per architecture (hailo8/8l/10h)
- `config_manager.py` — Unified interface for querying all configs
- `networks.json`, `inputs.json` — Model zoo metadata and input specs

### Resources

Default path: `/usr/local/hailo/resources/` (symlinked as `resources/` in repo root). Contains: `models/` (HEF files by arch), `videos/`, `images/`, `json/`, `npy/`, `so/` (compiled post-processing).

### Installation System (`hailo_apps/installation/`)

- `download_resources.py` — Parallel downloads with retry, validation, architecture detection
- `compile_cpp.py` — Meson-based C++ compilation
- `set_env.py` — Environment variable configuration
- `post_install.py` — Post-installation setup

### Environment

The virtual environment is named `venv_hailo_apps`. Environment variables are loaded from `/usr/local/hailo/resources/.env` (contains detected hailort/tappas versions, architecture info). The `setup_env.sh` script sets PYTHONPATH to project root and activates the venv.

## Running Applications

```bash
source setup_env.sh
hailo-detect                  # use CLI entry points
# or run directly:
python hailo_apps/python/pipeline_apps/detection/detection.py
```

Pipeline apps accept `--input` for camera/video/RTSP sources and various model/network flags. Use `--help` on any CLI command for options.

## Key Dependencies

- **opencv-python** pinned ≤4.10.0.84 (avoids Qt font warnings)
- **GStreamer** + **TAPPAS Core** (system packages, required for pipeline apps)
- **HailoRT** (system package, required for all hardware inference)
- **loguru** for logging, **lancedb** for vector DB, **scipy/lap/cython_bbox** for tracking
