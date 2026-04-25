# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**hailo-apps** is a production-ready infrastructure for deploying AI applications (computer vision, GenAI) on Hailo hardware accelerators (Hailo-8, Hailo-8L, Hailo-10H) running on Raspberry Pi 5. Python 3.10+, MIT license, version 25.12.0.

**Hardware target:** Raspberry Pi 5 + Hailo AI Hat+2 (Hailo-10H, 40 TOPS NPU).

## Build & Development Commands

```bash
# Environment setup (must be sourced, not executed)
source setup_env.sh

# Install package in editable mode
pip install -e .              # core only
pip install -e ".[dev]"       # with dev tools (ruff, pytest, pre-commit)
pip install -e ".[gen-ai]"    # with GenAI dependencies (PyAudio, piper-tts, sounddevice, webrtcvad)
pip install -e ".[ocr]"       # with OCR dependencies (paddlepaddle, shapely, pyclipper, symspellpy)
pip install -e ".[speech-rec]" # with speech recognition (transformers, torch, streamlit)

# Compile C++ post-processing shared libraries (Meson build)
hailo-compile-postprocess

# Download model resources for detected architecture
hailo-download-resources
```

### Environment Setup Details

`setup_env.sh` must be sourced (not executed). It performs:
1. **Kernel version check** — blocks incompatible RPi kernels (6.12.21–6.12.25)
2. **PYTHONPATH** — prepends project root
3. **Virtualenv** — activates `venv_hailo_apps`
4. **Environment vars** — loads `/usr/local/hailo/resources/.env` (hailort/tappas versions, architecture info)

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

Test configuration lives in `tests/test_control.yaml` (runtime params, 40s default) and `hailo_apps/config/test_definition_config.yaml`.

## Architecture

### Three Application Types

1. **Pipeline Apps** (`hailo_apps/python/pipeline_apps/`) — Production GStreamer-based video processing (detection, pose, segmentation, face recognition, CLIP, OCR, depth, tiling, multi-camera, ReID). Each app is a CLI entry point registered in `pyproject.toml [project.scripts]`.

2. **Standalone Apps** (`hailo_apps/python/standalone_apps/`, `hailo_apps/cpp/`) — Learning/prototype apps using HailoRT directly, no TAPPAS required.

3. **GenAI Apps** (`hailo_apps/python/gen_ai_apps/`) — Hailo-10H only: LLM chat, VLM chat, voice assistants, speech recognition, agent framework, Ollama-compatible API.

### Pipeline Apps — All CLI Entry Points

| Command | App | Description |
|---|---|---|
| `hailo-detect` | detection | Object detection (YOLOv5/6/7/8/9/10/11 variants) |
| `hailo-detect-simple` | detection_simple | Lightweight detection (YOLOv6n) |
| `hailo-pose` | pose_estimation | Human pose estimation (YOLOv8 Pose, 17 keypoints) |
| `hailo-seg` | instance_segmentation | Instance segmentation (YOLOv5/8 Seg) |
| `hailo-depth` | depth | Monocular depth estimation (SCDepthV3) |
| `hailo-face-recon` | face_recognition | Face recognition (SCRFD + ArcFace + LanceDB) |
| `hailo-clip` | clip | Zero-shot classification (CLIP ViT-B/32) |
| `hailo-ocr` | paddle_ocr | Text detection + recognition (PaddleOCR) |
| `hailo-tiling` | tiling | High-res small object detection via tile grid |
| `hailo-multisource` | multisource | Multi-camera parallel processing |
| `hailo-reid` | reid_multisource | Cross-camera face re-identification |
| `get-usb-camera` | camera_utils | Enumerate available USB cameras |
| `hailo-audio-troubleshoot` | audio_troubleshoot | Audio device diagnostics |

### Typical Pipeline App Structure
```
hailo_apps/python/pipeline_apps/<app_name>/
├── <app_name>.py              # CLI entry point with argparse
├── <app_name>_pipeline.py     # GStreamer pipeline class (extends GStreamerApp)
└── README.md
```

### GenAI Apps (Hailo-10H Only)

Located in `hailo_apps/python/gen_ai_apps/`:

| App | Run Command | Description |
|---|---|---|
| **Interactive LLM Chat** | `python -m hailo_apps.python.gen_ai_apps.interactive_llm_chat.interactive_llm_chat` | Multi-turn terminal chat (Qwen2.5-1.5B-Instruct) |
| **Simple LLM Chat** | `python -m hailo_apps.python.gen_ai_apps.simple_llm_chat.simple_llm_chat` | Basic text generation |
| **VLM Chat** | `python -m hailo_apps.python.gen_ai_apps.vlm_chat.vlm_chat --input usb` | Vision-language Q&A on live video (Qwen2-VL-2B-Instruct) |
| **Simple VLM Chat** | `python -m hailo_apps.python.gen_ai_apps.simple_vlm_chat.simple_vlm_chat` | Image-based visual Q&A |
| **Voice Assistant** | `python -m hailo_apps.python.gen_ai_apps.voice_assistant.voice_assistant` | Whisper STT + LLM + Piper TTS voice loop |
| **Simple Whisper** | `python -m hailo_apps.python.gen_ai_apps.simple_whisper_chat.simple_whisper_chat --audio file.wav` | Audio file transcription (Whisper-Base) |
| **Agent Tools** | `python -m hailo_apps.python.gen_ai_apps.agent_tools_example.agent [--voice]` | LLM with function calling (Qwen2.5-Coder-1.5B-Instruct) |

#### Agent Tools Framework

The agent tools example (`agent_tools_example/`) provides an extensible function-calling system:
- **Tool discovery**: Auto-discovers tools from module files in `tools/` directory
- **Base class**: `BaseTool` with `name`, `description`, `schema`, `run()`, `initialize()`, `cleanup()`
- **Built-in tools**: Math, Weather API, RGB LED control (NeoPixel via SPI), Servo control (hardware PWM), Elevator simulation
- **Dual mode**: Each hardware tool supports real hardware or Flask-based web simulator (configured in `config.py`)
- **Modes**: Text or voice interaction (`--voice` flag)

#### Hailo Ollama (Open WebUI Integration)

Located in `hailo_apps/python/gen_ai_apps/hailo_ollama/`. Provides an Ollama-compatible REST API for Hailo-accelerated LLM inference:

```bash
# Start the Ollama-compatible API server (port 8000)
hailo-ollama

# Pull a model
curl http://localhost:8000/api/pull -H 'Content-Type: application/json' \
     -d '{"model": "qwen2.5-instruct:1.5b", "stream": true}'

# Open WebUI (Docker, port 8080)
docker pull ghcr.io/open-webui/open-webui:main
docker run -d -e OLLAMA_BASE_URL=http://127.0.0.1:8000 \
  -v open-webui:/app/backend/data --name open-webui \
  --network=host --restart always ghcr.io/open-webui/open-webui:main
```

Requires Hailo GenAI Model Zoo (`hailo_gen_ai_model_zoo` deb package, v5.1.1 or v5.2.0).

#### GenAI Utils (`gen_ai_utils/`)

Shared infrastructure for all GenAI apps:
- **`llm_utils/`** — Context manager (token-window tracking), message formatter, streaming response handler, tool discovery/execution framework, terminal UI components
- **`voice_processing/`** — `AudioRecorder` (mic auto-detection), `SpeechToTextProcessor` (Whisper), `TextToSpeechProcessor` (Piper TTS, e.g. `en_US-amy-low`), `AudioPlayer` (with interruption support), `VoiceInteractionManager`, `AudioDiagnostics`

### Core Library (`hailo_apps/python/core/`)

Shared infrastructure used by all application types:

- **`gstreamer/`** — `GStreamerApp` base class (`gstreamer_app.py`), pipeline builder helpers (`gstreamer_helper_pipelines.py`), common GStreamer utilities
- **`common/`** — `hailo_inference.py` (async HailoRT inference with batching), `parser.py` (CLI args), `camera_utils.py` (input detection/enumeration), `defines.py` (constants, arch URLs), `hef_utils.py` (model file utils), `buffer_utils.py` (GStreamer buffer ops), `hailo_logger.py` (loguru-based logging), `db_handler.py` (LanceDB vector DB), `telegram_handler.py` (Telegram notifications), `core.py` (env loading)
- **`tracker/`** — ByteTrack multi-object tracking with Kalman filter

#### LanceDB Vector Database (`db_handler.py`)

Used by face recognition and ReID pipelines:
- 512-dimensional face embeddings via ArcFace
- Cosine similarity search with confidence thresholds
- CRUD operations on records with multi-sample averaging
- BTREE indexing on `global_id` and `label` fields

#### Telegram Notifications (`telegram_handler.py`)

Sends detection alerts with images via Telegram Bot API. Used by face recognition pipeline. Rate-limited to 1 notification per person per hour.

#### Face Recognition Pipeline

Three operational modes:
- **`--mode train`** — Enroll faces from camera/images into LanceDB
- **`--mode run`** — Real-time face matching against the database
- **`--mode delete`** — Remove face records from the database

Pipeline: SCRFD face detection → face alignment → ArcFace embedding → LanceDB similarity search → optional Telegram alert.

### C++ Post-Processing (`hailo_apps/postprocess/cpp/`)

20+ shared libraries (.so) built with **Meson** for performance-critical inference post-processing. Key modules:

| Module | Purpose |
|---|---|
| `yolo_hailortpp.cpp` | YOLO detection NMS, decoding, filtering |
| `yolov5seg.cpp` | YOLOv5 segmentation mask decoding |
| `yolov8pose_postprocess.cpp` | YOLOv8 pose keypoint processing |
| `scrfd.cpp` | Face detection anchor decoding |
| `arcface.cpp` | Face embedding normalization |
| `clip.cpp` | CLIP embedding normalization + similarity |
| `depth_estimation.cpp` | Depth map post-processing |
| `ocr_postprocess.cpp` | OCR text detection/recognition |
| `clip_croppers.cpp` | Region cropping for CLIP |
| `face_align.hpp` | Face alignment transformation |

Compiled outputs go to `resources/so/`.

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

## Running Applications

```bash
source setup_env.sh

# Pipeline apps (CLI entry points)
hailo-detect                              # default camera, default model
hailo-detect --input usb                  # USB webcam
hailo-detect --input rpi                  # Raspberry Pi camera
hailo-detect --input /path/to/video.mp4   # video file
hailo-detect --input rtsp://...           # RTSP stream

# GenAI apps (run as modules)
python -m hailo_apps.python.gen_ai_apps.interactive_llm_chat.interactive_llm_chat
python -m hailo_apps.python.gen_ai_apps.vlm_chat.vlm_chat --input usb
python -m hailo_apps.python.gen_ai_apps.voice_assistant.voice_assistant
python -m hailo_apps.python.gen_ai_apps.agent_tools_example.agent --voice
```

Pipeline apps accept `--input` for camera/video/RTSP sources, `--hef-path` for custom models, `--batch-size`, `--frame-rate`, `--sync/--disable-sync`, and various model/network flags. Use `--help` on any CLI command for options.

### Supported Input Sources

- **RPi Camera:** `--input rpi` (CSI camera via libcamera)
- **USB Camera:** `--input usb` or `/dev/videoX`
- **Video files:** MP4, AVI, MOV, MKV
- **Images:** JPG, JPEG, PNG, BMP
- **RTSP streams:** `rtsp://...`
- **Screen capture:** X11 `ximage` source

## Available Models (Hailo-10H)

### Computer Vision
- **Detection:** YOLOv5 (m/s), YOLOv6n, YOLOv7 (std/x), YOLOv8 (n/s/m/l/x), YOLOv9c, YOLOv10 (n/s/b/x), YOLOv11 (n/s/m/l/x)
- **Pose:** YOLOv8m_pose, YOLOv8s_pose
- **Instance Seg:** YOLOv5 (n/s/m/l)_seg, YOLOv8 (n/s/m)_seg, FastSAM-S
- **Semantic Seg:** FCN8 ResNet, SegFormer B0, STDC1, DeepLab V3
- **Depth:** SCDepthV3 (monocular), StereoNet (stereo)
- **Face:** SCRFD (10g/2.5g) + ArcFace MobileFaceNet
- **CLIP:** ViT-B/32 (image + text encoders)
- **OCR:** PaddleOCR DB (detection) + CRNN (recognition)
- **Lane Detection:** UFLD V2 TU
- **Super Resolution:** Real ESRGAN x2
- **Oriented Detection:** YOLO11s OBB

### Generative AI (Hailo-10H only)
- **LLM:** Qwen2.5-1.5B-Instruct
- **VLM:** Qwen2-VL-2B-Instruct
- **Speech-to-Text:** Whisper-Base
- **Text-to-Speech:** Piper TTS (multiple voice models)
- **Agent LLM:** Qwen2.5-Coder-1.5B-Instruct

## Hardware Control Interfaces

Available via the agent tools framework (`agent_tools_example/tools/`):

- **NeoPixel LEDs** — RGB control via SPI (`/dev/spidevX.X`), uses `rpi5-ws2812` driver
- **Servo Motors** — PWM control via GPIO pins 18/19, uses `rpi-hardware-pwm`
- Both support a Flask web simulator mode for development without hardware (toggle in `config.py`)

## Key Dependencies

- **opencv-python** pinned ≤4.10.0.84 (avoids Qt font warnings)
- **GStreamer** + **TAPPAS Core** (system packages, required for pipeline apps)
- **HailoRT** (system package, required for all hardware inference)
- **Hailo GenAI Model Zoo** (system package, required for GenAI apps + Ollama, v5.1.1/v5.2.0)
- **loguru** for logging, **lancedb** for vector DB, **scipy/lap/cython_bbox** for tracking
- **PyAudio/sounddevice/piper-tts** for voice apps (`[gen-ai]` extras)
- **paddlepaddle** for OCR (`[ocr]` extras)
- **transformers/torch** for Whisper (`[speech-rec]` extras)
- **Docker** for Open WebUI integration (optional)
