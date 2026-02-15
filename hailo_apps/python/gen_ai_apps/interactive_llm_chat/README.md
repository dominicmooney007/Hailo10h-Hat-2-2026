# Interactive LLM Chat — User Guide

A terminal-based interactive chat application that runs large language models on the Hailo-10H AI accelerator. You type messages, the model streams responses back token-by-token, and multi-turn conversation history is maintained automatically.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Command-Line Options](#command-line-options)
- [In-Chat Commands](#in-chat-commands)
- [How the Code Works](#how-the-code-works)
  - [1. Model Discovery](#1-model-discovery)
  - [2. Device and Model Initialization](#2-device-and-model-initialization)
  - [3. Conversation History](#3-conversation-history)
  - [4. Streaming Token Generation](#4-streaming-token-generation)
  - [5. Context Window Management](#5-context-window-management)
  - [6. Cleanup and Shutdown](#6-cleanup-and-shutdown)
- [Hailo Pipeline Dependencies](#hailo-pipeline-dependencies)
  - [Hardware Requirements](#hardware-requirements)
  - [HailoRT Runtime](#hailort-runtime)
  - [hailo_platform Python SDK](#hailo_platform-python-sdk)
  - [HEF Model Files](#hef-model-files)
  - [hailo-apps Shared Utilities](#hailo-apps-shared-utilities)
- [Architecture Diagram](#architecture-diagram)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

| Requirement | Details |
|---|---|
| Hardware | Hailo-10H accelerator (M.2 or Hat form factor) |
| OS | Linux (tested on Raspberry Pi OS / Debian) |
| Python | 3.10 or later |
| HailoRT | System package installed and driver loaded |
| hailo-apps | This repository, installed with `pip install -e ".[gen-ai]"` |
| Model files | At least one LLM HEF downloaded (see [Downloading Models](#downloading-models)) |

## Quick Start

```bash
# 1. Activate the environment
source setup_env.sh

# 2. Run the chat (shows a model picker if multiple models are downloaded)
python hailo_apps/python/gen_ai_apps/interactive_llm_chat/interactive_llm_chat.py

# 3. Or specify a model directly
python hailo_apps/python/gen_ai_apps/interactive_llm_chat/interactive_llm_chat.py \
    --hef-path Qwen2.5-1.5B-Instruct
```

### Downloading Models

If no models are downloaded yet:

```bash
source setup_env.sh
hailo-download-resources
```

This downloads all default models for your detected architecture into `/usr/local/hailo/resources/models/hailo10h/`. The LLM models defined in the project config are:

| Model | Config App | Description |
|---|---|---|
| `Qwen2.5-1.5B-Instruct` | `llm_chat` | General-purpose 1.5B parameter instruction-tuned model |
| `Qwen2.5-Coder-1.5B-Instruct` | `agent` | Code-focused 1.5B parameter instruction-tuned model |

## Command-Line Options

```
usage: interactive_llm_chat.py [-h] [--hef-path HEF_PATH]
                                [--max-tokens MAX_TOKENS]
                                [--temperature TEMPERATURE]
                                [--system-prompt SYSTEM_PROMPT]
```

| Flag | Default | Description |
|---|---|---|
| `--hef-path` | *(interactive picker)* | Path or name of a HEF model file. Skips the model selection menu. Can be a full path, a filename, or just the model name (e.g. `Qwen2.5-1.5B-Instruct`). |
| `--max-tokens` | `512` | Maximum number of tokens the model generates per response. Higher values allow longer answers but use more context. |
| `--temperature` | `0.7` | Controls randomness. `0.0` = deterministic, `1.0` = more creative. Values around `0.3-0.7` work well for conversation. |
| `--system-prompt` | `"You are a helpful assistant. Be concise and clear in your responses."` | Sets the model's persona and behavioral instructions. |

### Examples

```bash
# Creative writing mode
python interactive_llm_chat.py --temperature 0.9 --max-tokens 1024 \
    --system-prompt "You are a creative storyteller."

# Coding assistant with the Coder model
python interactive_llm_chat.py --hef-path Qwen2.5-Coder-1.5B-Instruct \
    --system-prompt "You are a Python coding assistant. Provide concise code examples."

# Deterministic Q&A
python interactive_llm_chat.py --temperature 0.1 --max-tokens 256
```

## In-Chat Commands

Once the chat is running, type any of these instead of a normal message:

| Command | Action |
|---|---|
| `/help` | Show the list of available commands |
| `/clear` | Erase all conversation history and reset the model's context window |
| `/context` | Display a visual progress bar showing how full the context window is |
| `/quit` or `/exit` | End the chat session and release hardware resources |
| `Ctrl+C` | Immediate graceful exit |

---

## How the Code Works

The application is a single Python script organized into four phases: model discovery, hardware initialization, the interactive loop, and cleanup.

### 1. Model Discovery

**Function: `find_downloaded_llm_models()`** (lines 39-68)

When no `--hef-path` is given, the app scans the Hailo resources directory for downloaded models:

```
/usr/local/hailo/resources/models/hailo10h/*.hef
```

Not every `.hef` file is an LLM. The function filters candidates using two strategies:

1. **Config cross-reference** — Queries `resources_config.yaml` via `get_model_names()` for apps `llm_chat` and `agent`, which are the two app entries that use text-only LLM models. Any `.hef` file whose stem matches a name from these configs is included.

2. **Naming pattern heuristic** — For models not in the config (e.g. manually downloaded), it includes files containing `"Instruct"` in the name while excluding known non-LLM prefixes: `Whisper` (speech), `clip_vit` (vision encoder), and `Qwen2-VL` (vision-language, requires a different API).

If multiple candidates are found, `select_model()` presents a numbered menu. If only one is found, it is used automatically.

### 2. Device and Model Initialization

**Location:** `main()`, lines 143-149

Initialization follows the standard Hailo GenAI pattern:

```python
# 1. Create a virtual device handle with a shared group ID
params = VDevice.create_params()
params.group_id = SHARED_VDEVICE_GROUP_ID   # "SHARED"
vdevice = VDevice(params)

# 2. Load the compiled model (HEF) onto the device
llm = LLM(vdevice, str(hef_path))
```

**`VDevice`** is the HailoRT abstraction over the physical accelerator. The `group_id` parameter allows multiple processes to share the same physical device — setting it to `"SHARED"` means this app can coexist with other Hailo applications.

**`LLM`** is the high-level GenAI class from `hailo_platform.genai`. It wraps the low-level inference pipeline and provides a chat-oriented API: `generate()` for streaming, `generate_all()` for batch, plus context management methods.

### 3. Conversation History

**Location:** `main()`, lines 151-196

The conversation is stored as a Python list of message dictionaries:

```python
conversation = [messages_system(system_prompt)]
```

Each message uses the format expected by the Hailo LLM API:

```python
{"role": "system",    "content": "You are a helpful assistant."}
{"role": "user",      "content": "What is Python?"}
{"role": "assistant", "content": "Python is a programming language..."}
```

The helper functions `messages_system()`, `messages_user()`, and `messages_assistant()` from `message_formatter.py` construct these dictionaries. The full conversation list is sent to the model on every turn, which is how the model "remembers" prior exchanges — it re-reads the entire history each time.

After each model response, the raw output is cleaned with `clean_response()` (strips XML wrapper tags like `<text>...</text>` and special tokens like `<|im_end|>`) and appended to the history.

### 4. Streaming Token Generation

**Location:** `main()`, lines 184-196, delegated to `generate_and_stream_response()`

Instead of waiting for the full response, tokens are printed one-by-one as the model produces them:

```python
raw_response = generate_and_stream_response(
    llm,
    prompt=conversation,
    temperature=args.temperature,
    max_tokens=args.max_tokens,
    prefix="Assistant: ",
    show_raw_stream=False,
)
```

Under the hood, `generate_and_stream_response()` (from `streaming.py`) does the following:

1. Calls `llm.generate()` which returns a context manager yielding tokens.
2. Each token passes through `StreamingTextFilter`, which maintains a state machine to strip XML tags (`<text>`, `<tool_call>`, `<tool_response>`) and special tokens (`<|im_end|>`) that the model may emit.
3. With `show_raw_stream=False`, only the cleaned text reaches the terminal — the user sees a smooth, readable stream.
4. The raw (unfiltered) response string is returned for storage in conversation history and further processing.

The `StreamingTextFilter` class handles edge cases like tags split across token boundaries by maintaining an internal buffer.

### 5. Context Window Management

**Location:** `main()`, lines 198-202

LLMs have a fixed context window (the maximum number of tokens they can "see" at once). As the conversation grows, it fills up. The app handles this automatically:

```python
if is_context_full(llm):
    print("[Context full - clearing history to continue]")
    llm.clear_context()
    conversation = [messages_system(system_prompt)]
```

`is_context_full()` (from `context_manager.py`) checks whether token usage has exceeded 95% of the model's maximum capacity by calling `llm.get_context_usage_size()` and `llm.max_context_capacity()`. When the threshold is crossed, context is cleared and conversation history resets to just the system prompt.

The `/context` command lets you check usage at any time, displaying a visual bar:

```
[Info] Context: [████████░░░░░░░░░░░░░░░░░░░░░░] 847/4096 (20%)
```

The `/clear` command manually resets context without waiting for it to fill.

### 6. Cleanup and Shutdown

**Location:** `main()`, lines 210-221

A `finally` block ensures hardware resources are always released, even if an error occurs:

```python
finally:
    if llm:
        llm.clear_context()   # Free context memory on the accelerator
        llm.release()         # Release the model from the device
    if vdevice:
        vdevice.release()     # Release the device handle
```

The order matters: the model must be released before the device. `Ctrl+C` during the input prompt triggers `KeyboardInterrupt`, which is caught by the loop and falls through to this cleanup block.

---

## Hailo Pipeline Dependencies

This section explains each layer of the Hailo software stack that the interactive chat depends on.

### Hardware Requirements

The LLM functionality requires a **Hailo-10H** accelerator. This is the only Hailo chip that supports generative AI workloads (autoregressive token generation). The Hailo-8 and Hailo-8L chips support inference for vision models (detection, segmentation, etc.) but lack the architecture needed for LLM execution.

The Hailo-10H is available as:
- **M.2 module** — for standard PCIe slots
- **Raspberry Pi AI Hat+** — purpose-built hat for Raspberry Pi 5

### HailoRT Runtime

**System package:** `hailort` (or `h10-hailort` on Raspberry Pi)

HailoRT is the low-level runtime that communicates with the Hailo hardware over PCIe. It provides:

- **Device management** — discovery, power, reset
- **Memory management** — DMA buffers, context allocation
- **Inference execution** — sending data to the chip and receiving results
- **Virtual device abstraction** (`VDevice`) — allows multiple applications to share a single physical chip via group IDs

The interactive chat uses HailoRT indirectly through the `hailo_platform` Python package. You can verify the driver is loaded with:

```bash
lsmod | grep hailo
hailort --version
```

### hailo_platform Python SDK

**Python package:** `hailo_platform` (installed as part of HailoRT)

This is the Python binding layer over HailoRT. The two classes used by the chat are:

| Class | Import | Purpose |
|---|---|---|
| `VDevice` | `hailo_platform.VDevice` | Creates a handle to the physical accelerator. The `create_params()` / `group_id` pattern enables device sharing between processes. |
| `LLM` | `hailo_platform.genai.LLM` | High-level GenAI interface. Takes a `VDevice` and a `.hef` file path. Provides `generate()` (streaming), `generate_all()` (batch), context management (`clear_context`, `get_context_usage_size`, `max_context_capacity`, `save_context`, `load_context`). |

The `LLM` class handles the complexity of autoregressive generation on hardware: it manages the KV-cache on the accelerator, handles tokenization, and yields decoded text tokens through its generator interface.

### HEF Model Files

**Format:** Hailo Executable Format (`.hef`)
**Location:** `/usr/local/hailo/resources/models/hailo10h/`

HEF files are pre-compiled neural network binaries optimized for a specific Hailo chip architecture. They are **not** interchangeable between architectures (a hailo8 HEF will not run on hailo10h, and vice versa).

For GenAI models, the HEF contains the transformer architecture (attention layers, embeddings, etc.) compiled for the Hailo-10H's dataflow architecture. The compilation is done offline using Hailo's model compilation tools — the end user downloads pre-compiled HEFs.

**Model registry:** Models are declared in `hailo_apps/config/resources_config.yaml` under app-specific sections. The `llm_chat` section currently defines:

```yaml
llm_chat:
  models:
    hailo10h:
      default:
        - name: Qwen2.5-1.5B-Instruct
          source: gen-ai-mz    # downloaded from Hailo's Gen-AI Model Zoo
```

The `source: gen-ai-mz` field tells the download system to fetch from Hailo's Gen-AI Model Zoo server. The `resolve_hef_path()` function in `core.py` handles the full resolution pipeline: checking local paths, the resources directory, and triggering auto-download for missing models.

### hailo-apps Shared Utilities

The interactive chat reuses several utility modules from the hailo-apps project rather than reimplementing common functionality:

| Module | File | What it provides |
|---|---|---|
| **Config Manager** | `hailo_apps/config/config_manager.py` | `get_model_names(app, arch, tier)` — queries `resources_config.yaml` to enumerate models registered for a given app and architecture. Used by the model discovery function. |
| **Defines** | `hailo_apps/python/core/common/defines.py` | Constants: `HAILO10H_ARCH` (`"hailo10h"`), `SHARED_VDEVICE_GROUP_ID` (`"SHARED"`), `LLM_CHAT_APP` (`"llm_chat"`), `RESOURCES_ROOT_PATH_DEFAULT` (`"/usr/local/hailo/resources"`), `HAILO_FILE_EXTENSION` (`".hef"`). |
| **Core** | `hailo_apps/python/core/common/core.py` | `resolve_hef_path()` — multi-strategy HEF resolution (local path, resources dir, auto-download). Used when `--hef-path` is provided. |
| **Logger** | `hailo_apps/python/core/common/hailo_logger.py` | `get_logger()` — loguru-based logging with consistent formatting across all hailo-apps. |
| **Message Formatter** | `hailo_apps/python/gen_ai_apps/gen_ai_utils/llm_utils/message_formatter.py` | `messages_system()`, `messages_user()`, `messages_assistant()` — construct message dictionaries in the format expected by the Hailo LLM API (`{"role": "...", "content": "..."}`). |
| **Streaming** | `hailo_apps/python/gen_ai_apps/gen_ai_utils/llm_utils/streaming.py` | `generate_and_stream_response()` — wraps `llm.generate()` with real-time token printing and `StreamingTextFilter` to strip XML/special tokens. `clean_response()` — post-generation cleanup of the raw response string. |
| **Context Manager** | `hailo_apps/python/gen_ai_apps/gen_ai_utils/llm_utils/context_manager.py` | `is_context_full()` — checks if token usage exceeds 95% of capacity. `print_context_usage()` — displays a visual progress bar of context utilization. |

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────┐
│                    User Terminal                     │
│                                                     │
│   You: What is Python?                              │
│   Assistant: Python is a high-level programming...  │
│                                                     │
└────────────────────────┬────────────────────────────┘
                         │ stdin/stdout
                         ▼
┌─────────────────────────────────────────────────────┐
│              interactive_llm_chat.py                 │
│                                                     │
│  ┌──────────────┐  ┌────────────────────────────┐   │
│  │ Model Picker  │  │ Chat Loop                  │   │
│  │              │  │  ┌─────────────────────┐   │   │
│  │ Scan .hef    │  │  │ Conversation History │   │   │
│  │ files in     │  │  │ [system, user, ...]  │   │   │
│  │ resources/   │  │  └─────────────────────┘   │   │
│  └──────────────┘  │  ┌─────────────────────┐   │   │
│                    │  │ Command Handler      │   │   │
│                    │  │ /help /clear /context │   │   │
│                    │  └─────────────────────┘   │   │
│                    └────────────────────────────┘   │
└────────────────────────┬────────────────────────────┘
                         │
          ┌──────────────┼──────────────────┐
          ▼              ▼                  ▼
┌──────────────┐ ┌──────────────┐ ┌────────────────┐
│  streaming.py │ │ context_     │ │ message_       │
│              │ │ manager.py   │ │ formatter.py   │
│ Token filter │ │ Usage check  │ │ Dict builders  │
│ XML stripping│ │ Progress bar │ │                │
└──────┬───────┘ └──────┬───────┘ └────────────────┘
       │                │
       └────────┬───────┘
                ▼
┌─────────────────────────────────────────────────────┐
│            hailo_platform.genai.LLM                  │
│                                                     │
│   generate()        → streaming token iterator      │
│   generate_all()    → full response string          │
│   clear_context()   → reset KV-cache                │
│   get_context_usage_size() / max_context_capacity() │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│              hailo_platform.VDevice                  │
│                                                     │
│   Shared virtual device (group_id = "SHARED")       │
│   DMA buffer management, device lifecycle           │
└────────────────────────┬────────────────────────────┘
                         │ PCIe / HailoRT driver
                         ▼
┌─────────────────────────────────────────────────────┐
│               Hailo-10H Accelerator                  │
│                                                     │
│   Runs compiled HEF model (transformer layers)      │
│   Hardware KV-cache for autoregressive generation   │
└─────────────────────────────────────────────────────┘
```

---

## Troubleshooting

### "No downloaded LLM models found"

No `.hef` files matching LLM patterns were found in `/usr/local/hailo/resources/models/hailo10h/`. Run:

```bash
source setup_env.sh
hailo-download-resources
```

Or download a specific model by running the app with `--hef-path Qwen2.5-1.5B-Instruct` — it will auto-download if the model is in the registry.

### "Failed to resolve HEF path"

The `--hef-path` value could not be matched to a file or known model name. Check:

- Is the file path correct and accessible?
- Is the model name spelled exactly as registered? (case-sensitive, e.g. `Qwen2.5-1.5B-Instruct`)
- Run with `--list-models` on the base `simple_llm_chat.py` to see registered names.

### Device initialization errors

```
Failed to create VDevice
```

- Verify the Hailo-10H is physically connected and powered.
- Check the driver is loaded: `lsmod | grep hailo`
- Check device access permissions: `ls -la /dev/hailo*`
- Ensure no other process has exclusive device access (the `SHARED` group ID should prevent this, but check).

### Context fills up quickly

The Qwen2.5-1.5B-Instruct model has a limited context window. If it fills up frequently:

- Use shorter prompts and ask for concise responses.
- Use `/clear` periodically to reset.
- Lower `--max-tokens` to limit response length.
- The app automatically clears context at 95% capacity and notifies you.

### Garbled or incomplete responses

- Try lowering `--temperature` (e.g. `0.3`) for more predictable output.
- Increase `--max-tokens` if responses seem cut off.
- If you see raw XML tags in output, this may indicate a `StreamingTextFilter` edge case — please report it.
