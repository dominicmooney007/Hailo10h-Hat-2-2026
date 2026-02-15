import argparse
import sys
from pathlib import Path

from hailo_platform import VDevice
from hailo_platform.genai import LLM

from hailo_apps.config.config_manager import get_model_names
from hailo_apps.python.core.common.defines import (
    HAILO10H_ARCH,
    HAILO_FILE_EXTENSION,
    LLM_CHAT_APP,
    RESOURCES_ROOT_PATH_DEFAULT,
    SHARED_VDEVICE_GROUP_ID,
)
from hailo_apps.python.core.common.hailo_logger import get_logger
from hailo_apps.python.gen_ai_apps.gen_ai_utils.llm_utils.context_manager import (
    is_context_full,
    print_context_usage,
)
from hailo_apps.python.gen_ai_apps.gen_ai_utils.llm_utils.message_formatter import (
    messages_assistant,
    messages_system,
    messages_user,
)
from hailo_apps.python.gen_ai_apps.gen_ai_utils.llm_utils.streaming import (
    clean_response,
    generate_and_stream_response,
)

logger = get_logger(__name__)

SYSTEM_PROMPT = "You are a helpful assistant. Be concise and clear in your responses."

# Known LLM-incompatible model prefixes (VLM, speech, vision-only)
EXCLUDE_PATTERNS = ("Whisper", "clip_vit", "Qwen2-VL")


def find_downloaded_llm_models() -> list[Path]:
    """Find HEF files in the hailo10h models directory that are likely LLM-compatible."""
    models_dir = Path(RESOURCES_ROOT_PATH_DEFAULT) / "models" / HAILO10H_ARCH
    if not models_dir.exists():
        return []

    # Get known LLM model names from config
    known_llm_names = set()
    for app in (LLM_CHAT_APP, "agent"):
        try:
            names = get_model_names(app, HAILO10H_ARCH, tier="all")
            known_llm_names.update(names)
        except Exception:
            pass

    candidates = []
    for hef_file in sorted(models_dir.glob(f"*{HAILO_FILE_EXTENSION}")):
        model_name = hef_file.stem
        # Include if it's a known LLM model from config
        if model_name in known_llm_names:
            candidates.append(hef_file)
            continue
        # Also include models matching LLM naming patterns (e.g. "Instruct" suffix)
        # but exclude known non-LLM patterns
        if any(model_name.startswith(p) for p in EXCLUDE_PATTERNS):
            continue
        if "Instruct" in model_name:
            candidates.append(hef_file)

    return candidates


def select_model(models: list[Path]) -> Path:
    """Display a numbered menu and return the user's choice."""
    print("\nAvailable LLM models:")
    print("-" * 50)
    for i, model_path in enumerate(models, 1):
        print(f"  [{i}] {model_path.stem}")
    print("-" * 50)

    while True:
        try:
            choice = input(f"Select model (1-{len(models)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                return models[idx]
            print(f"  Please enter a number between 1 and {len(models)}")
        except ValueError:
            print(f"  Please enter a number between 1 and {len(models)}")
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)


def print_help():
    """Print available commands."""
    print("\nCommands:")
    print("  /help     - Show this help message")
    print("  /clear    - Clear conversation history and context")
    print("  /context  - Show context usage stats")
    print("  /quit     - Exit the chat")
    print()


def main():
    parser = argparse.ArgumentParser(description="Interactive LLM Chat")
    parser.add_argument("--hef-path", type=str, default=None, help="Path to HEF model file")
    parser.add_argument(
        "--max-tokens", type=int, default=512, help="Max tokens per response (default: 512)"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature (default: 0.7)"
    )
    parser.add_argument(
        "--system-prompt", type=str, default=None, help="Custom system prompt"
    )
    args = parser.parse_args()

    # Resolve model path
    if args.hef_path:
        from hailo_apps.python.core.common.core import resolve_hef_path

        hef_path = resolve_hef_path(args.hef_path, app_name=LLM_CHAT_APP, arch=HAILO10H_ARCH)
        if hef_path is None:
            logger.error("Failed to resolve HEF path.")
            sys.exit(1)
    else:
        models = find_downloaded_llm_models()
        if not models:
            print("No downloaded LLM models found in "
                  f"{Path(RESOURCES_ROOT_PATH_DEFAULT) / 'models' / HAILO10H_ARCH}")
            print("Run 'hailo-download-resources' first, or specify --hef-path.")
            sys.exit(1)
        if len(models) == 1:
            hef_path = models[0]
            print(f"\nUsing model: {hef_path.stem}")
        else:
            hef_path = select_model(models)

    system_prompt = args.system_prompt or SYSTEM_PROMPT

    vdevice = None
    llm = None

    try:
        print(f"\nLoading {hef_path.stem}...")
        params = VDevice.create_params()
        params.group_id = SHARED_VDEVICE_GROUP_ID
        vdevice = VDevice(params)
        llm = LLM(vdevice, str(hef_path))
        print(f"Model loaded. Type /help for commands, /quit to exit.\n")

        conversation = [messages_system(system_prompt)]

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break

            if not user_input:
                continue

            # Handle commands
            cmd = user_input.lower()
            if cmd in ("/quit", "/exit"):
                print("Goodbye!")
                break
            if cmd == "/help":
                print_help()
                continue
            if cmd == "/clear":
                llm.clear_context()
                conversation = [messages_system(system_prompt)]
                print("[Context cleared]\n")
                continue
            if cmd == "/context":
                print_context_usage(llm, show_always=True)
                print()
                continue

            # Add user message to history
            conversation.append(messages_user(user_input))

            # Stream response
            raw_response = generate_and_stream_response(
                llm,
                prompt=conversation,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                prefix="Assistant: ",
                show_raw_stream=False,
            )

            # Clean and store assistant response
            cleaned = clean_response(raw_response)
            conversation.append(messages_assistant(cleaned))

            # Check context usage
            if is_context_full(llm):
                print("[Context full - clearing history to continue]")
                llm.clear_context()
                conversation = [messages_system(system_prompt)]

            print()  # blank line between turns

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)

    finally:
        if llm:
            try:
                llm.clear_context()
                llm.release()
            except Exception as e:
                logger.warning(f"Error releasing LLM: {e}")
        if vdevice:
            try:
                vdevice.release()
            except Exception as e:
                logger.warning(f"Error releasing VDevice: {e}")


if __name__ == "__main__":
    main()
