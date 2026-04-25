import argparse
import sys

import numpy as np

from hailo_platform import VDevice
from hailo_platform.genai import Speech2Text, Speech2TextTask

from hailo_apps.python.core.common.core import handle_list_models_flag, resolve_hef_path
from hailo_apps.python.core.common.defines import (
    WHISPER_CHAT_APP,
    SHARED_VDEVICE_GROUP_ID,
    HAILO10H_ARCH,
)
from hailo_apps.python.core.common.hailo_logger import get_logger
from hailo_apps.python.gen_ai_apps.gen_ai_utils.voice_processing.audio_recorder import (
    AudioRecorder,
)

logger = get_logger(__name__)


def main():
    """Record voice from USB microphone and transcribe with Whisper."""
    parser = argparse.ArgumentParser(
        description="Record from USB microphone and transcribe with Whisper"
    )
    parser.add_argument("--hef-path", type=str, default=None, help="Path to HEF model file")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    parser.add_argument(
        "--device", type=int, default=None, help="Audio input device ID (default: auto-detect)"
    )

    handle_list_models_flag(parser, WHISPER_CHAT_APP)
    args = parser.parse_args()

    # Resolve HEF path
    hef_path = resolve_hef_path(args.hef_path, app_name=WHISPER_CHAT_APP, arch=HAILO10H_ARCH)
    if hef_path is None:
        logger.error("Failed to resolve HEF path for Whisper model.")
        sys.exit(1)

    print(f"Model file: {hef_path}")

    vdevice = None
    speech2text = None
    recorder = None

    try:
        # Initialize Hailo device
        print("\n[1/3] Initializing Hailo device...")
        params = VDevice.create_params()
        params.group_id = SHARED_VDEVICE_GROUP_ID
        vdevice = VDevice(params)
        print("Hailo device initialized")

        # Load Whisper model
        print("[2/3] Loading Whisper model...")
        speech2text = Speech2Text(vdevice, str(hef_path))
        print("Model loaded")

        # Initialize microphone
        print("[3/3] Initializing microphone...")
        recorder = AudioRecorder(device_id=args.device)
        print("Microphone ready")

        print("\n" + "=" * 60)
        print("  Voice Whisper - Press Enter to record, Enter to stop")
        print("  Ctrl+C to quit")
        print("=" * 60)

        while True:
            input("\nPress Enter to start recording...")
            recorder.start()
            print("Recording... (press Enter to stop)")

            input()
            audio_data = recorder.stop()

            if audio_data.size == 0:
                print("No audio captured.")
                continue

            duration = audio_data.size / 16000.0
            print(f"Recorded {duration:.1f}s of audio. Transcribing...")

            segments = speech2text.generate_all_segments(
                audio_data=audio_data,
                task=Speech2TextTask.TRANSCRIBE,
                language="en",
                timeout_ms=15000,
            )

            if segments and len(segments) > 0:
                transcription = "".join([seg.text for seg in segments])
                print("-" * 60)
                print(transcription.strip())
                print("-" * 60)
            else:
                print("No transcription generated.")

    except KeyboardInterrupt:
        print("\nExiting...")

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)

    finally:
        if recorder:
            recorder.close()
        if speech2text:
            try:
                speech2text.release()
            except Exception as e:
                logger.warning(f"Error releasing Speech2Text: {e}")
        if vdevice:
            try:
                vdevice.release()
            except Exception as e:
                logger.warning(f"Error releasing VDevice: {e}")


if __name__ == "__main__":
    main()
