import argparse
import threading
from io import StringIO
from contextlib import redirect_stderr

from hailo_platform import VDevice
from hailo_platform.genai import LLM

from hailo_apps.python.core.common.defines import LLM_PROMPT_PREFIX, SHARED_VDEVICE_GROUP_ID, HAILO10H_ARCH, VOICE_ASSISTANT_APP, VOICE_ASSISTANT_MODEL_NAME
from hailo_apps.python.core.common.core import resolve_hef_path
from hailo_apps.python.core.common.hailo_logger import add_logging_cli_args, init_logging, level_from_args
from hailo_apps.python.gen_ai_apps.gen_ai_utils.voice_processing.interaction import VoiceInteractionManager
from hailo_apps.python.gen_ai_apps.gen_ai_utils.voice_processing.vad import add_vad_args
from hailo_apps.python.gen_ai_apps.gen_ai_utils.voice_processing.speech_to_text import SpeechToTextProcessor
from hailo_apps.python.gen_ai_apps.gen_ai_utils.voice_processing.text_to_speech import (
    TextToSpeechProcessor,
    PiperModelNotFoundError,
)
from hailo_apps.python.gen_ai_apps.gen_ai_utils.voice_processing.audio_diagnostics import AudioDiagnostics
from hailo_apps.python.gen_ai_apps.gen_ai_utils.llm_utils import streaming


def list_output_devices():
    """Print all available output devices, highlighting USB ones."""
    _, output_devices = AudioDiagnostics.list_audio_devices()
    print("\nAvailable output devices:")
    print("-" * 70)
    for dev in output_devices:
        usb_marker = " [USB]" if dev.is_usb else ""
        default_marker = " (default)" if dev.is_default else ""
        print(f"  ID {dev.id:3d}{usb_marker}{default_marker}: {dev.name} "
              f"({dev.max_output_channels}ch @ {int(dev.default_samplerate)}Hz)")
    print("-" * 70)


def find_usb_output_device(name_substring: str = None) -> int:
    """
    Find a USB output device.

    Args:
        name_substring: Optional case-insensitive substring to match against device name.

    Returns:
        Device ID of matching USB output, or None if not found.
    """
    _, output_devices = AudioDiagnostics.list_audio_devices()
    usb_outputs = [d for d in output_devices if d.is_usb]

    if not usb_outputs:
        return None

    if name_substring:
        needle = name_substring.lower()
        for dev in usb_outputs:
            if needle in dev.name.lower():
                return dev.id
        return None

    return usb_outputs[0].id


class VoiceAssistantApp:
    """
    Manages the main application logic for the voice assistant.
    Builds the pipeline using common AI components, with TTS routed to a USB speaker.
    """

    def __init__(self, debug=False, no_tts=False, output_device_id=None):
        self.debug = debug
        self.no_tts = no_tts
        self.abort_event = threading.Event()

        print("Initializing AI components... (This might take a moment)")

        # Suppress noisy ALSA messages during initialization
        with redirect_stderr(StringIO()):
            # 1. VDevice
            params = VDevice.create_params()
            params.group_id = SHARED_VDEVICE_GROUP_ID
            self.vdevice = VDevice(params)

            # 2. Speech to Text
            self.s2t = SpeechToTextProcessor(self.vdevice)

            # 3. LLM
            model_path = resolve_hef_path(
                hef_path=VOICE_ASSISTANT_MODEL_NAME,
                app_name=VOICE_ASSISTANT_APP,
                arch=HAILO10H_ARCH
            )
            if model_path is None:
                raise RuntimeError("Failed to resolve HEF path for LLM model. Please ensure the model is available.")

            self.llm = LLM(self.vdevice, str(model_path))

            # 4. TTS — routed to USB speaker
            self.tts = None
            if not no_tts:
                try:
                    self.tts = TextToSpeechProcessor(device_id=output_device_id)
                except PiperModelNotFoundError:
                    self.tts = None

        self.interaction = None

        print("✅ AI components ready!")

    def on_processing_start(self):
        self.on_abort()
        if self.tts:
            self.tts.interrupt()

    def on_abort(self):
        """Abort current generation and speech."""
        self.abort_event.set()
        if self.tts:
            self.tts.interrupt()

    def on_audio_ready(self, audio):
        self.abort_event.clear()

        # 1. Transcribe
        user_text = self.s2t.transcribe(audio)
        if not user_text:
            print("No speech detected.")
            return

        print(f"\nYou: {user_text}")
        print("\nLLM response:\n")

        # 2. Prepare TTS
        current_gen_id = None
        state = {
            'sentence_buffer': "",
            'first_chunk_sent': False
        }

        if self.tts:
            self.tts.clear_interruption()
            current_gen_id = self.tts.get_current_gen_id()

        # 3. Generate Response
        prompt_text = LLM_PROMPT_PREFIX + user_text
        formatted_prompt = [{'role': 'user', 'content': prompt_text}]

        def tts_callback(chunk: str):
            if self.tts:
                state['sentence_buffer'] += chunk
                state['sentence_buffer'] = self.tts.chunk_and_queue(
                    state['sentence_buffer'], current_gen_id, not state['first_chunk_sent']
                )

                if not state['first_chunk_sent'] and not self.tts.speech_queue.empty():
                    state['first_chunk_sent'] = True

        streaming.generate_and_stream_response(
            llm=self.llm,
            prompt=formatted_prompt,
            prefix="",
            show_raw_stream=self.debug,
            token_callback=tts_callback,
            abort_callback=self.abort_event.is_set
        )

        # 4. Send remaining text
        if self.tts and state['sentence_buffer'].strip():
            self.tts.queue_text(state['sentence_buffer'].strip(), current_gen_id)

        print()

        # 5. Handshake: Wait for TTS to finish, then restart listening
        if self.interaction:
            try:
                self.interaction.restart_after_tts()
            except Exception:
                pass

    def on_clear_context(self):
        self.llm.clear_context()
        print("Context cleared.")

    def close(self):
        if self.tts:
            self.tts.stop()

        try:
            self.llm.release()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(
        description='Voice-controlled AI assistant with TTS output routed to a USB speaker.')
    add_logging_cli_args(parser)
    parser.add_argument('--no-tts', action='store_true',
                        help='Disable text-to-speech output for lower resource usage.')
    parser.add_argument('--output-device', type=int, default=None,
                        help='Explicit output device ID (overrides USB auto-detection). '
                             'Use --list-devices to see available IDs.')
    parser.add_argument('--output-device-name', type=str, default=None,
                        help='Substring to match against USB device name (e.g. "Jabra").')
    parser.add_argument('--list-devices', action='store_true',
                        help='List available output devices and exit.')

    add_vad_args(parser)

    args = parser.parse_args()

    init_logging(level=level_from_args(args))

    if args.list_devices:
        list_output_devices()
        return

    debug_mode = getattr(args, 'debug', False)

    if args.no_tts:
        print("TTS disabled: Running in low-resource mode.")
        output_device_id = None
    elif args.output_device is not None:
        output_device_id = args.output_device
        print(f"Using explicit output device ID: {output_device_id}")
    else:
        output_device_id = find_usb_output_device(args.output_device_name)
        if output_device_id is None:
            if args.output_device_name:
                print(f"⚠️  No USB output device matching '{args.output_device_name}' found.")
            else:
                print("⚠️  No USB output device found.")
            print("    Falling back to default output device.")
            print("    Run with --list-devices to see available devices.")
        else:
            _, output_devices = AudioDiagnostics.list_audio_devices()
            dev = next((d for d in output_devices if d.id == output_device_id), None)
            if dev:
                print(f"🔊 Routing TTS to USB speaker: [{dev.id}] {dev.name}")

    app = VoiceAssistantApp(
        debug=debug_mode,
        no_tts=args.no_tts,
        output_device_id=output_device_id,
    )

    interaction = VoiceInteractionManager(
        title="Voice Assistant (USB speaker)",
        on_audio_ready=app.on_audio_ready,
        on_processing_start=app.on_processing_start,
        on_clear_context=app.on_clear_context,
        on_shutdown=app.close,
        on_abort=app.on_abort,
        debug=debug_mode,
        vad_enabled=args.vad,
        vad_aggressiveness=args.vad_aggressiveness,
        vad_energy_threshold=args.vad_energy_threshold,
        tts=app.tts,
    )

    app.interaction = interaction

    interaction.run()


if __name__ == "__main__":
    main()
