import argparse
import random
import sys
import time
from datetime import datetime

import cv2
import numpy as np

from hailo_platform import VDevice
from hailo_platform.genai import VLM

from hailo_apps.python.core.common.core import handle_list_models_flag, resolve_hef_path
from hailo_apps.python.core.common.defines import (
    HAILO10H_ARCH,
    SHARED_VDEVICE_GROUP_ID,
    VLM_CHAT_APP,
)
from hailo_apps.python.core.common.hailo_logger import get_logger
from hailo_apps.python.core.common.toolbox import open_usb_camera

logger = get_logger(__name__)

SYSTEM_PROMPT = (
    "You are a witty, friendly observer commenting on a live camera feed. "
    "Keep every reply under 60 words total. Be playful and concise."
)

USER_PROMPT_DEFAULT = (
    "Describe what you see in this image in one short sentence, then tell me a "
    "short, family-friendly joke loosely related to it.\n"
    "Format:\n"
    "Scene: <one sentence>\n"
    "Joke: <one or two lines>"
)

USER_PROMPT_COMPLIMENT = (
    "Describe what you see in this image in one short sentence, then give a warm, "
    "sincere compliment to whoever or whatever is in the frame, and finish with a "
    "short joke related to the scene.\n"
    "Format:\n"
    "Scene: <one sentence>\n"
    "Compliment: <one sentence>\n"
    "Joke: <one or two lines>"
)

COMPLIMENT_PROBABILITY = 0.25
VLM_INPUT_SIZE = 336
PREVIEW_WINDOW = "Scene Commentator"


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (VLM_INPUT_SIZE, VLM_INPUT_SIZE),
                      interpolation=cv2.INTER_LINEAR).astype(np.uint8)


def build_prompt(include_compliment: bool) -> list[dict]:
    user_text = USER_PROMPT_COMPLIMENT if include_compliment else USER_PROMPT_DEFAULT
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": user_text},
        ]},
    ]


def clean_response(raw: str) -> str:
    return raw.split(". [{'type'")[0].split("<|im_end|>")[0].strip()


def main():
    parser = argparse.ArgumentParser(description="Live USB camera + Hailo VLM scene commentator")
    parser.add_argument("--hef-path", type=str, default=None, help="Path to VLM HEF file")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    parser.add_argument("--interval", type=float, default=10.0,
                        help="Seconds between snapshots sent to the VLM (default: 10)")
    parser.add_argument("--no-preview", action="store_true",
                        help="Run headless — do not open a preview window")

    handle_list_models_flag(parser, VLM_CHAT_APP)
    args = parser.parse_args()

    hef_path = resolve_hef_path(args.hef_path, app_name=VLM_CHAT_APP, arch=HAILO10H_ARCH)
    if hef_path is None:
        logger.error("Failed to resolve HEF path for VLM model.")
        sys.exit(1)

    logger.info(f"Using HEF: {hef_path}")
    print(f"✓ Model file found: {hef_path}")

    vdevice = None
    vlm = None
    cap = None

    try:
        print("\n[1/3] Initializing Hailo device...")
        params = VDevice.create_params()
        params.group_id = SHARED_VDEVICE_GROUP_ID
        vdevice = VDevice(params)
        print("✓ Hailo device initialized")

        print("[2/3] Loading VLM model (this can take a moment)...")
        vlm = VLM(vdevice, str(hef_path))
        print("✓ Model loaded")

        print("[3/3] Opening USB camera...")
        cap = open_usb_camera("sd")
        print("✓ Camera open")

        show_preview = not args.no_preview
        if show_preview:
            print(f"\nLive preview running. Snapshot every {args.interval:.1f}s. "
                  "Press 'q' in the preview window (or Ctrl+C) to quit.\n")
        else:
            print(f"\nHeadless mode. Snapshot every {args.interval:.1f}s. "
                  "Press Ctrl+C to quit.\n")

        iteration = 0
        next_snapshot_at = time.monotonic()

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                logger.warning("Failed to read frame from camera; retrying...")
                time.sleep(0.05)
                continue

            if show_preview:
                cv2.imshow(PREVIEW_WINDOW, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\n'q' pressed — exiting.")
                    break

            now = time.monotonic()
            if now < next_snapshot_at:
                if not show_preview:
                    time.sleep(min(0.1, next_snapshot_at - now))
                continue

            next_snapshot_at = now + args.interval
            iteration += 1
            include_compliment = random.random() < COMPLIMENT_PROBABILITY

            image = preprocess_frame(frame)
            prompt = build_prompt(include_compliment)

            timestamp = datetime.now().strftime("%H:%M:%S")
            mode_tag = "scene+joke+compliment" if include_compliment else "scene+joke"
            print(f"\n[#{iteration} {timestamp}] ({mode_tag}) thinking...")

            try:
                raw = vlm.generate_all(
                    prompt=prompt,
                    frames=[image],
                    temperature=0.7,
                    seed=random.randrange(2**31),
                    max_generated_tokens=200,
                )
            except Exception as e:
                logger.error(f"VLM generation failed on iteration {iteration}: {e}")
                continue
            finally:
                try:
                    vlm.clear_context()
                except Exception as e:
                    logger.warning(f"clear_context failed: {e}")

            print("-" * 60)
            print(clean_response(raw))
            print("-" * 60)

    except KeyboardInterrupt:
        print("\nInterrupted — exiting.")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception as e:
                logger.warning(f"Error releasing camera: {e}")
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        if vlm is not None:
            try:
                vlm.clear_context()
                vlm.release()
            except Exception as e:
                logger.warning(f"Error releasing VLM: {e}")
        if vdevice is not None:
            try:
                vdevice.release()
            except Exception as e:
                logger.warning(f"Error releasing VDevice: {e}")


if __name__ == "__main__":
    main()
