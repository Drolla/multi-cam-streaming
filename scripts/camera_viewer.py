"""View and stream camera feeds from multiple USB cameras.

Supports three modes:
  - display: Show camera feeds locally
  - stream: Stream to YouTube via RTMP
  - both: Display locally and stream simultaneously

The --list-devices flag prints the available camera and audio devices and exits.
The --print-device-config flag prints a ready-to-edit 'cameras:' YAML block for
detected cameras, which is useful for filling in the cameras list in your config file.

Usage:
    python camera_viewer.py                           # Streams to YouTube (default)
    python camera_viewer.py --mode display            # Display only
    python camera_viewer.py --mode both               # Display and stream
    python camera_viewer.py --config custom.yaml      # Use custom config file
    python camera_viewer.py --list-devices             # List available cameras and audio devices, exit
    python camera_viewer.py --print-device-config      # Print a cameras: YAML block and exit

Configuration is loaded from a YAML file. See config.example.yaml for format. Cameras are
assigned to layout slots dynamically by motion priority: the most active camera fills
the highest-priority slot (slot 0), and so on.
"""
import argparse
import contextlib
import logging
import os
import platform
import sys
import time
from pathlib import Path

# Add parent directory to path so we can import multi_cam_streaming
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import sounddevice as sd
import yaml

from multi_cam_streaming import camera_manager
from multi_cam_streaming import ffmpeg
from multi_cam_streaming.audio_manager import (
    AudioManager, SAMPLE_RATE as _AUDIO_SAMPLE_RATE,
    _sd_input_devices, _match_linux_usb,
)
from multi_cam_streaming.audio_mixer import AudioMixer
from multi_cam_streaming.frame_compositor import FrameCompositor

_DEFAULT_WIDTH = 1280
_DEFAULT_HEIGHT = 720
_DEFAULT_FPS = 30
_DEFAULT_MOTION_CHECK_INTERVAL = 1.0
_DEFAULT_MOTION_THRESHOLD = 15
_DEFAULT_MOTION_CHANGE_THRESHOLD = 0.05
_DEFAULT_MOTION_NORMALIZATION = 0.7
_DEFAULT_MIN_SWITCH_INTERVAL = 5.0
_DEFAULT_TRANSITION_DURATION = 0.5
_LOGGABLE_MODULES = ("camera_manager", "audio_manager", "audio_mixer", "ffmpeg", "frame_compositor")


def _parse_camera_entry(entry):
    """Return (pattern_str, attrs_dict) for a camera config entry.

    Accepts either a plain string or a dict with a 'pattern' key.
    """
    if isinstance(entry, str):
        return entry, {'min_slot': 0, 'max_slot': float('inf'), 'activity_multiplier': 1.0,
                       'mic': None}
    pattern = entry['pattern']
    raw_max = entry.get('max_slot')
    attrs = {
        'min_slot': int(entry.get('min_slot', 0)),
        'max_slot': float('inf') if raw_max is None else int(raw_max),
        'activity_multiplier': float(entry.get('activity_multiplier', 1.0)),
        'mic': entry.get('mic'),
    }
    return pattern, attrs


def load_config(config_path):
    """Load and return config from a YAML file, expanding environment variables."""
    with open(config_path, 'r') as f:
        content = f.read()
    content = os.path.expandvars(content)
    return yaml.safe_load(content)


def read_frames(frame_sources, output_dims):
    """Read one frame per camera in ``frame_sources``.

    A physical camera referenced more than once is read only once; its frame is
    reused for each duplicate entry. Returns a plain list of BGR frames, one per
    entry in frame_sources, indexed by position.
    """
    cache = {}  # id(cap) -> frame
    result = []
    empty = np.zeros((output_dims[1], output_dims[0], 3), dtype=np.uint8)
    for cap in frame_sources:
        key = id(cap)
        if key not in cache:
            ret, frame = cap.read()
            cache[key] = frame if ret else empty.copy()
        result.append(cache[key])
    return result


def list_cameras():
    """Print all available camera devices and their indexes."""
    cm = camera_manager.CameraManager(identifier_list=[])
    devices = cm._get_device_indexes()
    print("Available camera devices:")
    for name, index in devices.items():
        print(f"  - {name}: index {index}")


def list_audio_devices():
    """Print all available audio input and output devices."""
    devices = sd.query_devices()
    default_in = sd.default.device[0]
    default_out = sd.default.device[1]
    host_apis = sd.query_hostapis()

    print("Available audio input devices (use index or name substring as 'mic:' in camera config):")
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] >= 1:
            api = host_apis[dev['hostapi']]['name']
            marker = " *default*" if i == default_in else ""
            print(f"  [{i}] {dev['name']}  ({api}){marker}")
    print()
    print("Available audio output devices (use index or name substring as 'audio: output:' or --audio-output):")
    for i, dev in enumerate(devices):
        if dev['max_output_channels'] >= 1:
            api = host_apis[dev['hostapi']]['name']
            marker = " *default*" if i == default_out else ""
            print(f"  [{i}] {dev['name']}  ({api}){marker}")


def print_device_config():
    """Print a ready-to-edit 'cameras:' YAML block, one entry per detected camera.

    Reuses AudioManager's own USB-topology matching (_match_linux_usb) so the
    generated 'mic:' value reflects exactly what AudioManager.open() would pick
    at runtime. Only Linux exposes a reliable hardware match; other platforms
    get a commented placeholder instead of a guessed value. Every optional
    attribute is included as a commented placeholder so the full set of knobs
    is visible without consulting docs.
    """
    cm = camera_manager.CameraManager(identifier_list=[])
    cam_devices = cm._get_device_indexes()
    if not cam_devices:
        print("No camera devices found.")
        return

    input_devices = _sd_input_devices()
    system = platform.system()

    print("cameras:")
    for name, index in cam_devices.items():
        match = _match_linux_usb(index, input_devices) if system == "Linux" else None
        print(f'  - pattern: "{name}"')
        if match is not None:
            match_name = sd.query_devices(match)['name']
            print(f'    mic: "{match_name}"')
        else:
            reason = ("no matching USB audio device found" if system == "Linux"
                       else "no reliable auto-match on this OS - assign via 'mic:' substring or index")
            print(f"    # mic: <>  # {reason}")
        print("    # min_slot: 0")
        print("    # max_slot: <none>")
        print("    # activity_multiplier: 1.0")


def run_camera_viewer(config_path, mode="stream", show_motion_debug=False,
                      audio_output=None):
    """Run the camera viewer in display, stream, or both modes."""
    config = load_config(config_path)

    camera_patterns, cam_attrs = zip(
        *[_parse_camera_entry(e) for e in config['cameras']]
    ) if config['cameras'] else ([], [])
    camera_identifiers = list(camera_patterns)

    video = config.get('video', {})
    out_w = int(video.get('width', _DEFAULT_WIDTH))
    out_h = int(video.get('height', _DEFAULT_HEIGHT))
    fps = int(video.get('fps', _DEFAULT_FPS))
    output_dims = (out_w, out_h)

    layouts = config.get('layouts', [])
    if not layouts:
        print("Error: no layouts defined in config.")
        return

    if mode in ("stream", "both"):
        youtube_stream_key = config['youtube'].get('stream_key', '')
        youtube_rtmp_url = config['youtube'].get('rtmp_url', '')
        if not youtube_stream_key or not youtube_rtmp_url:
            print("Error: youtube.stream_key and youtube.rtmp_url are required for streaming mode.")
            return
        youtube_url = youtube_rtmp_url + youtube_stream_key

    audio_cfg = config.get('audio', {})
    audio_enabled = bool(audio_cfg.get('enabled', False))
    # CLI --audio-output overrides yaml audio.output; None means no local playback
    effective_audio_output = audio_output if audio_output is not None else audio_cfg.get('output')
    audio_size_threshold = float(audio_cfg.get('size_threshold', 0.0))
    audio_transition_duration = float(audio_cfg.get('transition_duration', 0.0))
    audio_compression = audio_cfg.get('compression')

    with camera_manager.CameraManager(camera_identifiers) as cam_mgr:
        if not cam_mgr.cameras:
            print("No cameras found.")
            return

        raw_entries = config.get('cameras', [])
        audio_mgr_ctx = (
            AudioManager(raw_entries, cam_mgr.video_indexes)
            if audio_enabled else contextlib.nullcontext()
        )

        with audio_mgr_ctx as audio_mgr:
            audio_mixer_ctx = (
                AudioMixer(audio_mgr, pipe_needed=mode in ("stream", "both"),
                           output_device=effective_audio_output,
                           transition_duration=audio_transition_duration,
                           compression=audio_compression,
                           size_threshold=audio_size_threshold)
                if audio_enabled else contextlib.nullcontext()
            )

            with audio_mixer_ctx as audio_mixer:
                audio_pipe_fd = audio_mixer.audio_pipe_fd if audio_mixer else None
                audio_sample_rate = audio_mixer.audio_sample_rate if audio_mixer else _AUDIO_SAMPLE_RATE

                youtube_stream = None
                if mode in ("stream", "both"):
                    youtube_stream = ffmpeg.FFmpegStreamer(youtube_url=youtube_url, fps=fps,
                                                          frame_dims=output_dims,
                                                          audio_pipe_fd=audio_pipe_fd,
                                                          audio_sample_rate=audio_sample_rate)
                if audio_mixer is not None:
                    audio_mixer.signal_ready()

                motion_cfg = config.get('motion', {})
                compositor = FrameCompositor(
                    layouts, output_dims,
                    cam_attrs=list(cam_attrs),
                    motion_log_interval=float(motion_cfg.get('check_interval', _DEFAULT_MOTION_CHECK_INTERVAL)),
                    motion_threshold=int(motion_cfg.get('threshold', _DEFAULT_MOTION_THRESHOLD)),
                    motion_change_threshold=float(motion_cfg.get('change_threshold', _DEFAULT_MOTION_CHANGE_THRESHOLD)),
                    motion_normalization=float(motion_cfg.get('normalization', _DEFAULT_MOTION_NORMALIZATION)),
                    min_switch_interval=float(motion_cfg.get('min_switch_interval', _DEFAULT_MIN_SWITCH_INTERVAL)),
                    transition_duration=float(config.get('transition_duration', _DEFAULT_TRANSITION_DURATION)),
                    show_motion_debug=show_motion_debug)

                last_time = time.time()
                try:
                    while True:
                        frames = read_frames(cam_mgr.frame_sources, output_dims)
                        combined = compositor.process(frames)

                        if audio_mixer is not None and compositor.arrangement_changed:
                            sizes = compositor.target_sizes
                            if sizes:
                                audio_mixer.set_weights(sizes)

                        if mode in ("display", "both"):
                            cv2.imshow("multi_cam_streaming", combined)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                print("'q' pressed, exiting...")
                                break

                        if mode in ("stream", "both") and youtube_stream is not None:
                            youtube_stream.write_frame(combined)

                        now = time.time()
                        delay = last_time + 1.0 / fps - now
                        if delay > 0:
                            time.sleep(delay)
                            last_time += 1.0 / fps
                        else:
                            last_time = now

                except KeyboardInterrupt:
                    print("\nCtrl-C pressed, shutting down...")
                finally:
                    if youtube_stream is not None:
                        youtube_stream.cleanup()
                    if mode in ("display", "both"):
                        try:
                            cv2.destroyAllWindows()
                        except Exception:
                            pass
                    print("Resources released")


def main():
    parser = argparse.ArgumentParser(description="View and stream camera feeds")
    parser.add_argument(
        "--config",
        type=str,
        default="config_rpi.yaml",
        help="Path to YAML configuration file (default: config_rpi.yaml)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["display", "stream", "both"],
        default="stream",
        help="Mode to run: display (local only), stream (YouTube only), or both"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        metavar="LEVEL",
        help="Default logging level (DEBUG, INFO, WARNING, ERROR). Overrides log_levels.default in "
             "the config file."
    )
    for module_name in _LOGGABLE_MODULES:
        flag = f"--log-level-{module_name.replace('_', '-')}"
        parser.add_argument(
            flag,
            type=str,
            default=None,
            metavar="LEVEL",
            dest=f"log_level_{module_name}",
            help=f"Logging level (DEBUG, INFO, WARNING, ERROR) for the '{module_name}' module. "
                 f"Overrides log_levels.{module_name} in the config file."
        )
    parser.add_argument(
        "--show-motion-debug",
        action="store_true",
        help="Show a debug window with the reduced grayscale images used for motion scoring."
    )
    parser.add_argument(
        "--audio-output",
        type=str,
        default=None,
        metavar="DEVICE",
        help="Play mixed audio to the named output device (substring match). "
             "Overrides audio.output in config. Requires audio.enabled: true."
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available camera devices and audio input (mic) / output (speaker) devices, then exit."
    )
    parser.add_argument(
        "--print-device-config",
        action="store_true",
        help="Print a ready-to-edit 'cameras:' YAML block for detected cameras, with auto-matched "
             "mics filled in where possible, and exit."
    )

    args = parser.parse_args()

    if args.list_devices:
        list_cameras()
        print()
        list_audio_devices()
        exit(0)

    if args.print_device_config:
        print_device_config()
        exit(0)

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file not found: {args.config}")
        exit(1)

    # Resolve log level: CLI > config file > default WARNING
    config = load_config(args.config)
    log_levels = config.get('log_levels', {})
    raw_level = args.log_level or log_levels.get('default', 'WARNING')
    numeric_level = getattr(logging, raw_level.upper(), None)
    if not isinstance(numeric_level, int):
        parser.error(f"Invalid log level '{raw_level}'. Choose from DEBUG, INFO, WARNING, ERROR.")
    logging.basicConfig(level=numeric_level,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    # Per-module log level overrides: CLI > config file
    for module_name in _LOGGABLE_MODULES:
        raw_module_level = getattr(args, f"log_level_{module_name}") or log_levels.get(module_name)
        if raw_module_level is None:
            continue
        module_level = getattr(logging, str(raw_module_level).upper(), None)
        if not isinstance(module_level, int):
            parser.error(f"Invalid log level '{raw_module_level}' for module '{module_name}'. "
                         f"Choose from DEBUG, INFO, WARNING, ERROR.")
        logging.getLogger(f"multi_cam_streaming.{module_name}").setLevel(module_level)

    run_camera_viewer(args.config, mode=args.mode, show_motion_debug=args.show_motion_debug,
                      audio_output=args.audio_output)


if __name__ == "__main__":
    main()
