"""View and stream camera feeds from multiple USB cameras.

Supports three modes:
  - display: Show camera feeds locally
  - stream: Stream to YouTube via RTMP
  - both: Display locally and stream simultaneously

The --list-cameras flag prints the available camera devices and exits, which is
useful for filling in the cameras list in your config file.

Usage:
    python camera_viewer.py                           # Streams to YouTube (default)
    python camera_viewer.py --mode display            # Display only
    python camera_viewer.py --mode both               # Display and stream
    python camera_viewer.py --config custom.yaml      # Use custom config file
    python camera_viewer.py --list-cameras            # List available cameras and exit

Configuration is loaded from a YAML file. See config.example.yaml for format. Cameras are
assigned to layout slots dynamically by motion priority: the most active camera fills
the highest-priority slot (slot 0), and so on.
"""
import sys
from pathlib import Path

# Add parent directory to path so we can import multi_cam_streaming
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import logging
import time
import numpy as np
import argparse
import yaml
from multi_cam_streaming import camera_manager
from multi_cam_streaming import ffmpeg
from multi_cam_streaming.frame_compositor import FrameCompositor

_DEFAULT_WIDTH = 1280
_DEFAULT_HEIGHT = 720
_DEFAULT_FPS = 30
_DEFAULT_MOTION_CHECK_INTERVAL = 1.0
_DEFAULT_MOTION_THRESHOLD = 15
_DEFAULT_MOTION_CHANGE_THRESHOLD = 0.05
_DEFAULT_MIN_SWITCH_INTERVAL = 5.0
_DEFAULT_TRANSITION_DURATION = 0.5


def _parse_camera_entry(entry):
    """Return (pattern_str, attrs_dict) for a camera config entry.

    Accepts either a plain string or a dict with a 'pattern' key.
    """
    if isinstance(entry, str):
        return entry, {'min_slot': 0, 'max_slot': float('inf'), 'activity_multiplier': 1.0}
    pattern = entry['pattern']
    raw_max = entry.get('max_slot')
    attrs = {
        'min_slot': int(entry.get('min_slot', 0)),
        'max_slot': float('inf') if raw_max is None else int(raw_max),
        'activity_multiplier': float(entry.get('activity_multiplier', 1.0)),
    }
    return pattern, attrs


def load_config(config_path):
    """Load and return config from a YAML file, expanding environment variables."""
    import os
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


def run_camera_viewer(config_path, mode="stream", show_motion_debug=False):
    """Run the camera viewer in display, stream, or both modes."""
    config = load_config(config_path)

    camera_patterns, cam_attrs = zip(
        *[_parse_camera_entry(e) for e in config['cameras']]
    ) if config['cameras'] else ([], [])
    camera_identifiers = list(camera_patterns)

    output = config.get('output', {})
    out_w = int(output.get('width', _DEFAULT_WIDTH))
    out_h = int(output.get('height', _DEFAULT_HEIGHT))
    fps = int(output.get('fps', _DEFAULT_FPS))
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

    with camera_manager.CameraManager(camera_identifiers) as cam_mgr:
        if not cam_mgr.cameras:
            print("No cameras found.")
            return

        youtube_stream = None
        if mode in ("stream", "both"):
            youtube_stream = ffmpeg.FFmpegStreamer(youtube_url=youtube_url, fps=fps,
                                                  frame_ims=output_dims)

        motion_cfg = config.get('motion', {})
        compositor = FrameCompositor(
            layouts, output_dims,
            cam_attrs=list(cam_attrs),
            motion_log_interval=float(motion_cfg.get('check_interval', _DEFAULT_MOTION_CHECK_INTERVAL)),
            motion_threshold=int(motion_cfg.get('threshold', _DEFAULT_MOTION_THRESHOLD)),
            motion_change_threshold=float(motion_cfg.get('change_threshold', _DEFAULT_MOTION_CHANGE_THRESHOLD)),
            min_switch_interval=float(motion_cfg.get('min_switch_interval', _DEFAULT_MIN_SWITCH_INTERVAL)),
            transition_duration=float(config.get('transition_duration', _DEFAULT_TRANSITION_DURATION)),
            show_motion_debug=show_motion_debug)

        last_time = time.time()
        try:
            while True:
                frames = read_frames(cam_mgr.frame_sources, output_dims)
                combined = compositor.process(frames)

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
        "--list-cameras",
        action="store_true",
        help="List available camera devices and exit"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        metavar="LEVEL",
        help="Logging level (DEBUG, INFO, WARNING, ERROR). Overrides config file setting."
    )
    parser.add_argument(
        "--show-motion-debug",
        action="store_true",
        help="Show a debug window with the reduced grayscale images used for motion scoring."
    )

    args = parser.parse_args()

    if args.list_cameras:
        list_cameras()
        exit(0)

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file not found: {args.config}")
        exit(1)

    # Resolve log level: CLI > config file > default WARNING
    config = load_config(args.config)
    raw_level = args.log_level or config.get('log_level', 'WARNING')
    numeric_level = getattr(logging, raw_level.upper(), None)
    if not isinstance(numeric_level, int):
        parser.error(f"Invalid log level '{raw_level}'. Choose from DEBUG, INFO, WARNING, ERROR.")
    logging.basicConfig(level=numeric_level,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    run_camera_viewer(args.config, mode=args.mode, show_motion_debug=args.show_motion_debug)


if __name__ == "__main__":
    main()
