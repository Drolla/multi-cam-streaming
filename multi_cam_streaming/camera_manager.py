"""Camera discovery and lifecycle management for OpenCV-compatible devices."""
import cv2
import subprocess
import re
import platform
import logging
from typing import List, Dict


class CameraManager:
    """Manage opening and closing of cameras."""
    
    def __init__(self, identifier_list: List):
        """Initialize the camera manager.

        Args:
            identifier_list: List of camera entries — either plain regex pattern strings
                             or dicts with a 'pattern' key plus optional per-camera attributes.
        """
        self.identifier_list = identifier_list
        # Unique opened captures (one per physical device); used for configuration and release.
        self.cameras: List[cv2.VideoCapture] = []
        # One entry per identifier slot, in config order. Slots that resolve to the same
        # physical device share (reference) the same capture object.
        self.frame_sources: List[cv2.VideoCapture] = []
        # OpenCV/v4l2 device index per identifier slot (-1 if unmatched).
        self.video_indexes: List[int] = []

    @staticmethod
    def _get_v4l2_device_indexes() -> Dict[str, int]:
        """Get available cameras from v4l2-ctl (Linux only)."""
        try:
            result = subprocess.run(
                ["v4l2-ctl", "--list-devices"],
                capture_output=True, text=True, check=True
            )
        except FileNotFoundError:
            raise FileNotFoundError("v4l2-ctl not found. Install v4l-utils.")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"v4l2-ctl failed: {e.stderr}")

        devices = {}
        current = None
        for line in result.stdout.splitlines():
            if line and not line.startswith(" ") and line.endswith(":"):
                current = line[:-1]
            elif current and line.strip().startswith("/dev/video"):
                device_index = int(line.strip().replace("/dev/video", ""))
                devices[current] = device_index
                current = None

        logging.debug("Found cameras (Linux): %s", devices)
        return devices

    @staticmethod
    def _get_windows_device_indexes() -> Dict[str, int]:
        """Get available cameras using DirectShow (Windows only)."""
        try:
            from pygrabber.dshow_graph import FilterGraph
        except ImportError:
            raise ImportError("Install pygrabber: pip install pygrabber")

        graph = FilterGraph()
        device_names = graph.get_input_devices()

        # Handle duplicate names
        name_counts = {}
        devices = {}

        for idx, name in enumerate(device_names):
            if name not in name_counts:
                name_counts[name] = 1
                final_name = name
            else:
                name_counts[name] += 1
                final_name = f"{name} #{name_counts[name]}"

            devices[final_name] = idx

        logging.debug("Found cameras (Windows): %s", devices)
        return devices

    def _get_device_indexes(self) -> Dict[str, int]:
        system = platform.system()

        if system == "Linux":
            return self._get_v4l2_device_indexes()
        elif system == "Windows":
            return self._get_windows_device_indexes()
        else:
            raise RuntimeError(f"Unsupported OS: {system}")

    def open(self) -> None:
        """Discover and open cameras matching identifiers.

        Each identifier in ``identifier_list`` becomes a slot in ``frame_sources``.
        An identifier that matches the same physical device as a previous identifier
        reuses the already-open capture, so each device is opened only once while
        still appearing in every slot that selected it.
        """
        all_camera_indexes = self._get_device_indexes()

        # Map a device index to its already-open capture so duplicate selections share it.
        opened: Dict[int, cv2.VideoCapture] = {}

        for entry in self.identifier_list:
            identifier = entry if isinstance(entry, str) else entry['pattern']
            # Support glob-style wildcards: convert bare * to .* so users can
            # write "*Aukey*" as well as the regex equivalent ".*Aukey.*".
            pattern = re.sub(r'(?<!\.)(?<!\[)\*', '.*', identifier)
            match = next(
                ((name, idx) for name, idx in all_camera_indexes.items()
                 if re.search(pattern, name, re.IGNORECASE)),
                None,
            )
            if match is None:
                logging.warning("No camera matched pattern '%s'", identifier)
                self.video_indexes.append(-1)
                continue

            camera_name, camera_index = match
            cap = opened.get(camera_index)
            if cap is None:
                cap = cv2.VideoCapture(camera_index)
                if not cap.isOpened():
                    logging.warning("Camera '%s' (index %d) failed to open", camera_name, camera_index)
                    self.video_indexes.append(-1)
                    continue
                opened[camera_index] = cap
                self.cameras.append(cap)

            self.frame_sources.append(cap)
            self.video_indexes.append(camera_index)

    def close(self) -> None:
        """Close all camera captures."""
        for cap in self.cameras:
            cap.release()
        self.cameras.clear()
        self.frame_sources.clear()
        self.video_indexes.clear()
    
    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self
    
    def __exit__(self, *args):
        """Context manager exit."""
        self.close()
