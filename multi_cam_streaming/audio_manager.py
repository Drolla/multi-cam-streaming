"""Audio device discovery and microphone stream lifecycle management.

Discovers the sounddevice input device for each camera slot using a
three-tier priority (explicit config pattern → Linux pyudev USB matching →
Windows name matching), opens one InputStream per unique physical device,
and exposes the resulting PCM block queues to the AudioMixer.
"""
import logging
import platform
import queue

import sounddevice as sd

log = logging.getLogger(__name__)

_SAMPLE_RATE = 48000  # default; overridden per-device by querying device's native rate
_CHANNELS = 1         # mono
_DTYPE = 'int16'
_BLOCK_SIZE = 1024    # samples per callback block


def _sd_input_devices() -> list[tuple[int, str]]:
    """Return (sounddevice_index, name) for all input-capable devices."""
    return [
        (i, dev['name'])
        for i, dev in enumerate(sd.query_devices())
        if dev['max_input_channels'] >= 1
    ]


def _match_by_name(pattern: str, input_devices: list[tuple[int, str]]) -> int | None:
    """Return sounddevice index for the first device whose name contains pattern.

    If pattern is a plain integer string it is treated as a direct device index.
    Matching is case-insensitive substring; glob wildcards (* .*) are stripped.
    """
    if pattern.strip().isdigit():
        idx = int(pattern.strip())
        return idx if any(i == idx for i, _ in input_devices) else None
    plain = pattern.replace('.*', '').replace('*', '').strip()
    for sd_idx, name in input_devices:
        if plain.lower() in name.lower():
            return sd_idx
    return None


def _match_linux_usb(video_index: int, input_devices: list[tuple[int, str]]) -> int | None:
    """Return sounddevice index for the ALSA card sharing the same USB parent as /dev/videoN."""
    try:
        import pyudev
    except ImportError:
        log.debug("pyudev not installed; skipping USB audio matching")
        return None

    context = pyudev.Context()
    video_node = next(
        (dev for dev in context.list_devices(subsystem='video4linux')
         if dev.device_node == f'/dev/video{video_index}'),
        None,
    )
    if video_node is None:
        return None
    usb_parent = video_node.find_parent('usb', 'usb_device')
    if usb_parent is None:
        return None  # not a USB camera (e.g. CSI)

    alsa_card_to_sd: dict[int, int] = {}
    for sd_idx, name in input_devices:
        for card_num in range(32):
            if f'hw:{card_num},' in name or f'card{card_num}' in name.lower():
                alsa_card_to_sd[card_num] = sd_idx
                break

    for audio_dev in context.list_devices(subsystem='sound'):
        if audio_dev.find_parent('usb', 'usb_device') != usb_parent:
            continue
        sys_name = audio_dev.sys_name
        if sys_name.startswith('card'):
            try:
                card_num = int(sys_name[4:])
            except ValueError:
                continue
            if card_num in alsa_card_to_sd:
                return alsa_card_to_sd[card_num]

    return None


class AudioManager:
    """Discover and open microphone streams for a set of camera slots.

    One InputStream is opened per unique physical device — cameras that share
    the same sounddevice index share a single queue, avoiding contention.

    Usage::

        with AudioManager(camera_entries, video_indexes) as mgr:
            mixer = AudioMixer(mgr, ...)
    """

    def __init__(self, camera_entries: list, video_indexes: list[int]):
        """
        Args:
            camera_entries: Raw camera config entries (str or dict with optional 'mic' key).
            video_indexes:  OpenCV device indexes in the same order as camera_entries.
                            Use -1 for cameras with no numeric video index.
        """
        self._camera_entries = camera_entries
        self._video_indexes = video_indexes
        self._streams: list[sd.InputStream] = []

        # Populated by open():
        self.cam_to_sd: dict[int, int] = {}           # cam_pos → sounddevice index
        self.buffers: dict[int, queue.Queue] = {}      # cam_pos → PCM block queue
        self.sample_rates: dict[int, int] = {}         # cam_pos → sample rate (Hz)
        self.sample_rate: int = _SAMPLE_RATE           # representative rate (first opened)

    def open(self) -> None:
        """Discover mic devices and open one InputStream per unique physical device."""
        input_devices = _sd_input_devices()
        system = platform.system()
        warned: set[int] = set()

        for cam_pos, entry in enumerate(self._camera_entries):
            mic_pattern = entry.get('mic') if isinstance(entry, dict) else None
            video_idx = (self._video_indexes[cam_pos]
                         if cam_pos < len(self._video_indexes) else -1)
            sd_idx = None

            # Priority 1: explicit mic pattern from config
            if mic_pattern:
                sd_idx = _match_by_name(mic_pattern, input_devices)
                if sd_idx is None:
                    log.warning("No audio device matched mic pattern '%s' for camera %d",
                                mic_pattern, cam_pos)

            # Priority 2: Linux USB parent matching
            if sd_idx is None and system == 'Linux' and video_idx >= 0:
                sd_idx = _match_linux_usb(video_idx, input_devices)

            # Priority 3: Windows name matching against camera pattern
            if sd_idx is None and system == 'Windows':
                pattern = entry if isinstance(entry, str) else entry.get('pattern', '')
                sd_idx = _match_by_name(pattern, input_devices)

            if sd_idx is None:
                if cam_pos not in warned:
                    log.warning("No audio device found for camera %d; it will contribute no audio",
                                cam_pos)
                    warned.add(cam_pos)
                continue

            self.cam_to_sd[cam_pos] = sd_idx
            log.info("Camera %d → audio device %d ('%s')",
                     cam_pos, sd_idx, sd.query_devices(sd_idx)['name'])

        if not self.cam_to_sd:
            log.warning("No camera mics matched; audio will be silent")
            return

        # One queue per unique sounddevice index
        sd_idx_to_queue: dict[int, queue.Queue] = {}
        for sd_idx in self.cam_to_sd.values():
            if sd_idx not in sd_idx_to_queue:
                sd_idx_to_queue[sd_idx] = queue.Queue(maxsize=8)

        self.buffers = {
            cam_pos: sd_idx_to_queue[sd_idx]
            for cam_pos, sd_idx in self.cam_to_sd.items()
        }

        for sd_idx, buf in sd_idx_to_queue.items():
            dev_info = sd.query_devices(sd_idx)
            rate = int(dev_info['default_samplerate'])
            for cam_pos, mapped_idx in self.cam_to_sd.items():
                if mapped_idx == sd_idx:
                    self.sample_rates[cam_pos] = rate

            def _callback(indata, frames, time_info, status, _buf=buf):  # noqa: ARG001
                if status:
                    log.debug("Audio input status: %s", status)
                try:
                    _buf.put_nowait(indata.copy())
                except queue.Full:
                    pass

            try:
                stream = sd.InputStream(
                    device=sd_idx,
                    channels=_CHANNELS,
                    samplerate=rate,
                    dtype=_DTYPE,
                    blocksize=_BLOCK_SIZE,
                    callback=_callback,
                )
                stream.start()
                self._streams.append(stream)
                self.sample_rate = rate
                log.info("Audio device %d ('%s') opened at %d Hz", sd_idx, dev_info['name'], rate)
            except Exception as e:
                log.warning("Failed to open audio stream for device %d: %s", sd_idx, e)

    def close(self) -> None:
        """Stop and close all microphone input streams."""
        for stream in self._streams:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass
        self._streams.clear()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()
