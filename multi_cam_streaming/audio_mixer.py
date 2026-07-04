"""Motion-weighted audio mixer that blends built-in camera microphones.

Reads audio from each camera's microphone using sounddevice, applies
per-camera volume weights derived from motion scores, and writes the mixed
PCM stream to an OS pipe for FFmpeg to consume as a raw audio input.

Device matching priority per camera slot:
  1. Explicit 'mic' pattern from config — substring match against sounddevice names.
  2. Linux auto-detect — pyudev USB parent matching (same USB device as /dev/videoN).
  3. Windows auto-detect — sounddevice name substring match against camera pattern.
  4. No match — that camera contributes no audio; a WARNING is logged once.
"""
import logging
import os
import platform
import queue
import threading

import numpy as np
import sounddevice as sd

log = logging.getLogger(__name__)

_SAMPLE_RATE = 44100
_CHANNELS = 1       # mono mix output
_DTYPE = 'int16'
_BLOCK_SIZE = 1024  # samples per read block


def _sd_input_devices() -> list[tuple[int, str]]:
    """Return list of (sounddevice_index, name) for all input-capable devices."""
    return [
        (i, dev['name'])
        for i, dev in enumerate(sd.query_devices())
        if dev['max_input_channels'] >= 1
    ]


def _match_by_name(pattern: str, input_devices: list[tuple[int, str]]) -> int | None:
    """Return sounddevice index for the first device whose name contains pattern (case-insensitive)."""
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

    # Find the udev node for /dev/videoN
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

    # Build ALSA card number → sounddevice index map
    alsa_card_to_sd: dict[int, int] = {}
    for sd_idx, name in input_devices:
        for card_num in range(32):
            if f'hw:{card_num},' in name or f'card{card_num}' in name.lower():
                alsa_card_to_sd[card_num] = sd_idx
                break

    # Walk sound devices sharing the same USB parent
    for audio_dev in context.list_devices(subsystem='sound'):
        if audio_dev.find_parent('usb', 'usb_device') != usb_parent:
            continue
        sys_name = audio_dev.sys_name  # e.g. "card1"
        if sys_name.startswith('card'):
            try:
                card_num = int(sys_name[4:])
            except ValueError:
                continue
            if card_num in alsa_card_to_sd:
                return alsa_card_to_sd[card_num]

    return None


class AudioMixer:
    """Mix audio from multiple camera microphones weighted by motion scores.

    Usage::

        with AudioMixer(camera_entries, video_indexes) as mixer:
            if mixer.audio_pipe_fd is not None:
                streamer = FFmpegStreamer(..., audio_pipe_fd=mixer.audio_pipe_fd)
            while True:
                mixer.set_weights(normalized_motion_scores)
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
        self._cam_to_sd: dict[int, int] = {}   # cam_pos → sounddevice index
        self._streams: list[sd.InputStream] = []
        self._weights = np.zeros(len(camera_entries), dtype=np.float32)
        self._buffers: list[queue.Queue] = []
        self._mix_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._pipe_write_fd: int | None = None
        self.audio_pipe_fd: int | None = None  # readable fd — passed to FFmpeg
        self._out_stream: sd.OutputStream | None = None

    def open(self, output_device: str | None = None) -> None:
        """Discover mic devices, open streams, start the mix thread.

        Args:
            output_device: Substring of the output device name for local playback.
                           None disables local speaker output.
        """
        input_devices = _sd_input_devices()
        system = platform.system()
        warned: set[int] = set()

        for cam_pos, entry in enumerate(self._camera_entries):
            mic_pattern = entry.get('mic') if isinstance(entry, dict) else None
            video_idx = self._video_indexes[cam_pos] if cam_pos < len(self._video_indexes) else -1

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

            self._cam_to_sd[cam_pos] = sd_idx
            log.info("Camera %d → audio device %d ('%s')",
                     cam_pos, sd_idx, sd.query_devices(sd_idx)['name'])

        if not self._cam_to_sd:
            log.warning("No camera mics matched; audio stream will be silent")
            return

        pipe_read_fd, self._pipe_write_fd = os.pipe()
        self.audio_pipe_fd = pipe_read_fd

        n_cams = len(self._camera_entries)
        self._weights = np.ones(n_cams, dtype=np.float32) / max(n_cams, 1)
        self._buffers = [queue.Queue(maxsize=8) for _ in range(n_cams)]

        for cam_pos, sd_idx in self._cam_to_sd.items():
            buf = self._buffers[cam_pos]

            def _callback(indata, frames, time_info, status, _buf=buf):
                if status:
                    log.debug("Audio input status: %s", status)
                try:
                    _buf.put_nowait(indata.copy())
                except queue.Full:
                    pass  # drop block rather than block the audio thread

            try:
                stream = sd.InputStream(
                    device=sd_idx,
                    channels=_CHANNELS,
                    samplerate=_SAMPLE_RATE,
                    dtype=_DTYPE,
                    blocksize=_BLOCK_SIZE,
                    callback=_callback,
                )
                stream.start()
                self._streams.append(stream)
            except Exception as e:
                log.warning("Failed to open audio stream for camera %d: %s", cam_pos, e)

        if output_device is not None:
            out_devices = [
                (i, dev['name'])
                for i, dev in enumerate(sd.query_devices())
                if dev['max_output_channels'] >= 1
            ]
            out_idx = next(
                (i for i, name in out_devices if output_device.lower() in name.lower()),
                None,
            )
            if out_idx is None:
                log.warning("No output device matched '%s'; local audio playback disabled",
                            output_device)
            else:
                try:
                    self._out_stream = sd.OutputStream(
                        device=out_idx,
                        channels=_CHANNELS,
                        samplerate=_SAMPLE_RATE,
                        dtype=_DTYPE,
                        blocksize=_BLOCK_SIZE,
                    )
                    self._out_stream.start()
                    log.info("Audio output → device %d ('%s')",
                             out_idx, sd.query_devices(out_idx)['name'])
                except Exception as e:
                    log.warning("Failed to open audio output device '%s': %s", output_device, e)
                    self._out_stream = None

        self._mix_thread = threading.Thread(target=self._mix_loop, daemon=True,
                                            name='audio-mix')
        self._mix_thread.start()

    def set_weights(self, scores: list[float]) -> None:
        """Update per-camera volume weights from motion scores (called each scoring interval).

        Weights are normalized so they sum to 1 across matched mics only.
        Cameras with no matched audio device always get weight 0.
        """
        arr = np.array(scores, dtype=np.float32)
        for cam_pos in range(len(arr)):
            if cam_pos not in self._cam_to_sd:
                arr[cam_pos] = 0.0
        total = arr.sum()
        self._weights = arr / total if total > 0 else arr

    def _mix_loop(self) -> None:
        """Read one block from each mic, apply weights, write mixed PCM to the pipe."""
        silence = np.zeros(_BLOCK_SIZE, dtype=np.int32)

        try:
            while not self._stop_event.is_set():
                mixed = silence.copy()
                for cam_pos in self._cam_to_sd:
                    weight = float(self._weights[cam_pos]) if cam_pos < len(self._weights) else 0.0
                    if weight == 0.0:
                        continue
                    try:
                        block = self._buffers[cam_pos].get(timeout=0.1)
                        mixed += (block[:, 0].astype(np.int32) * weight).astype(np.int32)
                    except queue.Empty:
                        pass  # camera silent or lagging — contribute silence

                clipped = np.clip(mixed, -32768, 32767).astype(np.int16)
                try:
                    os.write(self._pipe_write_fd, clipped.tobytes())
                except OSError:
                    break  # pipe closed — FFmpeg exited
                if self._out_stream is not None:
                    try:
                        self._out_stream.write(clipped.reshape(-1, 1))
                    except Exception:
                        pass
        finally:
            try:
                os.close(self._pipe_write_fd)
            except OSError:
                pass

    def close(self) -> None:
        """Stop all audio streams and the mix thread."""
        self._stop_event.set()
        for stream in self._streams:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass
        self._streams.clear()
        if self._out_stream is not None:
            try:
                self._out_stream.stop()
                self._out_stream.close()
            except Exception:
                pass
            self._out_stream = None
        if self._mix_thread:
            self._mix_thread.join(timeout=2)
        if self.audio_pipe_fd is not None:
            try:
                os.close(self.audio_pipe_fd)
            except OSError:
                pass
            self.audio_pipe_fd = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()
