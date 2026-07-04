"""Motion-weighted audio mixer that blends camera microphone streams.

Receives an open AudioManager (which owns device discovery and InputStream
lifecycle), applies per-camera volume weights derived from motion scores,
and writes the mixed PCM to an OS pipe for FFmpeg and/or a local speaker.
"""
import logging
import os
import queue
import threading

import numpy as np
import sounddevice as sd

from multi_cam_streaming.audio_manager import (
    AudioManager,
    _BLOCK_SIZE,
    _CHANNELS,
    _DTYPE,
    _SAMPLE_RATE,
)

log = logging.getLogger(__name__)

_MIX_QUEUE_MAXSIZE = 4  # intentionally smaller than input queue; stale audio is dropped quickly


class AudioMixer:
    """Mix PCM blocks from an AudioManager weighted by per-camera motion scores.

    Usage::

        with AudioManager(camera_entries, video_indexes) as mgr:
            with AudioMixer(mgr, pipe_needed=True) as mixer:
                mixer.open(output_device="Speakers")
                streamer = FFmpegStreamer(..., audio_pipe_fd=mixer.audio_pipe_fd,
                                         audio_sample_rate=mixer.audio_sample_rate)
                while True:
                    mixer.set_weights(motion_scores)
    """

    def __init__(self, audio_manager: AudioManager, pipe_needed: bool = True,
                 output_device: str | None = None):
        """
        Args:
            audio_manager:  An already-open AudioManager providing buffers and cam_to_sd.
            pipe_needed:    True when FFmpeg will consume the audio pipe (stream/both modes).
                            False for display-only mode — pipe creation is skipped.
            output_device:  Index (as string) or name substring for local speaker playback.
                            Stored so __enter__ can call open() without arguments.
        """
        self._mgr = audio_manager
        self._pipe_needed = pipe_needed
        self._output_device = output_device
        self._weights = np.zeros(0, dtype=np.float32)
        self._mix_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._pipe_write_fd: int | None = None
        self._out_stream: sd.OutputStream | None = None
        self._out_queue: queue.Queue = queue.Queue(maxsize=_MIX_QUEUE_MAXSIZE)

        self.audio_pipe_fd: int | None = None   # readable fd — passed to FFmpeg
        self.audio_sample_rate: int = _SAMPLE_RATE

    def open(self, output_device: str | None = None) -> None:
        """Open pipe and/or output stream, initialise weights, start mix thread.

        Args:
            output_device: Index (as string) or name substring of the output device
                           for local speaker playback. None disables local output.
        """
        if not self._mgr.cam_to_sd:
            log.warning("AudioManager has no matched mics; mixer will produce silence")
            return

        n_cams = self._mgr.camera_count
        self._weights = np.ones(n_cams, dtype=np.float32) / max(n_cams, 1)
        self.audio_sample_rate = self._mgr.sample_rate

        if self._pipe_needed:
            pipe_read_fd, self._pipe_write_fd = os.pipe()
            self.audio_pipe_fd = pipe_read_fd

        if output_device is not None:
            out_devices = [
                (i, dev['name'])
                for i, dev in enumerate(sd.query_devices())
                if dev['max_output_channels'] >= 1
            ]
            if output_device.isdigit():
                idx = int(output_device)
                out_idx = idx if any(i == idx for i, _ in out_devices) else None
            else:
                out_idx = next(
                    (i for i, name in out_devices if output_device.lower() in name.lower()),
                    None,
                )
            if out_idx is None:
                log.warning("No output device matched '%s'; local audio playback disabled",
                            output_device)
            else:
                try:
                    out_dev_info = sd.query_devices(out_idx)
                    out_rate = int(out_dev_info['default_samplerate'])

                    def _out_callback(outdata, frames, time_info, status,  # noqa: ARG001
                                      _q=self._out_queue):
                        if status:
                            log.debug("Audio output status: %s", status)
                        try:
                            block = _q.get_nowait()
                            outdata[:] = block.reshape(-1, 1)
                        except queue.Empty:
                            outdata.fill(0)

                    self._out_stream = sd.OutputStream(
                        device=out_idx,
                        channels=_CHANNELS,
                        samplerate=out_rate,
                        dtype=_DTYPE,
                        blocksize=_BLOCK_SIZE,
                        callback=_out_callback,
                    )
                    self._out_stream.start()
                    log.info("Audio output → device %d ('%s') at %d Hz",
                             out_idx, out_dev_info['name'], out_rate)
                except Exception as e:
                    log.warning("Failed to open audio output device '%s': %s", output_device, e)
                    self._out_stream = None

        self._mix_thread = threading.Thread(target=self._mix_loop, daemon=True,
                                            name='audio-mix')
        self._mix_thread.start()

    def set_weights(self, scores: list[float]) -> None:
        """Update per-camera volume weights from motion scores.

        Called each scoring interval. Weights are normalised so they sum to 1
        across matched mics only; unmatched cameras always get weight 0.
        """
        arr = np.array(scores, dtype=np.float32)
        for cam_pos in range(len(arr)):
            if cam_pos not in self._mgr.cam_to_sd:
                arr[cam_pos] = 0.0
        total = arr.sum()
        self._weights = arr / total if total > 0 else arr

    def _mix_loop(self) -> None:
        """Pace on the primary mic queue, mix weighted blocks, output PCM."""
        silence = np.zeros(_BLOCK_SIZE, dtype=np.int32)
        first_rate = next(iter(self._mgr.sample_rates.values()), _SAMPLE_RATE)
        block_duration = _BLOCK_SIZE / first_rate

        unique_queues = list({id(q): q for q in self._mgr.buffers.values()}.values())

        try:
            while not self._stop_event.is_set():
                try:
                    primary_block = unique_queues[0].get(timeout=block_duration * 4)
                except queue.Empty:
                    continue

                mixed = silence.copy()
                weights = self._weights
                seen: dict[int, np.ndarray] = {id(unique_queues[0]): primary_block}

                for cam_pos, buf in self._mgr.buffers.items():
                    weight = float(weights[cam_pos]) if cam_pos < len(weights) else 0.0
                    if weight == 0.0:
                        continue
                    buf_id = id(buf)
                    if buf_id not in seen:
                        try:
                            seen[buf_id] = buf.get_nowait()
                        except queue.Empty:
                            seen[buf_id] = np.zeros((_BLOCK_SIZE, _CHANNELS), dtype=_DTYPE)
                    block = seen[buf_id]
                    mixed += (block[:, 0].astype(np.int32) * weight).astype(np.int32)

                clipped = np.clip(mixed, -32768, 32767).astype(np.int16)

                if self._pipe_write_fd is not None:
                    try:
                        os.write(self._pipe_write_fd, clipped.tobytes())
                    except OSError:
                        break  # pipe closed — FFmpeg exited

                if self._out_stream is not None:
                    try:
                        self._out_queue.put_nowait(clipped)
                    except queue.Full:
                        pass  # drop rather than stall the mix loop
        finally:
            if self._pipe_write_fd is not None:
                try:
                    os.close(self._pipe_write_fd)
                except OSError:
                    pass

    def close(self) -> None:
        """Stop the mix thread and release the output stream and pipe."""
        self._stop_event.set()
        if self._mix_thread:
            self._mix_thread.join(timeout=2)
        if self._out_stream is not None:
            try:
                self._out_stream.stop()
                self._out_stream.close()
            except Exception:
                pass
            self._out_stream = None
        if self.audio_pipe_fd is not None:
            try:
                os.close(self.audio_pipe_fd)
            except OSError:
                pass
            self.audio_pipe_fd = None

    def __enter__(self):
        """Open the mixer and return self."""
        self.open(self._output_device)
        return self

    def __exit__(self, *args):
        """Stop the mix thread and release all resources."""
        self.close()
