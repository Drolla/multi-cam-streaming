"""Weighted audio mixer that blends camera microphone streams.

Receives an open AudioManager (which owns device discovery and InputStream
lifecycle), applies per-camera volume weights supplied by the caller via
set_weights(), and writes the mixed PCM to an OS pipe for FFmpeg and/or a
local speaker.
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
_FULL_SCALE = 32768.0   # int16 full-scale magnitude, used as the 0 dBFS reference


def _to_db(peak: float) -> float:
    """Convert a peak PCM magnitude to dBFS, using _FULL_SCALE as the 0 dBFS reference."""
    if peak <= 0:
        return float('-inf')
    return 20 * np.log10(peak / _FULL_SCALE)


def _fmt_db(value_db: float) -> str:
    """Format a dBFS value at a fixed width so '-inf' aligns with '-nn.n' entries."""
    return f"{value_db:5.1f}" if value_db != float('-inf') else " -inf"


def _apply_compression(mixed: np.ndarray, threshold_db: float, ratio_db: float) -> np.ndarray:
    """Apply a simplified soft-knee compressor to a mixed int32 PCM block.

    Computes a single gain factor from the block's peak level and applies it
    uniformly - no attack/release envelope or lookahead, just an instantaneous
    per-block gain. Blocks are already small (~21ms at 48kHz), so this is a
    reasonable approximation without the complexity of real envelope tracking.
    """
    peak = float(np.max(np.abs(mixed)))
    if peak <= 0:
        return mixed
    level_db = 20 * np.log10(peak / _FULL_SCALE)
    if level_db <= threshold_db:
        return mixed
    gain_reduction_db = (level_db - threshold_db) * (1 - 1 / ratio_db)
    gain = 10 ** (-gain_reduction_db / 20)
    return (mixed.astype(np.float64) * gain).astype(np.int32)


class AudioMixer:
    """Mix PCM blocks from an AudioManager weighted by caller-supplied per-camera weights.

    Usage::

        with AudioManager(camera_entries, video_indexes) as mgr:
            with AudioMixer(mgr, pipe_needed=True, output_device="Speakers") as mixer:
                streamer = FFmpegStreamer(..., audio_pipe_fd=mixer.audio_pipe_fd,
                                         audio_sample_rate=mixer.audio_sample_rate)
                while True:
                    mixer.set_weights(weights)
    """

    def __init__(self, audio_manager: AudioManager, pipe_needed: bool = True,
                 output_device: str | None = None, transition_duration: float = 0.0,
                 compression: dict | None = None, size_threshold: float = 0.0):
        """
        Args:
            audio_manager:  An already-open AudioManager providing buffers and cam_to_sd.
            pipe_needed:    True when FFmpeg will consume the audio pipe (stream/both modes).
                            False for display-only mode — pipe creation is skipped.
            output_device:  Index (as string) or name substring for local speaker playback.
                            Stored so __enter__ can call open() without arguments.
            transition_duration: Time constant (seconds) for exponential smoothing of camera
                            weights, so volume changes ramp instead of jumping. 0 disables
                            smoothing (weights applied instantly, as before).
            compression:    Optional {'threshold_db': float, 'ratio_db': float} soft-knee
                            compressor settings applied to the final mixed signal. None
                            disables compression.
            size_threshold: Minimum displayed slot size (as passed to set_weights) for a
                            camera to contribute audio; cameras below this are gated to
                            weight 0 before normalization. 0 disables gating.
        """
        self._mgr = audio_manager
        self._pipe_needed = pipe_needed
        self._output_device = output_device
        self._transition_duration = transition_duration
        self._compression = compression
        self._size_threshold = size_threshold
        self._target_weights = np.zeros(0, dtype=np.float32)
        self._smoothed_weights = np.zeros(0, dtype=np.float32)
        self._mix_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._start_gate = threading.Event()  # set when FFmpeg is ready to consume the pipe
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
        self._target_weights = np.ones(n_cams, dtype=np.float32) / max(n_cams, 1)
        self._smoothed_weights = self._target_weights.copy()
        self.audio_sample_rate = self._mgr.sample_rate

        if self._pipe_needed:
            pipe_read_fd, self._pipe_write_fd = os.pipe()
            self.audio_pipe_fd = pipe_read_fd
        else:
            self._start_gate.set()  # no pipe to fill — safe to start immediately

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
        """Update per-camera target volume weights.

        Called on each frame-arrangement change. scores are displayed slot sizes (or any
        non-negative per-camera magnitude); values below size_threshold are gated to 0.
        Weights are normalised so they sum to 1 across matched mics only; unmatched
        cameras always get weight 0. The mix loop ramps self._smoothed_weights toward
        this target every block rather than applying it instantly (see transition_duration).
        """
        arr = np.array(scores, dtype=np.float32)
        if self._size_threshold > 0:
            arr[arr < self._size_threshold] = 0.0
        for cam_pos in range(len(arr)):
            if cam_pos not in self._mgr.cam_to_sd:
                arr[cam_pos] = 0.0
        total = arr.sum()
        self._target_weights = arr / total if total > 0 else arr
        if len(self._smoothed_weights) != len(self._target_weights):
            # Length changed (shouldn't normally happen mid-run) — snap instead of ramping
            # from a mismatched array.
            self._smoothed_weights = self._target_weights.copy()

        if log.isEnabledFor(logging.INFO):
            scores_str = "|".join(f"{s:.2f}" for s in scores)
            weights_str = "|".join(f"{w:.2f}" for w in self._target_weights)
            log.info("Audio weights changing: scores=%s smoothed_weights=%s", scores_str, weights_str)

    def signal_ready(self) -> None:
        """Release the mix loop to begin writing. Call after FFmpeg has started."""
        self._start_gate.set()

    def _mix_loop(self) -> None:
        """Pace on the primary mic queue, mix weighted blocks, output PCM."""
        self._start_gate.wait()  # hold until FFmpeg is ready to consume the pipe
        silence = np.zeros(_BLOCK_SIZE, dtype=np.int32)
        first_rate = next(iter(self._mgr.sample_rates.values()), _SAMPLE_RATE)
        block_duration = _BLOCK_SIZE / first_rate
        # One-pole smoothing coefficient; alpha=1.0 (instant) when smoothing is disabled.
        alpha = (1 - np.exp(-block_duration / self._transition_duration)
                 if self._transition_duration > 0 else 1.0)

        unique_queues = list({id(q): q for q in self._mgr.buffers.values()}.values())

        try:
            while not self._stop_event.is_set():
                try:
                    primary_block = unique_queues[0].get(timeout=block_duration * 4)
                except queue.Empty:
                    continue

                self._smoothed_weights += alpha * (self._target_weights - self._smoothed_weights)

                mixed = silence.copy()
                weights = self._smoothed_weights
                seen: dict[int, np.ndarray] = {id(unique_queues[0]): primary_block}
                debug_enabled = log.isEnabledFor(logging.DEBUG)
                stream_levels = [] if debug_enabled else None

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
                    volume = self._mgr.cam_volumes.get(cam_pos, 1.0)
                    adjusted = (block[:, 0].astype(np.int32) * volume * weight).astype(np.int32)
                    mixed += adjusted
                    if debug_enabled:
                        pre_db = _to_db(float(np.max(np.abs(block[:, 0]))))
                        post_db = _to_db(float(np.max(np.abs(adjusted))))
                        stream_levels.append(f"{_fmt_db(pre_db)}/{_fmt_db(post_db)}")

                if self._compression is not None:
                    mixed = _apply_compression(mixed, self._compression['threshold_db'],
                                               self._compression['ratio_db'])

                if debug_enabled:
                    raw_mixed_db = _to_db(float(np.max(np.abs(mixed))))
                    compressed_db = _to_db(float(np.max(np.abs(mixed))))
                    weights_str = ";".join(f"{w:.2f}" for w in self._smoothed_weights)
                    log.debug("weights=%s ; pre/post %s ; mix=%s; comp=%s",
                              weights_str,
                              " ; ".join(stream_levels), _fmt_db(raw_mixed_db), _fmt_db(compressed_db))

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
