"""FFmpeg-based streamer for sending video frames to a YouTube RTMP endpoint."""
import atexit
import logging
import subprocess

from multi_cam_streaming.audio_manager import SAMPLE_RATE as _AUDIO_SAMPLE_RATE

_FRAME_DIMS = (320, 240)  # Output frame dimensions (width, height)
_FPS = 30

log = logging.getLogger(__name__)


class FFmpegStreamer:
    """Stream video frames to YouTube via FFmpeg."""

    @staticmethod
    def _build_ffmpeg_cmd(fps=_FPS, frame_dims=_FRAME_DIMS, *, youtube_url,
                          audio_pipe_fd=None, audio_sample_rate=_AUDIO_SAMPLE_RATE):
        """Build the FFmpeg command with the given parameters.

        Args:
            fps: Frames per second
            frame_dims: Dimensions of the frames being written, i.e. the combined
                grid size, as (width, height)
            youtube_url: YouTube RTMP URL
            audio_pipe_fd: File descriptor of a readable pipe carrying raw 16-bit
                mono PCM. When provided, FFmpeg reads live audio from this pipe
                instead of generating silence.
            audio_sample_rate: Sample rate of the PCM data in the pipe (Hz).

        Returns:
            List of FFmpeg command arguments
        """
        if audio_pipe_fd is not None:
            audio_input = [
                "-f", "s16le",
                "-ar", str(audio_sample_rate),
                "-ac", "1",
                "-i", f"pipe:{audio_pipe_fd}",
            ]
        else:
            audio_input = [
                "-f", "lavfi",
                "-i", f"anullsrc=channel_layout=stereo:sample_rate={_AUDIO_SAMPLE_RATE}",
            ]
        return [
            "ffmpeg",
            "-y",
            "-fflags", "+genpts",
            "-use_wallclock_as_timestamps", "1",
            "-re",  # read input at real-time speed
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{frame_dims[0]}x{frame_dims[1]}",
            "-r", str(fps),
            "-i", "-",
            *audio_input,
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-pix_fmt", "yuv420p",
            "-b:v", "2000k",
            # Keyframe interval required by YouTube
            "-g", str(fps * 2),
            "-keyint_min", str(fps * 2),
            "-c:a", "aac",
            "-b:a", "128k",
            "-f", "flv",
            youtube_url
        ]

    def __init__(self, youtube_url, fps=None, frame_dims=None, audio_pipe_fd=None,
                 audio_sample_rate=_AUDIO_SAMPLE_RATE):
        """Initialize FFmpeg streamer.

        Args:
            youtube_url: YouTube RTMP URL (required)
            fps: Frames per second (default: module FPS)
            frame_dims: Dimensions of the frames being written, i.e. the combined grid
                size, as (width, height) (default: module FRAME_DIMS)
            audio_pipe_fd: Optional file descriptor of a readable pipe with raw PCM
                audio (16-bit mono). When None, a silent stream is used.
            audio_sample_rate: Sample rate of the PCM data in the pipe (Hz).
        """
        self.fps = fps if fps is not None else _FPS
        self.frame_dims = frame_dims if frame_dims is not None else _FRAME_DIMS
        self.youtube_url = youtube_url
        self._audio_pipe_fd = audio_pipe_fd
        self.ffmpeg_cmd = self._build_ffmpeg_cmd(
            self.fps, self.frame_dims,
            youtube_url=self.youtube_url,
            audio_pipe_fd=audio_pipe_fd,
            audio_sample_rate=audio_sample_rate,
        )
        self.process = None
        self._start_process()
        atexit.register(self.cleanup)

    def _start_process(self):
        """Start the FFmpeg subprocess."""
        extra = {}
        if self._audio_pipe_fd is not None:
            extra['pass_fds'] = (self._audio_pipe_fd,)
        # stdout/stderr are intentionally inherited so FFmpeg progress is visible in the terminal
        self.process = subprocess.Popen(
            self.ffmpeg_cmd,
            stdin=subprocess.PIPE,
            bufsize=0,
            **extra,
        )
        log.info("FFmpeg started")

    def _restart_process(self):
        """Restart the FFmpeg subprocess."""
        log.warning("Restarting FFmpeg...")
        self._safe_terminate()
        self._start_process()

    def _safe_terminate(self):
        """Safely terminate the FFmpeg subprocess."""
        if not self.process:
            return

        try:
            if self.process.stdin:
                self.process.stdin.close()
        except Exception:
            pass

        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()

    def cleanup(self):
        """Clean up resources."""
        log.info("Cleaning up FFmpeg...")
        self._safe_terminate()

    def write_frame(self, frame):
        """Write a frame to the FFmpeg stream.

        Args:
            frame: OpenCV frame (numpy array)
        """
        try:
            self.process.stdin.write(frame.tobytes())
        except BrokenPipeError:
            # FFmpeg died → restart and retry once
            self._restart_process()
            try:
                self.process.stdin.write(frame.tobytes())
            except Exception as e:
                log.error("Failed to write frame after restart: %s", e)
                self.cleanup()
                raise
