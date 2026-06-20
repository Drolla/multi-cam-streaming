import logging
import subprocess
import atexit

FRAME_DIMS = (320, 240)  # Output frame dimensions (width, height)
FPS = 30

log = logging.getLogger(__name__)


class FFmpegStreamer:
    """Stream video frames to YouTube via FFmpeg."""

    @staticmethod
    def _build_ffmpeg_cmd(fps=FPS, frame_ims=FRAME_DIMS, *, youtube_url):
        """Build the FFmpeg command with the given parameters.

        Args:
            fps: Frames per second
            frame_ims: Dimensions of the frames being written, i.e. the combined
                grid size, as (width, height)
            youtube_url: YouTube RTMP URL

        Returns:
            List of FFmpeg command arguments
        """
        return [
            "ffmpeg",
            "-y",
            "-fflags", "+genpts",
            "-use_wallclock_as_timestamps", "1",
            "-re",  # read input at real-time speed
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{frame_ims[0]}x{frame_ims[1]}",
            "-r", str(fps),
            "-i", "-",
            "-f", "lavfi",
            "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
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

    def __init__(self, youtube_url, fps=None, frame_ims=None):
        """Initialize FFmpeg streamer.

        Args:
            youtube_url: YouTube RTMP URL (required)
            fps: Frames per second (default: module FPS)
            frame_ims: Dimensions of the frames being written, i.e. the combined grid
                size, as (width, height) (default: module FRAME_DIMS)
        """
        self.fps = fps if fps is not None else FPS
        self.frame_dims = frame_ims if frame_ims is not None else FRAME_DIMS
        self.youtube_url = youtube_url
        self.ffmpeg_cmd = self._build_ffmpeg_cmd(self.fps, self.frame_dims, youtube_url=self.youtube_url)
        self.process = None
        self._start_process()
        atexit.register(self.cleanup)

    def _start_process(self):
        """Start the FFmpeg subprocess."""
        self.process = subprocess.Popen(
            self.ffmpeg_cmd,
            stdin=subprocess.PIPE,
            #stdout=subprocess.DEVNULL,
            #stderr=subprocess.DEVNULL,
            bufsize=0
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
