"""multi_cam_streaming - Stream USB cameras to YouTube."""

__version__ = "1.0.0"
__author__ = "Andreas Drollinger"

from .camera_manager import CameraManager
from .ffmpeg import FFmpegStreamer
from .frame_compositor import FrameCompositor

__all__ = ["CameraManager", "FFmpegStreamer", "FrameCompositor"]
