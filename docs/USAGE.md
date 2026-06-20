# Usage Guide

## Prerequisites

- Virtual environment activated: `source venv/bin/activate`
- Configuration file created: `cp config.example.yaml config_rpi.yaml` (or `config_windows.yaml`)
- YouTube stream key set in your config file or environment variable
- USB cameras connected

## Stream to YouTube

Stream combined camera feed to YouTube in real-time.

### Prerequisites

1. Get your YouTube Stream Key:
   - Go to [YouTube Studio](https://studio.youtube.com)
   - Click "Create" → "Go Live"
   - Select "Stream" tab
   - Copy your RTMP stream key

2. Set the stream key:
   ```bash
   export YOUTUBE_STREAM_KEY="your-stream-key-here"
   ```
   Or edit your config file (e.g. `config_rpi.yaml`):
   ```yaml
   youtube:
     stream_key: "your-stream-key-here"
   ```

### Start Streaming
```bash
python3 scripts/camera_viewer.py
```

### Display Camera Feed Locally
```bash
python3 scripts/camera_viewer.py --mode display
```

### Display and Stream Simultaneously
```bash
python3 scripts/camera_viewer.py --mode both
```

### With Custom Config
```bash
python3 scripts/camera_viewer.py --config custom.yaml --mode stream
```

### List Available Cameras
```bash
python3 scripts/camera_viewer.py --list-cameras
```

### If installed as command
```bash
stream-to-youtube
stream-to-youtube --mode display
stream-to-youtube --mode both
stream-to-youtube --config custom.yaml --mode stream
stream-to-youtube --list-cameras
```

### Controls
- **q** - Quit display (when in display or both modes)
- **Ctrl+C** - Stop streaming/display (graceful shutdown)

## Configuration

### Config File Format

See `config.example.yaml` for a fully annotated reference. Key sections:

```yaml
# Each entry is a plain pattern string or a dict with optional per-camera attributes.
cameras:
  - "usb.*0-1"
  - pattern: "usb.*1-1"
    min_slot: 1              # never fills slot 0 (largest)
    max_slot: 2              # never fills slots beyond 2
    activity_multiplier: 1.5 # perceived activity boosted 50%

motion:
  check_interval: 1.0      # seconds between motion score computations
  threshold: 15            # pixel diff noise gate (0–255)
  change_threshold: 0.05   # min score delta to consider a layout switch
  min_switch_interval: 5.0 # minimum seconds between accepted switches

output:
  width: 1280
  height: 720
  fps: 30

youtube:
  stream_key: "${YOUTUBE_STREAM_KEY}"
  rtmp_url: "rtmp://a.rtmp.youtube.com/live2/"

transition_duration: 0.5   # seconds to animate between layouts

layouts:
  - name: quad-4x1
    frames:
      - {pos: [0.0, 0.0], size: 0.5}  # slot 0 — top-left (largest gets most active cam)
      - {pos: [0.5, 0.0], size: 0.5}  # slot 1 — top-right
      - {pos: [0.0, 0.5], size: 0.5}  # slot 2 — bottom-left
      - {pos: [0.5, 0.5], size: 0.5}  # slot 3 — bottom-right
```

Cameras are assigned to layout slots dynamically by **motion priority**: the camera
with the most detected motion fills the largest slot, the next most active fills the
second-largest, and so on. The layout itself is also selected automatically based on
how dominant the top-scoring camera is relative to the others.

Each frame slot is defined by:
- `pos: [x, y]` — top-left corner as a fraction of the output dimensions (0.0–1.0)
- `size` — scalar fraction applied to both width and height (0.0–1.0)

Slots are composited in list order — later slots render on top of earlier ones, enabling
picture-in-picture and overlay layouts. Slots without an assigned camera show black.

The same camera can appear more than once in the `cameras` list — it is opened and
read only once and reused.

### Find Camera Identifiers

List the camera devices detected by this project (works on any platform):
```bash
python3 scripts/camera_viewer.py --list-cameras
```

On Linux/Raspberry Pi you can also use v4l-utils directly:
```bash
v4l2-ctl --list-devices
```

Output example:
```
UVC Camera (046d:0825): /dev/video0 /dev/video1
    usb-0000:00:14.0-1
```

Use the device name (e.g., `usb.*0-1`) or full name as the regex pattern.

## Common Issues

### "No cameras found"
```bash
# Check connected cameras
v4l2-ctl --list-devices

# Verify identifier patterns in your config file
# Use a simpler pattern if needed:
cameras:
  - ".*"  # Match all cameras
```

### Camera feed frozen or slow
- Reduce FPS in your config file
- Reduce frame dimensions
- Check CPU usage with `top`

### FFmpeg errors during streaming
- Verify YouTube stream key is correct
- Check internet connection
- View FFmpeg logs by uncommenting in ffmpeg.py:
  ```python
  #stdout=subprocess.DEVNULL,
  #stderr=subprocess.DEVNULL,
  ```

### Permission denied accessing cameras
```bash
# Add user to video group
sudo usermod -a -G video $(whoami)
newgrp video
```

## Running in Background (Systemd Service)

Create `/etc/systemd/system/multi-cam-stream.service`:

```ini
[Unit]
Description=Raspberry Pi Camera to YouTube Stream
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/multi_cam_streaming
Environment="YOUTUBE_STREAM_KEY=your-stream-key"
ExecStart=/home/pi/multi_cam_streaming/venv/bin/stream-to-youtube
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable multi-cam-stream
sudo systemctl start multi-cam-stream
```

Check status:
```bash
sudo systemctl status multi-cam-stream
```

View logs:
```bash
sudo journalctl -u multi-cam-stream -f
```

## Tips & Tricks

### Adjust Video Quality

Modify your config file for better quality (higher bitrate in FFmpeg):

```yaml
frame_dims:
  width: 640
  height: 480
fps: 60
```

### Multiple Streams

Create different config files:
```bash
cp config.example.yaml stream1.yaml
cp config.example.yaml stream2.yaml
# Edit each with different settings
python3 scripts/camera_viewer.py --config stream1.yaml --mode stream
```

### Test Camera Feed Locally

```bash
# View camera feed without streaming
python3 scripts/camera_viewer.py --mode display
```
```
