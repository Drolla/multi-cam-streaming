# Installation Guide

## System Requirements

- Python 3.7 or higher
- FFmpeg installed and in PATH
- Cameras supported by OpenCV (USB, CSI, IP cameras, etc.)
- v4l-utils — Linux/Raspberry Pi only, for camera detection
- pygrabber — Windows only, installed automatically via pip

## Linux / Raspberry Pi Setup

### 1. Install System Dependencies

**Ubuntu / Debian / Raspberry Pi (Raspberry Pi OS):**
```bash
sudo apt-get update
sudo apt-get install python3 python3-pip python3-venv
sudo apt-get install ffmpeg
sudo apt-get install v4l-utils
```

### 2. Clone and Setup Project

```bash
cd ~
git clone https://github.com/yourusername/multi_cam_streaming.git
cd multi_cam_streaming
```

### 3. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure the Application

```bash
cp config.example.yaml config.yaml
nano config.yaml  # Edit with your settings
```

### 6. Set YouTube Credentials

**Method 1: Environment Variable (Recommended)**
```bash
export YOUTUBE_STREAM_KEY="your-youtube-stream-key-here"
```

**Method 2: Edit your config file**
```bash
nano config.yaml
# Set stream_key to your actual key (or use the YOUTUBE_STREAM_KEY env variable)
```

### 7. Verify Camera Detection

```bash
python3 scripts/camera_viewer.py --list-cameras
```

### 8. Verify Installation

```bash
python3 scripts/camera_viewer.py --mode display
```

If you see camera feeds displayed, the installation is successful. Press 'q' to exit.

---

## Windows Setup

### 1. Install FFmpeg

Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH, or install via winget:
```powershell
winget install ffmpeg
```

### 2. Clone and Setup Project

```powershell
git clone https://github.com/yourusername/multi_cam_streaming.git
cd multi_cam_streaming
```

### 3. Create Virtual Environment

```powershell
python -m venv venv
venv\Scripts\activate
```

### 4. Install Python Dependencies

```powershell
pip install -r requirements.txt
```

pygrabber is installed automatically on Windows and is used for camera detection.

### 5. Configure the Application

```powershell
copy config.example.yaml config.yaml
notepad config.yaml
```

### 6. Set YouTube Credentials

**Method 1: Environment Variable (Recommended)**
```powershell
$env:YOUTUBE_STREAM_KEY = "your-youtube-stream-key-here"
```

**Method 2: Edit your config file**
```powershell
# Set stream_key in config.yaml to your actual key
```

### 7. Verify Camera Detection

```powershell
python scripts/camera_viewer.py --list-cameras
```

### 8. Verify Installation

```powershell
python scripts/camera_viewer.py --mode display
```

---

## Optional: Install as Command-line Tools

To use the installed commands from anywhere:

```bash
pip install -e .
```

Then you can run:
```bash
stream-to-youtube
```

---

## Troubleshooting

### FFmpeg not found
```bash
# Linux
which ffmpeg
sudo apt-get install ffmpeg

# Windows — ensure ffmpeg is in PATH after installation
ffmpeg -version
```

### v4l2-ctl not found (Linux only)
```bash
sudo apt-get install v4l-utils
```

### Cameras not detected
```bash
# Linux: check connected cameras
lsusb
v4l2-ctl --list-devices

# Check camera permissions (Linux)
ls -la /dev/video*
# You may need to add user to video group:
sudo usermod -a -G video $USER
# Then logout and login again
```

### Permission denied errors (Linux)
```bash
sudo usermod -a -G video $(whoami)
newgrp video
```

## Next Steps

- See [USAGE.md](USAGE.md) for how to run the applications
- Check [config.example.yaml](../config.example.yaml) for configuration options
