"""Prevent MediaPipe's unused audio backend from probing ALSA during import."""

import os
import sys


# This project uses only MediaPipe vision tasks. PortAudio can block while
# probing unavailable ALSA devices on Linux; preserve an explicit user setting.
if sys.platform.startswith("linux"):
    os.environ.setdefault("ALSA_CONFIG_PATH", "/dev/null")
