#!/usr/bin/env python3
"""The ELP 16MP USB camera that reads the tic-tac-toe board.

It stands apart from Pupper's own (fisheye) camera, so it can sit anywhere
and point at the board at any angle. Each grab opens the camera, lets the
auto-exposure settle, keeps the last frame and lets the camera go again, so
look.py and the robot server can take turns with it.

    python3 board/usb_camera.py            # save one frame to ~/board_look/usb.jpg
    python3 board/usb_camera.py out.jpg
"""

from __future__ import annotations

import glob
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
PARAMS = HERE / "usb_camera.yaml"  # its lens: a plain (pinhole) lens, no fisheye
WIDTH, HEIGHT = 2048, 1536  # the biggest 4:3 size it streams at 30 fps
SETTLE_S = 0.8  # auto-exposure and white balance settle in this long after opening


def device() -> str | None:
    """Its stable path (/dev/videoN numbers move around between boots)."""
    found = sorted(glob.glob("/dev/v4l/by-id/usb-*16MP*-video-index0"))
    return found[0] if found else None


def available() -> bool:
    return device() is not None


def grab(settle_s: float = SETTLE_S) -> np.ndarray | None:
    """One fresh BGR frame, or None (unplugged, or another program has it open)."""
    dev = device()
    if dev is None:
        return None
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    try:
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        frame, end = None, time.monotonic() + settle_s
        while True:
            ok, f = cap.read()
            if ok:
                frame = f
            if time.monotonic() >= end and frame is not None:
                return frame
            if time.monotonic() > end + 3:
                return frame
    finally:
        cap.release()


def grab_jpeg(quality: int = 92) -> bytes | None:
    frame = grab()
    if frame is None:
        return None
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else None


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "board_look" / "usb.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"camera: {device() or 'not found'}")
    f = grab()
    if f is None:
        sys.exit("no frame (unplugged, or busy: is another program using it?)")
    cv2.imwrite(str(out), f)
    print(f"{f.shape[1]} x {f.shape[0]} -> {out}")
