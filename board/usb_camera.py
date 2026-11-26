#!/usr/bin/env python3
"""The ELP 16MP USB camera that reads the tic-tac-toe board.

It stands apart from Pupper's own (fisheye) camera, so it can sit anywhere
and point at the board at any angle. Each grab opens the camera, lets the
auto-exposure settle, keeps the last frame and lets the camera go again, so
look.py and the robot server can take turns with it.

While the web app shows it live (Camera Source "Board Camera (USB)"), the robot
server keeps it open as a stream (latest_jpeg); grab() in that process then
takes its next frame from the stream instead of opening the camera again. The
stream lets the camera go a few seconds after the last request, so look.py
works again once the web app stops asking.

    python3 board/usb_camera.py            # save one frame to ~/board_look/usb.jpg
    python3 board/usb_camera.py out.jpg
"""

from __future__ import annotations

import glob
import sys
import threading
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


def _open(dev: str):
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


class _Stream(threading.Thread):
    """Keeps the camera open and the newest frame at hand while someone asks."""
    IDLE_S = 5.0  # let the camera go this long after the last request

    def __init__(self, dev: str):
        super().__init__(daemon=True, name="usb_camera")
        self.dev = dev
        self.cond = threading.Condition()
        self.frame: np.ndarray | None = None
        self.frame_t = 0.0  # monotonic time it was read
        self.wanted_t = time.monotonic()
        self.alive = True

    def run(self):
        cap = _open(self.dev)
        try:
            if not cap.isOpened():
                return
            ready_t = time.monotonic() + SETTLE_S  # frames before this aren't settled
            fails = 0
            while time.monotonic() - self.wanted_t < self.IDLE_S:
                ok, f = cap.read()
                if not ok:
                    fails += 1
                    if fails > 30:
                        return  # unplugged
                    time.sleep(0.05)
                    continue
                fails = 0
                if time.monotonic() >= ready_t:
                    with self.cond:
                        self.frame, self.frame_t = f, time.monotonic()
                        self.cond.notify_all()
        finally:
            cap.release()
            with self.cond:
                self.alive = False
                self.cond.notify_all()

    def want(self):
        self.wanted_t = time.monotonic()

    def next_frame(self, timeout: float = 3.0) -> np.ndarray | None:
        """A frame read after this call (so a grab never gets a stale one)."""
        self.want()
        since, end = time.monotonic(), time.monotonic() + timeout
        with self.cond:
            while self.alive and self.frame_t <= since:
                left = end - time.monotonic()
                if left <= 0:
                    return None
                self.cond.wait(left)
            return self.frame if self.frame_t > since else None


_stream: _Stream | None = None
_stream_lock = threading.Lock()


def _live_stream(start: bool) -> _Stream | None:
    global _stream
    with _stream_lock:
        if _stream is not None and _stream.alive:
            _stream.want()
            return _stream
        if not start:
            return None
        dev = device()
        if dev is None:
            return None
        _stream = _Stream(dev)
        _stream.start()
        return _stream


def latest_jpeg(max_width: int = 1024, quality: int = 85) -> bytes | None:
    """The newest frame from the stream (started if need be), shrunk for the web
    app; None while it's starting, unplugged, or busy in another program."""
    s = _live_stream(start=True)
    if s is None:
        return None
    with s.cond:
        frame = s.frame
    if frame is None:
        return None
    h, w = frame.shape[:2]
    if w > max_width:
        frame = cv2.resize(frame, (max_width, round(h * max_width / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else None


def grab(settle_s: float = SETTLE_S) -> np.ndarray | None:
    """One fresh BGR frame, or None (unplugged, or another program has it open)."""
    s = _live_stream(start=False)
    if s is not None:
        frame = s.next_frame()
        if frame is not None:
            return frame
    dev = device()
    if dev is None:
        return None
    cap = _open(dev)
    try:
        if not cap.isOpened():
            return None
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
