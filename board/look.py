#!/usr/bin/env python3
"""Look at the board with Pupper's camera and print what it reads.

    python3 board/look.py                    # one frame from the camera, read it
    python3 board/look.py --watch            # keep reading, about once a second (place the board)
    python3 board/look.py --image photo.jpg  # read a saved frame instead
    python3 board/look.py --json             # the full reading: every cell's ink, position, bearing

The frame comes from the robot stack's camera topic when the stack is running
(robot/start.sh), else straight from the camera with rpicam-still. Debug images
go to --out (default ~/board_look, kept across reboots): raw.jpg (markers and cell centres on the
fisheye frame), view.jpg (the zoomed, undistorted view), topdown.jpg (the
board from above, with each cell's ink score) and last_found_frame.jpg (the last
frame the board was found in: read it again with --image). Look at topdown.jpg first when
a reading is wrong.

Print the board with board/make_board.py; --scale must match the print.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from board_vision import CAMERA_PARAMS, BoardReader  # noqa: E402
from layout import Layout  # noqa: E402

WIDTH, HEIGHT = 1400, 1050  # what the robot stack's camera node publishes; the lens model is for this
TOPIC = "/camera/image_raw/compressed"


class RosCamera:
    """The latest frame from the robot stack's camera topic."""

    def __init__(self):
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import CompressedImage

        rclpy.init()
        self.rclpy = rclpy
        self.node = Node("board_look")
        self.msg = None
        self.node.create_subscription(CompressedImage, TOPIC, self._on_image, 1)

    def _on_image(self, msg):
        self.msg = msg

    def grab(self, timeout: float = 5.0) -> np.ndarray | None:
        self.msg = None
        end = time.monotonic() + timeout
        while self.msg is None and time.monotonic() < end:
            self.rclpy.spin_once(self.node, timeout_sec=0.2)
        if self.msg is None:
            return None
        return cv2.imdecode(np.frombuffer(bytes(self.msg.data), np.uint8), cv2.IMREAD_COLOR)

    @staticmethod
    def available() -> bool:
        try:
            import rclpy  # noqa: F401
        except ImportError:
            return False
        out = subprocess.run(["ros2", "topic", "list"], capture_output=True, text=True, timeout=15).stdout
        return TOPIC in out.split()


class StillCamera:
    """A still straight from the camera (only while the robot stack is off: it holds the camera)."""

    def __init__(self):
        wb = yaml.safe_load(CAMERA_PARAMS.read_text()).get("white_balance", {})
        self.awb = [] if wb.get("awb_enable", True) else ["--awbgains", ",".join(str(g) for g in wb["colour_gains"])]

    def grab(self, timeout: float = 10.0) -> np.ndarray | None:
        cmd = ["rpicam-still", "-n", "-t", "300", "--width", str(WIDTH), "--height", str(HEIGHT),
               "-e", "jpg", "-o", "-", *self.awb]
        try:
            out = subprocess.run(cmd, capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return None
        if out.returncode != 0 or not out.stdout:
            sys.stderr.write(out.stderr.decode(errors="ignore")[-500:])
            return None
        return cv2.imdecode(np.frombuffer(out.stdout, np.uint8), cv2.IMREAD_COLOR)


def report(reading, as_json: bool) -> None:
    if as_json:
        print(json.dumps(reading.as_dict(), indent=1))
        return
    if not reading.found:
        print(f"no board: {reading.message}")
        return
    dist = f", {reading.board_distance_cm:.0f} cm away" if reading.board_distance_cm else ""
    print(f"{reading.message}{dist}; a cell is {reading.native_px_per_cell:.0f} px wide in the raw frame"
          f" (fit error {reading.fit_error_cm:.1f} cm)")
    for row in reading.cells:
        marks = "  ".join(c.mark for c in row)
        inks = " ".join(f"{c.ink:.2f}" for c in row)
        print(f"   {marks}      ink {inks}")
    c = reading.cells[1][1]
    if c.bearing_deg:
        right, down = c.bearing_deg
        print(f"   centre cell: {right:+.0f} deg right, {down:+.0f} deg down of the lens axis")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", type=Path, help="read this saved frame (1400 x 1050, straight from the camera)")
    ap.add_argument("--source", choices=["auto", "ros", "still"], default="auto")
    ap.add_argument("--watch", action="store_true", help="keep reading until Ctrl-C")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--scale", type=float, default=1.4, help="the board's make_board.py --scale (1.4: the A3 board)")
    ap.add_argument("--out", type=Path, default=Path.home() / "board_look", help="debug images go here")
    args = ap.parse_args()

    reader = BoardReader(Layout(scale=args.scale))
    args.out.mkdir(parents=True, exist_ok=True)

    if args.image:
        frame = cv2.imread(str(args.image))
        if frame is None:
            return print(f"cannot read {args.image}") or 1
        camera = None
    else:
        source = args.source
        if source == "auto":
            source = "ros" if RosCamera.available() else "still"
        if source == "still" and not shutil.which("rpicam-still"):
            return print("no camera: the robot stack isn't running and rpicam-still is missing") or 1
        camera = RosCamera() if source == "ros" else StillCamera()
        print(f"camera: {'robot stack topic ' + TOPIC if source == 'ros' else 'rpicam-still (robot stack off)'}")

    while True:
        if camera is not None:
            frame = camera.grab()
            if frame is None:
                print("no frame from the camera")
                if not args.watch:
                    return 1
                continue
        if frame.shape[1] != WIDTH or frame.shape[0] != HEIGHT:
            print(f"note: frame is {frame.shape[1]} x {frame.shape[0]}; the lens model is for {WIDTH} x {HEIGHT}")
        t0 = time.monotonic()
        reading = reader.read(frame)
        cv2.imwrite(str(args.out / "frame.jpg"), frame)
        if reading.found:  # kept until the next good reading, for --image later
            cv2.imwrite(str(args.out / "last_found_frame.jpg"), frame)
        for name, img in reading.debug.items():
            cv2.imwrite(str(args.out / f"{name}.jpg"), img)
        if not args.json:
            print(f"[{time.strftime('%H:%M:%S')}] read in {time.monotonic() - t0:.2f} s")
        report(reading, args.json)
        if not args.watch or camera is None:
            break
    print(f"debug images: {args.out}/", file=sys.stderr)
    return 0 if reading.found else 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        pass
