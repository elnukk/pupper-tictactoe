#!/usr/bin/env python3
"""Measure how far Pupper's timed walks really go, using the board as a ruler.

    robot/start.sh --kiosk                      # first: the robot stack and server running
    python3 board/calibrate_walk.py             # then: the board's near edge 10-20 cm in front of Pupper
    python3 board/calibrate_walk.py --dry-run   # only look and print where Pupper is: nothing moves

Pupper looks at the board, walks a short timed move (forward, sideways or a
turn), looks again, walks it back, and looks again. From where the board appears
each time it measures how far each move really went, fits "distance = rate x
time + offset" for each kind of move, and writes board/walk_calibration.yaml,
which the stamping uses to turn "walk 23 cm" into seconds. It also measures how
far the camera looks off Pupper's walking direction. It prints how much the
moves varied: that spread is about how far off a blind stamp will land.

Lay the board flat and taped down, "Pupper's side" towards Pupper, with its
near edge 10-20 cm in front of Pupper's face, where Pupper reads it (with the
A4 board, about 12 cm). Forward moves may walk partly onto the board: Pupper
still sees its far markers from there, and a look that loses the board is
skipped. Moves are short (at most about 14 cm or 30 degrees) and always walked
back. Ctrl-C stops Pupper.
Calibrate again on a different floor.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stamp_plan as sp  # noqa: E402
from board_vision import BoardReader  # noqa: E402
from layout import Layout  # noqa: E402

SERVER = "http://localhost:8000"
SETTLE_S = 1.5  # after a move: the queue starts it, the body stops swaying
# Timed moves, seconds. At 0.2 m/s commanded: 5, 8, 11 cm forward (short bursts really go about 1.5x
# that), 8, 14, 20 cm sideways. At 30 deg/s: 15, 30 deg.
# Sideways stops at 0.7 s: stamping only steps a few cm sideways, and longer moves lose the board.
DURATIONS = {"forward": [0.25, 0.4, 0.55], "sideways": [0.4, 0.7], "turn": [0.5, 1.0]}


def http(path: str, body: dict | None = None, timeout: float = 10.0):
    req = urllib.request.Request(SERVER + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="GET" if body is None else "POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read()
        return data if r.headers.get_content_type() == "image/jpeg" else json.loads(data or b"null")


class Looker:
    def __init__(self, layout: Layout):
        self.reader = BoardReader(layout)

    def pose(self, tries: int = 4) -> sp.Pose | None:
        for _ in range(tries):
            frame = cv2.imdecode(np.frombuffer(http("/camera/raw"), np.uint8), cv2.IMREAD_COLOR)
            reading = self.reader.read(frame)
            if reading.found and reading.board_R is not None:
                return sp.pose_from_board(reading.board_R, reading.board_t)
            time.sleep(0.3)
        return None


def move(axis: str, sign: int, seconds: float, speed: float, turn_speed: float) -> None:
    """One timed move through the robot server (its convention: right and clockwise positive)."""
    body = {"forward_backward_velocity": 0.0, "right_left_velocity": 0.0, "turning_velocity": 0.0, "duration": seconds}
    if axis == "forward":
        body["forward_backward_velocity"] = sign * speed
    elif axis == "sideways":
        body["right_left_velocity"] = -sign * speed  # left (robot positive) is negative here
    else:
        body["turning_velocity"] = -sign * turn_speed  # counter-clockwise (robot positive) is negative here
    res = http("/move/velocity", body)
    if not res.get("success"):
        raise SystemExit(f"the robot refused the move: {res.get('message')}")
    time.sleep(seconds + SETTLE_S)


def fit(samples: list[tuple[float, float]]) -> tuple[float, float, float]:
    """(rate, offset, spread) for distance = rate * seconds + offset, least squares."""
    t = np.array([s for s, _ in samples])
    d = np.array([abs(x) for _, x in samples])
    if len(set(t.tolist())) < 2:  # one duration only (the longer moves lost the board): no offset to fit
        rate = float(np.mean(d / t))
        return rate, 0.0, float(np.std(d - rate * t)) if len(d) > 2 else 0.0
    A = np.stack([t, np.ones_like(t)], 1)
    (rate, offset), *_ = np.linalg.lstsq(A, d, rcond=None)
    spread = float(np.std(d - A @ np.array([rate, offset]))) if len(d) > 2 else 0.0
    return float(rate), float(offset), spread


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scale", type=float, default=1.4, help="the board's make_board.py --scale (A3: 1.4)")
    ap.add_argument("--speed", type=float, default=0.2, help="walking speed, m/s (the policy's slowest that moves)")
    ap.add_argument("--turn-speed", type=float, default=20.0, help="turning speed, deg/s (the slowest the robot takes; 30: 1 s turns lose the board)")
    ap.add_argument("--rounds", type=int, default=2, help="how many times to repeat every move")
    ap.add_argument("--dry-run", action="store_true", help="only look and print the pose")
    args = ap.parse_args()

    try:
        http("/health")
    except Exception as e:
        print(f"no robot server at {SERVER} ({e}): start it with robot/start.sh --kiosk")
        return 1
    layout = Layout(scale=args.scale)
    looker = Looker(layout)
    p = looker.pose()
    if p is None:
        print("can't see the board: lay it flat, its near edge 10-20 cm in front of Pupper, \"Pupper's side\" towards it")
        return 1
    w, h = layout.size
    gap = p.y - h  # from the floor under the lens to the board's near edge
    print(f"Pupper's lens is {p.height_cm:.0f} cm up, {gap:.0f} cm from the board's near edge, "
          f"{p.x - w / 2:+.0f} cm to the right of its middle, turned {sp.turn_between(p.heading, sp.FACING):+.0f} deg")
    if args.dry_run:
        return 0
    if not 5 <= gap <= 30:
        print("put the board's near edge 10-20 cm in front of Pupper's lens, where it reads the board")
        return 1

    print("standing up...")
    http("/walking/activate", {})
    time.sleep(5)
    home = looker.pose()  # every pair of moves starts here: re-centred after each, so errors don't pile up
    if home is None:
        print("lost sight of the board after standing up; put it a little closer and run again")
        return 1
    samples = {axis: [] for axis in DURATIONS}
    drift = []  # how far forward moves stray sideways: the camera's yaw on the body
    cal = sp.WalkCalibration()

    def rate(axis: str) -> float:
        """The speed measured so far (cm/s or deg/s), for re-centring; the commanded one at first."""
        got = [abs(d) / t for t, d, _ in samples[axis] if t > 0]
        return statistics.median(got) if got else (args.turn_speed if axis == "turn" else args.speed * 100)

    def put_back() -> None:
        """The board is out of view: stop, and let the person put Pupper back home."""
        nonlocal home
        http("/tools/stop", {})
        while True:
            input("    lost the board: put Pupper back (board's near edge ~10 cm away, centred, square)"
                  " and press Enter (Ctrl-C stops) ")
            now = looker.pose()
            if now is not None:
                home = now
                return
            print("    still can't see the board")

    def recenter() -> None:
        """Walk back to `home`, biggest error first, at most 3 small moves."""
        for _ in range(3):
            now = looker.pose()
            if now is None:
                put_back()
                return
            fwd, left, turned = sp.moved(home, now)
            for axis, err, tol in (("turn", turned, 4.0), ("forward", fwd, 2.5), ("sideways", left, 2.5)):
                if abs(err) > tol:
                    move(axis, -1 if err > 0 else 1, max(0.2, abs(err) / rate(axis)), args.speed, args.turn_speed)
                    break
            else:
                return

    try:
        for rnd in range(args.rounds):
            for axis, durations in DURATIONS.items():
                # Forward: away from the board first, so Pupper never walks onto it.
                signs = (-1, 1) if axis == "forward" else (1, -1)
                for seconds in durations:
                    for sign in signs:
                        before = looker.pose()
                        move(axis, sign, seconds, args.speed, args.turn_speed)
                        after = looker.pose()
                        if before is None or after is None:
                            print(f"  {axis} {sign * seconds:+.2f} s: lost sight of the board; skipped")
                            put_back()
                            break  # (its return move would now walk Pupper away from home: skip it too)
                        fwd, left, turned = sp.moved(before, after)
                        got = {"forward": fwd, "sideways": left, "turn": turned}[axis]
                        samples[axis].append((seconds, sign * got, sign))
                        if axis == "forward" and abs(fwd) > 5:
                            s = 1 if fwd > 0 else -1  # a backward move strays the other way round: flip it to forward
                            drift.append(math.degrees(math.atan2(left * s, fwd * s)))
                        unit = "deg" if axis == "turn" else "cm"
                        print(f"  round {rnd + 1} {axis:8s} {sign * seconds:+.2f} s -> {got:+6.1f} {unit}"
                              f"   (also {fwd:+.1f} cm fwd, {left:+.1f} cm left, {turned:+.1f} deg)")
                    recenter()
    except KeyboardInterrupt:
        http("/tools/stop", {})
        print("\nstopped")
        return 1

    notes = {}
    old = sp.WalkCalibration.load()
    for axis, speed in (("forward", args.speed), ("sideways", args.speed), ("turn", args.turn_speed)):
        use = [(t, d) for t, d, sign in samples[axis] if axis != "forward" or sign > 0]
        if len(use) < 3:
            if axis != "forward" and old.measured:
                setattr(cal, axis, getattr(old, axis))
                print(f"not enough {axis} measurements ({len(use)}); keeping the old {axis} calibration")
                continue
            print(f"not enough {axis} measurements ({len(use)}); nothing saved")
            return 1
        if axis == "forward":  # walking back covers more than forward in short bursts: its own fit
            back = [(t, d) for t, d, sign in samples[axis] if sign < 0]
            if len(back) >= 3:
                b_rate, b_offset, b_spread = fit(back)
                if b_rate > 0:
                    cal.backward = sp.Axis(speed=speed, rate=round(b_rate, 2), offset=round(b_offset, 2))
                    notes["backward_spread_cm"] = round(b_spread, 2)
                    print(f"backward: {b_rate:5.1f} cm/s, {b_offset:+.1f} cm per move; moves vary by about {b_spread:.1f} cm")
        rate, offset, spread = fit(use)
        if rate <= 0:
            print(f"the {axis} moves didn't move Pupper (rate {rate:.1f}); nothing saved")
            return 1
        setattr(cal, axis, sp.Axis(speed=speed, rate=round(rate, 2), offset=round(offset, 2)))
        unit = "deg" if axis == "turn" else "cm"
        notes[f"{axis}_spread_{unit}"] = round(spread, 2)
        commanded = speed * (100 if axis != "turn" else 1)
        print(f"{axis:8s}: {rate:5.1f} {unit}/s (commanded {commanded:.0f}), {offset:+.1f} {unit} per move; "
              f"moves vary by about {spread:.1f} {unit}")
    cal.camera_yaw_deg = -statistics.median(drift) if drift else 0.0
    cal.measured = True
    notes.update(samples=sum(len(s) for s in samples.values()), date=time.strftime("%Y-%m-%d %H:%M"),
                 lens_height_cm=round(p.height_cm, 1))
    cal.notes = notes
    cal.save()
    print(f"camera looks {cal.camera_yaw_deg:+.1f} deg left of the walking direction")
    print(f"saved {sp.CALIBRATION}")
    worst = notes["forward_spread_cm"]
    print(f"a blind walk of up to 30 cm should land within about {2 * worst:.1f} cm (twice the forward spread);"
          " with A3 cells (7.5 cm) up to about 2.5 cm is fine")
    return 0


if __name__ == "__main__":
    sys.exit(main())
