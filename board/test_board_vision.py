#!/usr/bin/env python3
"""Desk test for board_vision.py: no robot, no camera.

Renders the printed board, with a game on it, through Pupper's fisheye lens
model (robot/camera_params.yaml) at known places (on the floor ahead at
several distances, turned, and upright like on a wall), blurs, adds noise and
JPEG-compresses it like the real camera, then checks the reader gets the grid
and the distance right, or says it can't see the board (only a confident wrong
reading fails). It shows the working range on paper; the real camera
(look.py) has the last word.

    python3 board/test_board_vision.py            # the table
    python3 board/test_board_vision.py --save DIR # also write each scene and its debug images
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from board_vision import CAMERA_PARAMS, BoardReader, unproject  # noqa: E402
from fisheye_converter import load_camera_model  # noqa: E402
from layout import Layout, render  # noqa: E402

W, H = 1400, 1050  # what the camera node publishes
LENS_RADIUS_PX = 385  # the lens's image circle on the sensor (black outside)
CAMERA_HEIGHT_CM = 15.0  # Pupper standing
CAMERA_PITCH_UP_DEG = 10.0

GAME = [["X", ".", "O"],
        [".", "X", "."],
        ["O", ".", "."]]


def board_texture(layout: Layout, ppc: float = 40.0):
    """The board with GAME drawn on: X in red stamp ink, O in black marker."""
    gray, margin = render(layout, ppc)
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for r in range(3):
        for c in range(3):
            cx, cy = layout.cell_center(r, c)
            u, v = int((cx * ppc) + margin), int((cy * ppc) + margin)
            k = int(layout.cell * 0.3 * ppc)
            t = max(2, int(layout.cell * 0.09 * ppc))
            if GAME[r][c] == "X":
                cv2.line(img, (u - k, v - k), (u + k, v + k), (30, 30, 200), t)
                cv2.line(img, (u - k, v + k), (u + k, v - k), (30, 30, 200), t)
            elif GAME[r][c] == "O":
                cv2.circle(img, (u, v), k, (40, 40, 40), t)
    return img, ppc, margin


def scene(layout: Layout, origin, ex, ey, rng) -> np.ndarray:
    """Fisheye frame of the board at origin (its (0, 0) corner) with axes ex, ey
    (unit vectors of board x and y), in the level, camera-centred world frame."""
    model = load_camera_model(CAMERA_PARAMS, W, H)
    tex, ppc, margin = board_texture(layout)
    v, u = np.mgrid[0:H, 0:W].astype(np.float64)
    rays, valid = unproject(model, u, v)
    th = math.radians(CAMERA_PITCH_UP_DEG)
    R = np.array([[1, 0, 0], [0, math.cos(th), -math.sin(th)], [0, math.sin(th), math.cos(th)]])
    world = rays @ R.T
    n = np.cross(ex, ey)
    denom = world @ n
    t = np.where(np.abs(denom) > 1e-6, (origin @ n) / np.where(denom == 0, 1, denom), -1)
    p = world * t[..., None] - origin
    bx, by = p @ ex, p @ ey
    mx = ((bx * ppc) + margin).astype(np.float32)
    my = ((by * ppc) + margin).astype(np.float32)
    hit = (t > 0) & valid
    mx[~hit], my[~hit] = -1, -1
    floor = rng.integers(90, 130, (H, W, 3)).astype(np.uint8)  # grey, speckled floor
    floor = cv2.GaussianBlur(floor, (0, 0), 2)
    img = cv2.remap(tex, mx, my, cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    on_board = cv2.remap(np.full(tex.shape[:2], 255, np.uint8), mx, my, cv2.INTER_NEAREST,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
    img = np.where(on_board[..., None], (img * 0.85).astype(np.uint8), floor)  # paper isn't pure white
    img = cv2.GaussianBlur(img, (0, 0), 1.2)  # lens blur
    img = np.clip(img + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8)  # sensor noise
    img[np.hypot(u - model.cx, v - model.cy) > LENS_RADIUS_PX] = 0
    ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return cv2.imdecode(jpg, cv2.IMREAD_COLOR)


def floor_board(layout: Layout, distance_cm: float, yaw_deg: float = 0.0, right_cm: float = 0.0):
    """Board flat on the floor, its centre distance_cm ahead, near edge towards Pupper."""
    w, h = layout.size
    a = math.radians(yaw_deg)
    ex = np.array([math.cos(a), 0.0, -math.sin(a)])  # board x: to the right
    ey = np.array([-math.sin(a), 0.0, -math.cos(a)])  # board y: down the page = towards Pupper
    centre = np.array([right_cm, CAMERA_HEIGHT_CM, distance_cm])
    return centre - ex * w / 2 - ey * h / 2, ex, ey, centre


def wall_board(layout: Layout, distance_cm: float, centre_height_cm: float = 15.0):
    """Board upright, facing Pupper, its centre at centre_height_cm above the floor."""
    w, h = layout.size
    ex, ey = np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
    centre = np.array([0.0, CAMERA_HEIGHT_CM - centre_height_cm, distance_cm])
    return centre - ex * w / 2 - ey * h / 2, ex, ey, centre


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--save", type=Path, help="write each scene and its debug images here")
    ap.add_argument("--scale", type=float, default=1.0, help="board scale (make_board.py --scale)")
    ap.add_argument("--seed", type=int, default=0, help="sensor noise seed")
    args = ap.parse_args()
    layout = Layout(scale=args.scale)
    reader = BoardReader(layout)
    rng = np.random.default_rng(args.seed)
    cases = [(f"floor {d} cm", floor_board(layout, d)) for d in (35, 45, 60, 80, 100)]
    cases += [("floor 50 cm, turned 30 deg", floor_board(layout, 50, yaw_deg=30)),
              ("floor 50 cm, 20 cm right", floor_board(layout, 50, right_cm=20)),
              ("wall 40 cm", wall_board(layout, 40)),
              ("wall 80 cm", wall_board(layout, 80))]
    th = math.radians(CAMERA_PITCH_UP_DEG)
    R = np.array([[1, 0, 0], [0, math.cos(th), -math.sin(th)], [0, math.sin(th), math.cos(th)]])
    print(f"board scale {args.scale:g}: cells {layout.cell:.1f} cm; camera {CAMERA_HEIGHT_CM:g} cm up, "
          f"pitched {CAMERA_PITCH_UP_DEG:g} deg up\n")
    print(f"{'scene':<28} {'read':<12} {'markers':<9} {'px/cell':>7} {'dist (true)':>13}  grid")
    failures = 0
    for name, (origin, ex, ey, centre) in cases:
        frame = scene(layout, origin, ex, ey, rng)
        reading = reader.read(frame)
        true_dist = float(np.linalg.norm(R.T @ centre))
        dist_ok = reading.board_distance_cm is not None and abs(reading.board_distance_cm - true_dist) < 0.1 * true_dist
        wrong = any(m not in ("?", want) for row, want_row in zip(reading.grid, GAME) for m, want in zip(row, want_row))
        if not reading.found:
            status = "out of range"
        elif wrong or not dist_ok:
            status = "WRONG"  # the only failure: a confident reading that is not true
        elif reading.grid != GAME:
            status = "unsure"
        else:
            status = "ok"
        failures += status == "WRONG"
        dist = f"{reading.board_distance_cm:.0f} ({true_dist:.0f})" if reading.board_distance_cm else f"- ({true_dist:.0f})"
        print(f"{name:<28} {status:<12} {str(reading.markers):<9} {reading.native_px_per_cell:7.1f} {dist:>13}  "
              f"{reading.grid_text() if reading.found else reading.message}")
        if args.save:
            args.save.mkdir(parents=True, exist_ok=True)
            stem = name.replace(" ", "_").replace(",", "")
            cv2.imwrite(str(args.save / f"{stem}_frame.jpg"), frame)
            for k, img in reading.debug.items():
                cv2.imwrite(str(args.save / f"{stem}_{k}.jpg"), img)
    print(f"\nexpected grid: {' / '.join(' '.join(r) for r in GAME)}")
    print("no wrong readings (out of range and unsure are fine: Pupper says it can't see)"
          if not failures else f"{failures} WRONG reading(s): the reader claimed a board that isn't there")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
