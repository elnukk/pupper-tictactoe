"""The tic-tac-toe board: where everything on the printed page is, in cm.

One description of the board that make_board.py prints and board_vision.py
reads, so the two always agree.

    marker 0 ┌──┐            ┌──┐ marker 1        far edge (away from Pupper)
             └──┘            └──┘
                  │    │
               ───┼────┼───
                  │    │          3 x 3 grid, row 0 at the top
               ───┼────┼───
                  │    │
             ┌──┐            ┌──┐
    marker 3 └──┘            └──┘ marker 2        near edge (Pupper's side)

Coordinates: x to the right, y down the page, origin at marker 0's top-left
corner. The four ArUco markers (DICT_4X4_50, ids 0-3) are what the camera finds;
from any one of them the vision knows where every cell is. Lay the board with
the near edge (the printed "Pupper" line) towards the robot, so row 0 is the
far row and column 0 is on Pupper's left.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

ARUCO_DICT = cv2.aruco.DICT_4X4_50
MARKER_IDS = (0, 1, 2, 3)  # top-left, top-right, bottom-right, bottom-left


@dataclass(frozen=True)
class Layout:
    """A board, in cm. scale=1 fits US Letter / A4; scale=2 is a 4x poster."""

    scale: float = 1.0
    width: float = 20.0  # the rectangle the four markers span
    height: float = 26.0
    marker: float = 4.2  # side of a marker's black square
    grid: float = 16.0  # side of the 3 x 3 grid
    line: float = 0.6  # grid line thickness

    def s(self, v: float) -> float:
        return v * self.scale

    @property
    def size(self) -> tuple[float, float]:
        return self.s(self.width), self.s(self.height)

    @property
    def cell(self) -> float:
        return self.s(self.grid) / 3

    @property
    def grid_origin(self) -> tuple[float, float]:
        """Top-left corner of the grid."""
        w, h = self.size
        return (w - self.s(self.grid)) / 2, (h - self.s(self.grid)) / 2

    def marker_origin(self, marker_id: int) -> tuple[float, float]:
        w, h = self.size
        m = self.s(self.marker)
        return {0: (0.0, 0.0), 1: (w - m, 0.0), 2: (w - m, h - m), 3: (0.0, h - m)}[marker_id]

    def marker_corners(self, marker_id: int) -> np.ndarray:
        """The marker's 4 corners in the order OpenCV reports them: TL, TR, BR, BL."""
        x, y = self.marker_origin(marker_id)
        m = self.s(self.marker)
        return np.array([[x, y], [x + m, y], [x + m, y + m], [x, y + m]], np.float64)

    def cell_center(self, row: int, col: int) -> tuple[float, float]:
        gx, gy = self.grid_origin
        return gx + (col + 0.5) * self.cell, gy + (row + 0.5) * self.cell

    def grid_lines(self) -> list[tuple[float, float, float, float]]:
        """The 4 grid lines as filled rectangles (x, y, w, h)."""
        gx, gy = self.grid_origin
        g, t = self.s(self.grid), self.s(self.line)
        out = []
        for i in (1, 2):
            out.append((gx + i * self.cell - t / 2, gy, t, g))  # vertical
            out.append((gx, gy + i * self.cell - t / 2, g, t))  # horizontal
        return out

    def marker_modules(self, marker_id: int) -> list[tuple[float, float, float, float]]:
        """The marker's black squares as rectangles (x, y, w, h), border included."""
        bits = marker_bits(marker_id)  # 6 x 6, 1 = black
        n = bits.shape[0]
        x0, y0 = self.marker_origin(marker_id)
        q = self.s(self.marker) / n
        return [(x0 + c * q, y0 + r * q, q, q) for r in range(n) for c in range(n) if bits[r, c]]


def marker_bits(marker_id: int) -> np.ndarray:
    """A 4x4 marker with its one-module black border: 6 x 6, 1 = black."""
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    img = cv2.aruco.generateImageMarker(dictionary, marker_id, 6, borderBits=1)
    return (img < 128).astype(np.uint8)


def render(layout: Layout, px_per_cm: float = 40.0, margin_cm: float = 1.0) -> tuple[np.ndarray, float]:
    """The board as a grayscale image, white paper with black ink.

    Returns (image, margin_px): board cm (x, y) is at pixel
    ((x + margin_cm) * px_per_cm, (y + margin_cm) * px_per_cm).
    """
    w, h = layout.size
    img = np.full((int(round((h + 2 * margin_cm) * px_per_cm)), int(round((w + 2 * margin_cm) * px_per_cm))), 255, np.uint8)

    def fill(x, y, rw, rh):
        x0, y0 = int(round((x + margin_cm) * px_per_cm)), int(round((y + margin_cm) * px_per_cm))
        x1, y1 = int(round((x + rw + margin_cm) * px_per_cm)), int(round((y + rh + margin_cm) * px_per_cm))
        img[y0:y1, x0:x1] = 0

    for rect in layout.grid_lines():
        fill(*rect)
    for mid in MARKER_IDS:
        for rect in layout.marker_modules(mid):
            fill(*rect)
    return img, margin_cm * px_per_cm
