#!/usr/bin/env python3
"""Where a stamp really landed: the new ink between two readings of the board.

Pupper reads the board from straight above (Reading.top, board_vision) before a
stamping trip and again after it. Ink that is in the second picture but not in
the first, near the cell it aimed for, is the stamp's X: its middle, against the
cell's middle, is how far the stamp missed, in cm on the board.

    from landing import find_landing
    hit = find_landing(before.top, after.top, layout, target_xy)   # None: no new ink found
    hit.short_cm, hit.right_cm                                       # + = short of / right of the target
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from board_vision import INK_CONTRAST, TOPDOWN_PX_PER_CM
from layout import Layout

SEARCH_CELLS = 1.6  # look this many cells around the target (a miss by a whole cell is still found)
MIN_INK_CM2 = 1.5  # less new ink than this isn't a stamp (noise, a shadow's edge)
ALIGN_CM = 0.6  # the two readings place the board this close: old ink grown by this much doesn't count as new


@dataclass
class Landing:
    x: float  # the X's middle, board cm (x to Pupper's right, y towards Pupper)
    y: float
    short_cm: float  # how far short of the target (towards Pupper); negative: past it
    right_cm: float  # how far to Pupper's right of the target; negative: left of it
    ink_cm2: float  # how much new ink: a whole stamp, or only part of one

    def describe(self) -> str:
        fb = f"{abs(self.short_cm):.1f} cm {'short' if self.short_cm > 0 else 'past'}"
        lr = f"{abs(self.right_cm):.1f} cm {'right' if self.right_cm > 0 else 'left'}"
        return f"landed {fb}, {lr} of the cell's middle"


def ink_mask(top: np.ndarray) -> np.ndarray:
    """Ink in a top-down board image: darker than the paper right around it (as board_vision scores cells)."""
    gray = cv2.cvtColor(top, cv2.COLOR_BGR2GRAY).astype(np.float64) if top.ndim == 3 else top.astype(np.float64)
    k = int(1.5 * TOPDOWN_PX_PER_CM) | 1
    paper = cv2.GaussianBlur(cv2.dilate(gray, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))), (0, 0), k / 3)
    return (gray < INK_CONTRAST * paper) & (gray > 0)  # (0: outside the camera's frame)


def find_landing(before: np.ndarray | None, after: np.ndarray | None, layout: Layout,
                 target: tuple[float, float]) -> Landing | None:
    if before is None or after is None or before.shape != after.shape:
        return None
    p = TOPDOWN_PX_PER_CM
    grow = max(1, int(round(ALIGN_CM * p)))
    old = cv2.dilate(ink_mask(before).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1,) * 2))
    new = ink_mask(after) & (old == 0)
    new = cv2.morphologyEx(new.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    tx, ty = target
    r = SEARCH_CELLS * layout.cell
    x0, x1 = max(0, int((tx - r) * p)), min(new.shape[1], int((tx + r) * p))
    y0, y1 = max(0, int((ty - r) * p)), min(new.shape[0], int((ty + r) * p))
    win = new[y0:y1, x0:x1]
    n, labels, stats, _ = cv2.connectedComponentsWithStats(win, connectivity=8)
    if n <= 1:
        return None
    # The stamp is the biggest new blob, with any blobs right next to it (an X broken by a grid line).
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    keep = np.zeros_like(win, dtype=bool)
    bx, by, bw, bh = (stats[big, i] for i in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
    pad = int(0.25 * layout.cell * p)
    for i in range(1, n):
        x, y, w, h = (stats[i, j] for j in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
        if x < bx + bw + pad and x + w > bx - pad and y < by + bh + pad and y + h > by - pad:
            keep |= labels == i
    area = float(keep.sum()) / (p * p)
    if area < MIN_INK_CM2:
        return None
    ys, xs = np.nonzero(keep)
    lx, ly = (x0 + xs.mean() + 0.5) / p, (y0 + ys.mean() + 0.5) / p
    return Landing(lx, ly, short_cm=ly - ty, right_cm=lx - tx, ink_cm2=area)
