"""Reading the tic-tac-toe board from one fisheye frame.

    from board_vision import BoardReader
    reading = BoardReader().read(frame)        # frame: BGR, straight from the camera
    reading.grid                               # [["X", ".", "O"], ...], row 0 = far row
    reading.cells[r][c].camera_xyz_cm          # where a cell is, for aiming the stamp

How it works, step by step (each step is a debug image, see look.py):
  1. Find the board's ArUco markers in the raw fisheye frame (where they are
     small and bent), or, if none show up there, in a few zoomed views.
  2. Point a "virtual camera" at them: a flat, undistorted, zoomed-in view
     rendered from the full-resolution fisheye frame with the lens model
     (robot/camera_params.yaml). Find the markers again there, now straight-edged.
  3. The marker corners, whose places on the board are known (layout.py), give
     the homography from that view to the board, and solvePnP gives the board's
     pose: how far away and which way it lies.
  4. Warp the board to a top-down image and score each cell by how much ink
     is inside it, away from the grid lines: empty, X (ink in the middle) or
     O (a ring, empty middle).

Nothing here talks to ROS, so it runs at a desk on saved images.

Camera frame: x right, y down, z forward out of the lens (OpenCV's), in cm.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "robot"))
from fisheye_converter import DoubleSphereModel, load_camera_model  # noqa: E402

from layout import ARUCO_DICT, MARKER_IDS, Layout  # noqa: E402

CAMERA_PARAMS = HERE.parent / "robot" / "camera_params.yaml"

TOPDOWN_PX_PER_CM = 20.0  # the top-down board image
INK_CONTRAST = 0.9  # a pixel this much darker than the paper around it is ink
EMPTY_BELOW = 0.05  # a cell with less ink than this (fraction of its inside) is empty
SHAPE_INSET = 0.1  # the shape is judged over more of the cell than the empty test (marks wander)
O_HOLE_ABOVE = 0.015  # paper enclosed by ink, as a fraction of the cell's area: an O
X_CENTRE_ABOVE = 0.6  # ink at a mark's own centroid: an X's crossing (Xs ~0.9+, Os ~0.35 or less)
CELL_INSET = 0.2  # ignore this fraction of the cell on each side (grid lines, misalignment)
MIN_CELL_PX = 11  # a cell smaller than this in the raw frame (either way) is too blurry to call: "?"


# ----------------------------------------------------------------------------- lens
def unproject(model: DoubleSphereModel, u, v):
    """Pixels -> unit rays (x, y, z) in the camera frame, and which pixels are inside the lens."""
    mx = (np.asarray(u, np.float64) - model.cx) / model.fx
    my = (np.asarray(v, np.float64) - model.cy) / model.fy
    r2 = mx * mx + my * my
    a, xi = model.alpha, model.xi
    valid = np.ones_like(r2, bool) if a <= 0.5 else r2 <= 1.0 / (2 * a - 1)
    r2c = np.where(valid, r2, 0.0)
    mz = (1 - a * a * r2c) / (a * np.sqrt(np.maximum(1 - (2 * a - 1) * r2c, 0)) + 1 - a)
    k = (mz * xi + np.sqrt(mz * mz + (1 - xi * xi) * r2c)) / (mz * mz + r2c)
    ray = np.stack([k * mx, k * my, k * mz - xi], axis=-1)
    ray /= np.linalg.norm(ray, axis=-1, keepdims=True)
    return ray, valid


@dataclass
class VirtualCamera:
    """A pinhole camera looking along a ray of the fisheye: rotation R (its axes
    in the camera frame, as columns), focal length f in px, square size px."""

    R: np.ndarray
    f: float
    size: int

    @staticmethod
    def looking_at(direction, fov_deg: float, size: int) -> "VirtualCamera":
        z = np.asarray(direction, np.float64)
        z /= np.linalg.norm(z)
        down = np.array([0.0, 1.0, 0.0]) if abs(z[1]) < 0.95 else np.array([0.0, 0.0, 1.0])
        x = np.cross(down, z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        f = (size / 2) / math.tan(math.radians(fov_deg) / 2)
        return VirtualCamera(np.stack([x, y, z], axis=1), f, size)

    @property
    def K(self) -> np.ndarray:
        c = (self.size - 1) / 2
        return np.array([[self.f, 0, c], [0, self.f, c], [0, 0, 1]], np.float64)

    def rays(self, px: np.ndarray) -> np.ndarray:
        """Virtual pixels (..., 2) -> rays in the camera frame (..., 3)."""
        c = (self.size - 1) / 2
        local = np.stack([(px[..., 0] - c) / self.f, (px[..., 1] - c) / self.f, np.ones(px.shape[:-1])], -1)
        return local @ self.R.T

    def project(self, rays: np.ndarray) -> np.ndarray:
        """Rays in the camera frame (..., 3) -> virtual pixels (..., 2)."""
        local = rays @ self.R
        c = (self.size - 1) / 2
        return np.stack([self.f * local[..., 0] / local[..., 2] + c, self.f * local[..., 1] / local[..., 2] + c], -1)

    def render(self, frame: np.ndarray, model: DoubleSphereModel) -> np.ndarray:
        v, u = np.mgrid[0:self.size, 0:self.size].astype(np.float64)
        return remap_rays(frame, model, self.rays(np.stack([u, v], -1)))


def remap_rays(frame: np.ndarray, model: DoubleSphereModel, rays: np.ndarray) -> np.ndarray:
    """Sample the fisheye frame along the given rays (h, w, 3) -> image (h, w)."""
    u, v, valid = model.project(rays[..., 0], rays[..., 1], rays[..., 2])
    u = np.where(valid, u, -1).astype(np.float32)
    v = np.where(valid, v, -1).astype(np.float32)
    return cv2.remap(frame, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _shape(ink: np.ndarray, cell_px: float) -> tuple[float, float]:
    """(hole, centre ink) of the mark in one cell's ink mask.

    hole: paper the ink closes in, as a fraction of the cell's area: an O
    encloses its middle wherever it is drawn; an X encloses nothing (but its
    arms and a blurred grid line can close a sliver, so the centre ink decides
    first). The mask's edge is cleared, so strokes can't close a hole with the
    grid lines there. centre ink: how much ink is near the mark's own centroid (not the
    cell's centre): high where an X's strokes cross, low in an O's middle.
    """
    if not ink.any():
        return 0.0, 0.0
    k = max(3, int(cell_px * 0.06) | 1)  # seal small gaps a blurred ring breaks into
    m = cv2.morphologyEx(ink.astype(np.uint8) * 255, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    e = max(2, int(cell_px * 0.05))  # grid lines blur into the crop's edge: don't let them close holes
    m[:e, :], m[-e:, :], m[:, :e], m[:, -e:] = 0, 0, 0, 0
    outside = cv2.copyMakeBorder(255 - m, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=255)
    cv2.floodFill(outside, None, (0, 0), 0)  # paper reachable from the edge isn't enclosed
    hole = float(np.count_nonzero(outside)) / (cell_px * cell_px)
    ys, xs = np.nonzero(ink)
    r = max(2, int(cell_px * 0.1))
    cy, cx = int(ys.mean()), int(xs.mean())
    mid = ink[max(0, cy - r):cy + r, max(0, cx - r):cx + r]
    return hole, float(mid.mean()) if mid.size else 0.0


# ----------------------------------------------------------------------------- results
@dataclass
class Cell:
    row: int
    col: int
    mark: str  # "X", "O", "." (empty) or "?" (ink, but neither shape)
    ink: float  # fraction of the cell's inside that is ink
    center_ink: float  # ... of its middle (high for X, low for O)
    camera_xyz_cm: tuple[float, float, float] | None = None  # cell centre, camera frame
    fisheye_px: tuple[float, float] | None = None  # cell centre in the raw frame
    native_px: float = 0.0  # the cell's smaller side in the raw frame
    hole: float = 0.0  # paper enclosed by the mark, fraction of the cell (an O has one)

    @property
    def bearing_deg(self) -> tuple[float, float] | None:
        """(right, down) angles of the cell centre from straight ahead."""
        if self.camera_xyz_cm is None:
            return None
        x, y, z = self.camera_xyz_cm
        return math.degrees(math.atan2(x, z)), math.degrees(math.atan2(y, math.hypot(x, z)))


@dataclass
class Reading:
    found: bool
    message: str
    markers: list[int] = field(default_factory=list)  # marker ids used
    cells: list[list[Cell]] = field(default_factory=list)
    native_px_per_cell: float = 0.0  # how big a cell is in the raw frame: the real resolution
    fit_error_cm: float = 0.0  # how well markers and grid crossings agree on the board's place
    board_distance_cm: float | None = None
    # The board's pose in the camera frame (cm): a board point (x, y) is at board_R @ (x, y, 0) + board_t.
    board_R: np.ndarray | None = None
    board_t: np.ndarray | None = None
    debug: dict[str, np.ndarray] = field(default_factory=dict)  # name -> image

    @property
    def grid(self) -> list[list[str]]:
        return [[c.mark for c in row] for row in self.cells]

    def grid_text(self) -> str:
        return " / ".join(" ".join(row) for row in self.grid) if self.cells else "(no board)"

    def as_dict(self) -> dict:
        return {
            "found": self.found,
            "message": self.message,
            "grid": self.grid,
            "markers": self.markers,
            "native_px_per_cell": round(self.native_px_per_cell, 1),
            "fit_error_cm": round(self.fit_error_cm, 2),
            "board_distance_cm": None if self.board_distance_cm is None else round(self.board_distance_cm, 1),
            "cells": [
                {
                    "row": c.row, "col": c.col, "mark": c.mark,
                    "ink": round(c.ink, 3), "center_ink": round(c.center_ink, 3), "hole": round(c.hole, 3),
                    "camera_xyz_cm": None if c.camera_xyz_cm is None else [round(v, 1) for v in c.camera_xyz_cm],
                    "bearing_deg": None if c.bearing_deg is None else [round(v, 1) for v in c.bearing_deg],
                    "fisheye_px": None if c.fisheye_px is None else [round(v, 1) for v in c.fisheye_px],
                    "native_px": round(c.native_px, 1),
                }
                for row in self.cells for c in row
            ],
        }


# ----------------------------------------------------------------------------- the reader
class BoardReader:
    def __init__(self, layout: Layout | None = None, camera_params: Path = CAMERA_PARAMS):
        self.layout = layout or Layout()
        self.camera_params = camera_params
        self.model: DoubleSphereModel | None = None
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        params.minMarkerPerimeterRate = 0.01  # markers are small in a 1400 px fisheye frame
        params.adaptiveThreshWinSizeMax = 53
        self.detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(ARUCO_DICT), params)

    # Where to look when the raw frame shows no marker: ahead, and down towards the floor.
    SEARCH_VIEWS = [(0, 0), (0, 35), (0, 65), (-40, 30), (40, 30)]  # (right, down) degrees

    def _detect(self, gray: np.ndarray) -> dict[int, np.ndarray]:
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None:
            return {}
        return {int(i): c.reshape(4, 2).astype(np.float64)
                for i, c in zip(ids.ravel(), corners) if int(i) in MARKER_IDS}

    def read(self, frame: np.ndarray) -> Reading:
        h, w = frame.shape[:2]
        if self.model is None or (self.model.width, self.model.height) != (w, h):
            self.model = load_camera_model(self.camera_params, w, h)
        model = self.model
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

        # 1. markers in the raw frame, else in zoomed views. Each marker is kept as
        # the rays to its 4 corners, so a view that loses it still knows where it is.
        known: dict[int, np.ndarray] = {}
        raw = self._detect(gray)
        for mid, c in raw.items():
            known[mid] = unproject(model, c[:, 0], c[:, 1])[0]
        if not known:
            for right, down in self.SEARCH_VIEWS:
                d = [math.sin(math.radians(right)), math.sin(math.radians(down)), 1.0]
                cam = VirtualCamera.looking_at(d, 70, 600)
                known = {mid: cam.rays(c) for mid, c in self._detect(cam.render(gray, model)).items()}
                if known:
                    break
            if not known:
                return Reading(False, "no board markers in view", debug={"raw": frame})

        # 2. a virtual camera on the board: first around the markers, then around
        # the whole board as they place it (they may all sit on one edge)
        rays = np.concatenate(list(known.values()))
        cam = self._aim(rays, pad=1.8 if len(known) < 4 else 1.3)
        view = cam.render(gray, model)
        for mid, c in self._detect(view).items():
            known[mid] = cam.rays(c)  # straight edges here: better corners
        H = self._fit(cam, known)
        if H is None:
            return Reading(False, "could not fit the board", markers=sorted(known), debug={"raw": frame, "view": view})
        bw, bh = self.layout.size
        outline = np.array([[[0, 0]], [[bw, 0]], [[bw, bh]], [[0, bh]]], np.float64)
        cam = self._aim(cam.rays(cv2.perspectiveTransform(outline, np.linalg.inv(H)).reshape(-1, 2)), pad=1.1)
        view = cam.render(gray, model)
        for mid, c in self._detect(view).items():
            known[mid] = cam.rays(c)
        found = {mid: cam.project(r) for mid, r in known.items()}

        # 3. homography (view px -> board cm), locked onto the grid crossings too,
        # which the far side of a board on the floor needs: its markers are thin there
        img_pts = np.concatenate([found[i] for i in sorted(found)])
        board_pts = np.concatenate([self.layout.marker_corners(i) for i in sorted(found)])
        H = self._fit(cam, known)
        if H is None:
            return Reading(False, "could not fit the board", markers=sorted(found), debug={"raw": frame, "view": view})
        if len(found) < 2:  # one marker places the far cells too loosely (they read wrong in tests)
            return Reading(False, "only one board marker in view; turn towards the board or move back",
                           markers=sorted(found), debug={"raw": frame, "view": view})
        markers_img, markers_board = img_pts, board_pts
        crossings = 0
        for _ in range(3):
            xs_img, xs_board = self._find_crossings(self._topdown(gray, model, cam, H), H)
            if len(xs_img) < 2:
                break
            crossings = len(xs_img)
            img_pts = np.concatenate([markers_img, xs_img])
            board_pts = np.concatenate([markers_board, xs_board])
            H, _ = cv2.findHomography(img_pts, board_pts, 0)
        fitted = cv2.perspectiveTransform(img_pts.reshape(-1, 1, 2), H).reshape(-1, 2)
        fit_cm = float(np.sqrt(np.mean(np.sum((fitted - board_pts) ** 2, axis=1))))

        obj = np.hstack([board_pts, np.zeros((len(board_pts), 1))])
        ok, rvec, tvec = cv2.solvePnP(obj, img_pts, cam.K, None, flags=cv2.SOLVEPNP_IPPE)
        Rb = cv2.Rodrigues(rvec)[0] if ok else None

        def board_to_camera(xy) -> np.ndarray:  # board cm -> camera-frame cm
            p = Rb @ np.array([xy[0], xy[1], 0.0]) + tvec.ravel()
            return cam.R @ p

        def board_to_fisheye(xy) -> tuple[float, float]:
            vp = cv2.perspectiveTransform(np.array([[xy]], np.float64), np.linalg.inv(H)).reshape(2)
            ray = cam.rays(vp[None])[0]
            u, v, _ = model.project(ray[0], ray[1], ray[2])
            return float(u), float(v)

        # 4. top-down board, straight from the raw frame (one interpolation), in colour
        top = self._topdown(frame, model, cam, H)
        cells = self._score(top)
        for row in cells:
            for c in row:
                xy = self.layout.cell_center(c.row, c.col)
                c.fisheye_px = board_to_fisheye(xy)
                half = self.layout.cell / 2
                (lx, ly), (rx, ry), (tx_, ty_), (bx, by) = (board_to_fisheye((xy[0] + dx, xy[1] + dy))
                                                            for dx, dy in ((-half, 0), (half, 0), (0, -half), (0, half)))
                c.native_px = min(math.hypot(rx - lx, ry - ly), math.hypot(bx - tx_, by - ty_))
                if c.native_px < MIN_CELL_PX:
                    c.mark = "?"  # too few real pixels to tell an X from an empty cell
                if Rb is not None:
                    c.camera_xyz_cm = tuple(float(v) for v in board_to_camera(xy))

        # The real resolution: a cell's width in the raw frame.
        a, b = board_to_fisheye(self.layout.cell_center(1, 0)), board_to_fisheye(self.layout.cell_center(1, 2))
        native = math.dist(a, b) / 2
        centre = cells[1][1].camera_xyz_cm
        reading = Reading(
            True, f"board read from {len(found)} marker(s) and {crossings} grid crossing(s)", markers=sorted(found), cells=cells,
            native_px_per_cell=native, fit_error_cm=fit_cm,
            board_distance_cm=None if centre is None else float(np.linalg.norm(centre)),
            board_R=None if Rb is None else cam.R @ Rb, board_t=None if Rb is None else cam.R @ tvec.ravel(),
        )
        reading.debug = {"raw": self._draw_raw(frame, raw, cells), "view": view, "topdown": self._draw_top(top, cells)}
        return reading

    # ------------------------------------------------------------------------- views
    @staticmethod
    def _aim(rays: np.ndarray, pad: float) -> VirtualCamera:
        """A virtual camera centred on these rays, wide enough for all of them."""
        direction = rays.mean(axis=0)
        direction /= np.linalg.norm(direction)
        spread = max(math.degrees(math.acos(np.clip(r @ direction, -1, 1))) for r in rays)
        fov = float(np.clip(2 * spread * pad + 4, 20, 140))
        size = int(np.clip(fov * 8, 300, 1200))  # ~2x the lens's own px/deg; more only blurs
        return VirtualCamera.looking_at(direction, fov, size)

    def _fit(self, cam: VirtualCamera, known: dict[int, np.ndarray]) -> np.ndarray | None:
        """Homography view px -> board cm from the markers' corners."""
        img = np.concatenate([cam.project(known[i]) for i in sorted(known)])
        board = np.concatenate([self.layout.marker_corners(i) for i in sorted(known)])
        H, _ = cv2.findHomography(img, board, 0)
        return H

    def _topdown(self, frame: np.ndarray, model: DoubleSphereModel, cam: VirtualCamera, H: np.ndarray) -> np.ndarray:
        """The board seen from straight above, TOPDOWN_PX_PER_CM, sampled from the raw frame."""
        bw, bh = self.layout.size
        tw, th = int(bw * TOPDOWN_PX_PER_CM), int(bh * TOPDOWN_PX_PER_CM)
        ty, tx = np.mgrid[0:th, 0:tw].astype(np.float64)
        board_xy = np.stack([(tx + 0.5) / TOPDOWN_PX_PER_CM, (ty + 0.5) / TOPDOWN_PX_PER_CM], -1)
        view_xy = cv2.perspectiveTransform(board_xy.reshape(-1, 1, 2), np.linalg.inv(H)).reshape(th, tw, 2)
        return remap_rays(frame, model, cam.rays(view_xy))

    # ------------------------------------------------------------------------- grid crossings
    def _find_crossings(self, top: np.ndarray, H: np.ndarray):
        """Where the 4 grid-line crossings really are, from the top-down image made
        with the current guess H (view px -> board cm): (view px, board cm) pairs.
        A crossing found more than a quarter cell away is ignored (likely a mark)."""
        p = TOPDOWN_PX_PER_CM
        cell = self.layout.cell * p
        t = max(3, int(self.layout.s(self.layout.line) * p))
        half = int(cell * 0.4)  # the template: a dark + on paper
        tpl = np.full((2 * half + 1, 2 * half + 1), 255, np.uint8)
        tpl[half - t // 2:half + t // 2 + 1, :] = 0
        tpl[:, half - t // 2:half + t // 2 + 1] = 0
        search = int(cell * 0.25)
        gx, gy = self.layout.grid_origin
        img, board = [], []
        for i in (1, 2):
            for j in (1, 2):
                bx, by = gx + i * self.layout.cell, gy + j * self.layout.cell
                u, v = int(bx * p), int(by * p)
                x0, y0 = u - half - search, v - half - search
                x1, y1 = u + half + search + 1, v + half + search + 1
                if x0 < 0 or y0 < 0 or x1 > top.shape[1] or y1 > top.shape[0]:
                    continue
                score = cv2.matchTemplate(top[y0:y1, x0:x1], tpl, cv2.TM_CCOEFF_NORMED)
                _, best, _, (mx, my) = cv2.minMaxLoc(score)
                if best < 0.45:
                    continue
                seen = np.array([[[(x0 + mx + half) / p, (y0 + my + half) / p]]])  # where H puts it, cm
                img.append(cv2.perspectiveTransform(seen, np.linalg.inv(H)).reshape(2))
                board.append([bx, by])
        return np.array(img, np.float64).reshape(-1, 2), np.array(board, np.float64).reshape(-1, 2)

    # ------------------------------------------------------------------------- cells
    def _score(self, top: np.ndarray) -> list[list[Cell]]:
        gray = cv2.cvtColor(top, cv2.COLOR_BGR2GRAY).astype(np.float64) if top.ndim == 3 else top.astype(np.float64)
        sat = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)[..., 1].astype(np.float64) if top.ndim == 3 else np.zeros_like(gray)
        gx, gy = self.layout.grid_origin
        g = self.layout.s(self.layout.grid)
        p = TOPDOWN_PX_PER_CM
        # Ink is darker than the paper right around it. Far away the camera blurs
        # even a thick black stroke to a light grey (the grid lines come out ~15%
        # darker than the paper at 45 cm), so compare with the local paper, not black.
        k = int(1.5 * p) | 1  # wider than any blurred stroke, so the paper's level shows through
        paper = cv2.GaussianBlur(cv2.dilate(gray, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))), (0, 0), k / 3)
        paper_sat = np.percentile(sat[int(gy * p):int((gy + g) * p), int(gx * p):int((gx + g) * p)], 50)
        ink = (gray < INK_CONTRAST * paper) | ((sat > paper_sat + 70) & (gray < 0.95 * paper))  # dark, or coloured ink

        cells = []
        for r in range(3):
            row = []
            for c in range(3):
                cx, cy = self.layout.cell_center(r, c)
                half = self.layout.cell / 2
                inner = half * (1 - 2 * CELL_INSET)
                x0, x1 = int((cx - inner) * p), int((cx + inner) * p)
                y0, y1 = int((cy - inner) * p), int((cy + inner) * p)
                patch = ink[y0:y1, x0:x1]
                frac = float(patch.mean()) if patch.size else 0.0
                # The shape, over more of the cell: hand-drawn marks are rarely centred.
                wide = half * (1 - 2 * SHAPE_INSET)
                shape = ink[int((cy - wide) * p):int((cy + wide) * p), int((cx - wide) * p):int((cx + wide) * p)]
                hole, mid_frac = _shape(shape, self.layout.cell * p)
                if frac < EMPTY_BELOW:
                    mark = "."
                elif mid_frac > X_CENTRE_ABOVE:  # ink at the mark's own centre: the strokes cross
                    mark = "X"
                elif hole > O_HOLE_ABOVE:  # a ring around paper: wherever it sits, it's an O
                    mark = "O"
                else:
                    mark = "?"
                row.append(Cell(r, c, mark, frac, mid_frac, hole=hole))
            cells.append(row)
        return cells

    # ------------------------------------------------------------------------- debug images
    def _draw_raw(self, frame, raw, cells):
        out = frame.copy() if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        for mid, c in raw.items():
            cv2.polylines(out, [c.astype(np.int32)], True, (0, 255, 0), 2)
            cv2.putText(out, str(mid), tuple(c[0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        for row in cells:
            for c in row:
                if c.fisheye_px:
                    u, v = (int(round(x)) for x in c.fisheye_px)
                    cv2.circle(out, (u, v), 4, (0, 0, 255), -1)
                    cv2.putText(out, c.mark, (u + 5, v - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return out

    def _draw_top(self, top, cells):
        out = top.copy() if top.ndim == 3 else cv2.cvtColor(top, cv2.COLOR_GRAY2BGR)
        p = TOPDOWN_PX_PER_CM
        for row in cells:
            for c in row:
                cx, cy = self.layout.cell_center(c.row, c.col)
                inner = self.layout.cell / 2 * (1 - 2 * CELL_INSET)
                cv2.rectangle(out, (int((cx - inner) * p), int((cy - inner) * p)),
                              (int((cx + inner) * p), int((cy + inner) * p)), (255, 128, 0), 1)
                cv2.putText(out, f"{c.mark} {c.ink:.2f}", (int((cx - inner) * p), int((cy + inner) * p) - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        return out
