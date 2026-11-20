"""Where Pupper stands on the board, and how to walk its stamp over a cell. Plain Python.

Everything is in the board's own frame, in cm, on the floor: x to the right
along the printed board (Pupper's right when it faces the board the right way),
y down the page, towards "Pupper's side". So Pupper faces the board squarely
when its heading is (0, -1).

  pose_from_board(R, t)     the camera's place and Pupper's heading on the board,
                            from a reading's board pose
  plan(pose, cell, config)  the turn, sideways step and forward walk that put
                            the stamp over the cell
  WalkCalibration           how far a timed walk really goes (calibrate_walk.py
                            measures it, using the board as a ruler)

Directions follow the robot: forward, left positive, turns counter-clockwise
(to the left) positive.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
STAMP_CONFIG = HERE / "stamp.yaml"
CALIBRATION = HERE / "walk_calibration.yaml"


# ----------------------------------------------------------------------------- config
@dataclass
class StampConfig:
    """stamp.yaml: where the stamp is, and how carefully to aim."""

    enabled: bool = False  # true once the stamp is mounted and the walk is calibrated
    stamp_behind_camera_cm: float = 10.0  # the stamp's centre, behind the lens along Pupper's centreline
    stamp_left_of_camera_cm: float = 0.0  # ... and to Pupper's left of it (right: negative)
    crouch_trick: str = "push_up"  # the trick that presses the stamp down, and ends standing
    blind_max_cm: float = 30.0  # the longest forward walk without a fresh look
    blind_target_cm: float = 22.0  # when farther than blind_max, walk until this much is left, then look
    lateral_tolerance_cm: float = 1.5  # line the stamp up with the cell to within this, sideways
    yaw_tolerance_deg: float = 4.0  # and face the board squarely to within this
    max_steps: int = 8  # looks before giving up
    max_walk_cm: float = 150.0  # a stamping trip never walks further than this in total

    @staticmethod
    def load(path: Path = STAMP_CONFIG) -> "StampConfig":
        data = yaml.safe_load(path.read_text()) if path.exists() else {}
        known = StampConfig.__dataclass_fields__
        return StampConfig(**{k: v for k, v in (data or {}).items() if k in known})


@dataclass
class Axis:
    """One way of walking: commanded at `speed`, it covers rate * seconds + offset
    (cm, or degrees for turning). offset is what starting and stopping add (or,
    negative, lose)."""

    speed: float  # m/s, or deg/s for turning
    rate: float  # cm/s, or deg/s, really covered
    offset: float = 0.0

    def seconds_for(self, amount: float) -> float:
        """How long to walk to cover |amount|; 0 for an amount too small to walk."""
        need = abs(amount) - self.offset
        return max(0.0, need / self.rate) if self.rate > 0 else 0.0


@dataclass
class WalkCalibration:
    """walk_calibration.yaml; until it is measured, the commanded speeds are taken at their word."""

    forward: Axis = field(default_factory=lambda: Axis(0.2, 20.0))
    sideways: Axis = field(default_factory=lambda: Axis(0.2, 20.0))
    turn: Axis = field(default_factory=lambda: Axis(30.0, 30.0))
    camera_yaw_deg: float = 0.0  # how far the camera looks left of Pupper's walking direction
    measured: bool = False
    notes: dict = field(default_factory=dict)  # spread of the measurements, for the person

    @staticmethod
    def load(path: Path = CALIBRATION) -> "WalkCalibration":
        if not path.exists():
            return WalkCalibration()
        d = yaml.safe_load(path.read_text()) or {}
        return WalkCalibration(
            forward=Axis(**d.get("forward", asdict(Axis(0.2, 20.0)))),
            sideways=Axis(**d.get("sideways", asdict(Axis(0.2, 20.0)))),
            turn=Axis(**d.get("turn", asdict(Axis(30.0, 30.0)))),
            camera_yaw_deg=float(d.get("camera_yaw_deg", 0.0)),
            measured=bool(d.get("measured", False)),
            notes=d.get("notes", {}) or {},
        )

    def save(self, path: Path = CALIBRATION) -> None:
        d = {"forward": asdict(self.forward), "sideways": asdict(self.sideways), "turn": asdict(self.turn),
             "camera_yaw_deg": round(self.camera_yaw_deg, 2), "measured": self.measured, "notes": self.notes}
        path.write_text("# Written by board/calibrate_walk.py: how far Pupper's timed walks really go.\n"
                        + yaml.safe_dump(d, sort_keys=False))


# ----------------------------------------------------------------------------- pose
def left_of(h: np.ndarray) -> np.ndarray:
    """The direction to the left of heading h, on the board (whose y runs towards Pupper)."""
    return np.array([h[1], -h[0]])


def turn_between(h: np.ndarray, d: np.ndarray) -> float:
    """Degrees to turn (left positive) to go from heading h to heading d."""
    return math.degrees(math.atan2(-(h[0] * d[1] - h[1] * d[0]), h[0] * d[0] + h[1] * d[1]))


def rotate(h: np.ndarray, deg: float) -> np.ndarray:
    """Heading h turned deg to the left."""
    a = math.radians(deg)
    return math.cos(a) * h + math.sin(a) * left_of(h)


@dataclass
class Pose:
    """Pupper on the board: the point on the floor under the lens, its heading, the lens height."""

    x: float
    y: float
    heading: np.ndarray  # unit vector on the board
    height_cm: float

    @property
    def xy(self) -> np.ndarray:
        return np.array([self.x, self.y])


def pose_from_board(R: np.ndarray, t: np.ndarray, camera_yaw_deg: float = 0.0) -> Pose:
    """Pupper's pose from the board's pose in the camera frame (reading.board_R, board_t).

    The board's z axis points down into the floor (x right, y down the page), so
    the lens is at negative z. Pupper's heading is the camera's optical axis laid
    flat on the board, turned back by the camera's own yaw (from calibration)."""
    c = -R.T @ np.asarray(t, float).ravel()  # the lens, in board cm
    f = R.T @ np.array([0.0, 0.0, 1.0])  # the optical axis, in board coordinates
    h = f[:2] / max(np.linalg.norm(f[:2]), 1e-9)
    return Pose(float(c[0]), float(c[1]), rotate(h, -camera_yaw_deg), float(-c[2]))


def stamp_point(pose: Pose, cfg: StampConfig) -> np.ndarray:
    """Where the stamp would land if Pupper crouched right now."""
    return pose.xy - pose.heading * cfg.stamp_behind_camera_cm + left_of(pose.heading) * cfg.stamp_left_of_camera_cm


FACING = np.array([0.0, -1.0])  # squarely towards the far edge


@dataclass
class Plan:
    turn_deg: float  # to face the board squarely
    forward_cm: float  # then walk this far (negative: back)
    left_cm: float  # and step this far to the left (negative: right)

    def describe(self) -> str:
        return f"turn {self.turn_deg:+.1f} deg, forward {self.forward_cm:+.1f} cm, left {self.left_cm:+.1f} cm"


def plan(pose: Pose, target: tuple[float, float], cfg: StampConfig) -> Plan:
    """The moves that bring the stamp over target (board cm).

    Within yaw_tolerance Pupper won't turn, so the walk and the step are measured
    along the heading it really has (a 4 degree error would otherwise put the
    stamp 2 cm sideways over a 30 cm walk). A bigger turn moves the lens too (it
    turns about the body's middle), so after one the stamping job looks again
    before it walks."""
    turn = turn_between(pose.heading, FACING)
    heading = FACING if abs(turn) > cfg.yaw_tolerance_deg else pose.heading
    after = Pose(pose.x, pose.y, heading, pose.height_cm)
    e = np.asarray(target, float) - stamp_point(after, cfg)
    return Plan(turn if abs(turn) > cfg.yaw_tolerance_deg else 0.0, float(e @ heading), float(e @ left_of(heading)))


def moved(before: Pose, after: Pose) -> tuple[float, float, float]:
    """(forward cm, left cm, turned deg) from one pose to the next, measured along the first heading."""
    d = after.xy - before.xy
    return float(d @ before.heading), float(d @ left_of(before.heading)), turn_between(before.heading, after.heading)
