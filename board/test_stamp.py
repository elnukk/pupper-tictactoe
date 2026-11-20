#!/usr/bin/env python3
"""Desk test for stamping: a simulated Pupper walks its stamp over cells.

    python3 board/test_stamp.py           # about two minutes (every look renders a camera frame)
    python3 board/test_stamp.py --quick   # two cells only
    BOARD_SCALE=1 STAMP_BEHIND_CM=5 python3 board/test_stamp.py   # the A4 board, the stamp 5 cm behind the lens

No robot, camera or Gemini. A made-up world, written from first principles
(east, north, up; turns counter-clockwise), holds the A3 board on the floor and
a Pupper whose walking is deliberately off (it covers 85% of what it is told,
drifts while walking). Every look renders a fisheye frame through the real
lens model from where Pupper's camera is, with the floor close under Pupper
hidden (its own body), and the real reader and planner (board_vision,
stamp_plan) and the real stamping job (lab/board_game.py) drive it through a
fake robot (robot/test_tools.py). The crouch records where the stamp's middle
would land.

Checks: with a calibrated walk the stamp lands within 2.5 cm of the cell's
middle (inside an A3 cell with room for the stamp's own X), from a start that
is off to the side and turned; a stamping trip ends back where it can see the
board; and a game move through pupper_move stamps an X the next move reads.
"""

from __future__ import annotations

import argparse
import asyncio
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "robot"))
import test_tools  # noqa: E402  (installs a fake ROS before robot_api is imported)
import stamp_plan as sp  # noqa: E402
import test_board_vision as tbv  # noqa: E402
from board_vision import BoardReader, unproject  # noqa: E402
from fisheye_converter import load_camera_model  # noqa: E402
from layout import Layout  # noqa: E402
from robot_api import Result  # noqa: E402
from tricks import Trick  # noqa: E402

sys.path.insert(0, str(HERE.parent / "lab"))
import board_game as bg  # noqa: E402

LAYOUT = Layout(scale=float(os.environ.get("BOARD_SCALE", "1.4")))  # board_game reads the same
W, H = LAYOUT.size
BOARD_CENTRE = np.array([0.0, 70.0])  # world cm: east, north
CAMERA_AHEAD_CM = 12.0  # the lens is this far ahead of the point Pupper turns about
HIDDEN_BELOW_DEG = 55.0  # Pupper's own body hides the floor this steeply below level
STAMP_BEHIND_CM = float(os.environ.get("STAMP_BEHIND_CM", "10"))
LANDING_CM = 2.5 * LAYOUT.scale / 1.4  # how far off the stamp's middle may land: 2.5 cm in an A3 cell
# How the simulated Pupper really walks, against what it is told:
TRUE_FORWARD, TRUE_SIDEWAYS, TRUE_TURN = 0.85, 0.80, 0.90
DRIFT_DEG_PER_CM = 0.05  # it veers left a little while walking forward
failures = 0


def check(what: str, ok: bool, detail: str = "") -> None:
    global failures
    failures += not ok
    print(f"  {'ok  ' if ok else 'FAIL'} {what}" + (f"   ({detail})" if detail else ""))


class World:
    """The floor, the board on it, and where Pupper really is."""

    def __init__(self, x: float, y: float, heading_deg: float):
        self.x, self.y, self.theta = x, y, math.radians(heading_deg)  # the point Pupper turns about
        self.cmd = (0.0, 0.0, 0.0)
        self.t = None
        self.marks = [["."] * 3 for _ in range(3)]
        self.landings: list[tuple[int, float]] = []  # (cell index, cm from its middle)

    def fwd(self):
        return np.array([math.cos(self.theta), math.sin(self.theta)])

    def left(self):
        return np.array([-math.sin(self.theta), math.cos(self.theta)])

    def advance(self, now: float) -> None:
        if self.t is not None and now > self.t:
            dt = now - self.t
            vx, vy, wz = self.cmd
            step = self.fwd() * vx * 100 * TRUE_FORWARD * dt + self.left() * vy * 100 * TRUE_SIDEWAYS * dt
            self.x, self.y = self.x + step[0], self.y + step[1]
            self.theta += math.radians(wz * TRUE_TURN * dt + DRIFT_DEG_PER_CM * abs(vx) * 100 * dt)
        self.t = now

    def lens(self):
        return np.array([self.x, self.y]) + self.fwd() * CAMERA_AHEAD_CM

    def stamp(self):
        return self.lens() - self.fwd() * STAMP_BEHIND_CM

    def to_board(self, p):
        """World cm -> board cm (board x east, board y south)."""
        return np.array([p[0] - BOARD_CENTRE[0] + W / 2, BOARD_CENTRE[1] + H / 2 - p[1]])

    def frame(self, rng):
        """What the camera sees from here: the board posed in the level camera frame."""
        c = self.lens()
        f, r = self.fwd(), np.array([math.sin(self.theta), -math.cos(self.theta)])
        corner = np.array([BOARD_CENTRE[0] - W / 2, BOARD_CENTRE[1] + H / 2]) - c  # board (0, 0)
        origin = np.array([corner @ r, tbv.CAMERA_HEIGHT_CM, corner @ f])
        ex = np.array([r[0], 0.0, f[0]])  # board x = east
        ey = np.array([-r[1], 0.0, -f[1]])  # board y = south
        tbv.GAME = self.marks
        img = tbv.scene(LAYOUT, origin, ex, ey, rng)
        img[HIDDEN] = 0  # Pupper's own body
        return img


_model = load_camera_model(tbv.CAMERA_PARAMS, tbv.W, tbv.H)
_v, _u = np.mgrid[0:tbv.H, 0:tbv.W].astype(np.float64)
_rays, _ = unproject(_model, _u, _v)
_th = math.radians(tbv.CAMERA_PITCH_UP_DEG)
_down = _rays[..., 1] * math.cos(_th) - _rays[..., 2] * math.sin(_th)  # the level frame's down (scene()'s pitch)
HIDDEN = np.degrees(np.arcsin(np.clip(_down, -1, 1))) > HIDDEN_BELOW_DEG


class Sim:
    """A fake robot (test_tools.Rig) whose speed commands move a World."""

    def __init__(self, world: World):
        tricks = lambda: {"push_up": Trick(name="push_up", description="Push-ups.", kind="policy",  # noqa: E731
                                           path=Path("push_up.json"), duration=3.0)}
        self.rig = test_tools.Rig(tricks=tricks)
        self.world, self.robot = world, self.rig.ts
        self.rng = np.random.default_rng(0)
        self.reader = BoardReader(LAYOUT)
        self.robot.latest_image.put_nowait(type("Msg", (), {"data": b"frame"})())
        send = self.robot.send_speed

        def send_speed(vx, vy, wz):
            world.advance(self.rig.clock.now)
            world.cmd = (vx, vy, wz)
            send(vx, vy, wz)

        self.robot.send_speed = send_speed
        bg._read = self.read
        bg.crouch = self.crouch

    def read(self, _jpeg):
        self.world.advance(self.rig.clock.now)
        return self.reader.read(self.world.frame(self.rng))

    async def crouch(self, robot, cfg):
        self.world.advance(self.rig.clock.now)
        bx, by = self.world.to_board(self.world.stamp())
        gx, gy = LAYOUT.grid_origin
        col, row = math.floor((bx - gx) / LAYOUT.cell), math.floor((by - gy) / LAYOUT.cell)
        if 0 <= row < 3 and 0 <= col < 3:
            cx, cy = LAYOUT.cell_center(row, col)
            self.world.landings.append((row * 3 + col, math.hypot(bx - cx, by - cy)))
            self.world.marks[row][col] = "X"
        else:
            self.world.landings.append((-1, math.inf))
        return Result(True, "crouched")


async def finish(sim: Sim, limit_s: float = 180.0) -> bool:
    """Wait (real time: every look renders a frame) until the queued stamping job is done."""
    import time
    end = time.monotonic() + limit_s
    while time.monotonic() < end:
        await sim.rig.settle(2000)
        if not bg.stamping["running"] and sim.robot.current is None and sim.robot.commands.empty():
            return True
        await asyncio.sleep(0.05)
    return False


CFG = sp.StampConfig(enabled=True, stamp_behind_camera_cm=STAMP_BEHIND_CM)
TRUE_CAL = sp.WalkCalibration(forward=sp.Axis(0.2, 20 * TRUE_FORWARD), sideways=sp.Axis(0.2, 20 * TRUE_SIDEWAYS),
                              turn=sp.Axis(30.0, 30 * TRUE_TURN), measured=True)


def start() -> World:
    """Pupper's lens 18 cm before the board's near edge (its home spot: from here it reads every
    cell), 6 cm right of the board's middle, turned 8 degrees right."""
    near_edge = BOARD_CENTRE[1] - H / 2
    lens = np.array([6.0, near_edge - 18])
    heading = 90.0 - 8.0
    w = World(0, 0, heading)
    w.x, w.y = lens - w.fwd() * CAMERA_AHEAD_CM
    return w


async def stamp_once(cell: int, cal: sp.WalkCalibration) -> tuple[Result, World, Sim]:
    world = start()
    sim = Sim(world)
    res = await bg.stamp_cell(sim.robot, cell, CFG, cal)
    sim.rig.close()
    return res, world, sim


async def main(quick: bool) -> None:
    print(f"stamping with a calibrated walk (board scale {LAYOUT.scale:g}, cells {LAYOUT.cell:.1f} cm, stamp"
          f" {STAMP_BEHIND_CM:g} cm behind the lens; start 6 cm off and turned 8 deg)")
    for cell in ([4, 0] if quick else [4, 0, 2, 6, 8, 7]):
        res, world, _ = await stamp_once(cell, TRUE_CAL)
        hit = world.landings[-1] if world.landings else (-1, math.inf)
        check(f"cell {cell + 1}: the stamp lands in it, within {LANDING_CM:.1f} cm of its middle",
              hit[0] == cell and hit[1] <= LANDING_CM, f"landed in cell {hit[0] + 1}, {hit[1]:.1f} cm off; {res.message}")
        check(f"cell {cell + 1}: the trip ends where Pupper sees the board and its X", res.ok, res.message)

    if not quick:
        print("stamping with the walk not calibrated (the simulated Pupper covers 80-90% of what it's told)")
        res, world, _ = await stamp_once(8, sp.WalkCalibration())
        hit = world.landings[-1] if world.landings else (-1, math.inf)
        print(f"  info: cell 9 -> landed in cell {hit[0] + 1}, {hit[1]:.1f} cm off ({res.message})."
              " The looks correct the sideways error; the blind walk needs the calibration.")

    print("without the board in view")
    world = start()
    world.theta += math.pi  # facing away
    sim = Sim(world)
    res = await bg.stamp_cell(sim.robot, 4, CFG, TRUE_CAL)
    sim.rig.close()
    check("nothing is stamped, and Pupper says it lost sight of the board",
          not res.ok and "lost sight" in res.message and not world.landings, res.message)

    print("a game move: pupper_move queues the stamping job, and the next move reads Pupper's X")
    world = start()
    sim = Sim(world)
    sp_load = sp.StampConfig.load
    sp.StampConfig.load = staticmethod(lambda path=None: CFG)
    sp.WalkCalibration.load = staticmethod(lambda path=None: TRUE_CAL)
    try:
        await bg.new_game(sim.robot, first="pupper", difficulty="unbeatable")
        res = await bg.pupper_move(sim.robot)
        check("pupper_move picks a cell and says Pupper is walking over to stamp it",
              res.ok and "stamp its X" in res.message, res.message)
        check("the stamping job finishes", await finish(sim))
        cell = bg.game.pending
        stamped = [i for i, (r, c) in enumerate([(i // 3, i % 3) for i in range(9)]) if world.marks[r][c] == "X"]
        check("the queued job stamped that cell", stamped == [cell], f"X in {[i + 1 for i in stamped]}")
        check("the job's outcome says it checked the X", bool(bg.stamping["result"] and bg.stamping["result"].ok),
              bg.stamping["result"].message if bg.stamping["result"] else "no outcome")
        free = next(i for i in range(9) if i != cell)
        world.marks[free // 3][free % 3] = "O"  # the person's move
        res = await bg.pupper_move(sim.robot)
        check("the next pupper_move accepts Pupper's stamped X and the person's O, and moves again",
              res.ok and f"They played cell {free + 1}" in res.message, res.message)
        await finish(sim)
    finally:
        sp.StampConfig.load = sp_load
        sim.rig.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    asyncio.run(main(ap.parse_args().quick))
    print("\nall checks passed" if not failures else f"\n{failures} check(s) FAILED")
    sys.exit(1 if failures else 0)
