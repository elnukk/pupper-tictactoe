"""Tic-tac-toe: Pupper plays X on a printed board it reads with its camera, and
stamps its moves with a stamp under its chest.

The camera reading is board/board_vision.py (the same reader as board/look.py),
from the USB board camera (board/usb_camera.py) when it's plugged in: it can
sit anywhere, at any angle. Without it, Pupper's own fisheye camera reads.
Stamping always steers by Pupper's own camera (it needs Pupper's pose).
The rules, Pupper's moves and the referee are board/tictactoe.py; where Pupper
is on the board and how to walk the stamp over a cell is board/stamp_plan.py.
Gemini only talks: it never reads the board from the video and never picks
Pupper's move.

  new_game       who starts, how hard; the board must be empty. Pupper is X.
  look_at_board  the grid, whose turn it is, a win or draw, anything that
                 doesn't add up (two moves at once, a mark that changed)
  pupper_move    reads the board, then picks Pupper's move and stamps it: one
                 queued job walks the stamp over the cell, crouches, backs off
                 and looks to check. Without a stamp (board/stamp.yaml
                 enabled: false), the person draws Pupper's X instead.

Stamping, step by step (stamp_cell below): look, turn to face the board
squarely, step sideways until the stamp is in line with the cell, walk closer
while the board is still in view, then walk the last stretch blind (the board is
under Pupper by then) using the measured walk (board/walk_calibration.yaml),
crouch with the push_up trick, back off the same distance and look again.
"stop" cancels it like any move.

Cells are numbered 1 to 9 from Pupper's side: row 1 is the far row, and each
row starts on Pupper's left. The printed board's size: BOARD_SCALE in the
environment (make_board.py --scale), 1.4 for the A3 board (the default).
"""

from __future__ import annotations

import asyncio
import json
import math
import logging
import os
import sys
import time
from pathlib import Path

from part3_follow import prepare_to_move
from part4_tricks import play_policy_trick
from basics import stand_up
from robot_api import FELL_MESSAGE, LLM_CMD_RATE, LOCOMOTION, LOCOMOTION_CONTROLLER, WALK_READY_S, Result, Robot, with_note
from tool_registry import Param, tool

BOARD_DIR = Path(__file__).resolve().parent.parent / "board"
if str(BOARD_DIR) not in sys.path:
    sys.path.append(str(BOARD_DIR))  # last, so it never hides a lab/ or robot/ module
import stamp_plan as sp  # noqa: E402
import tictactoe as ttt  # noqa: E402
import numpy as np  # noqa: E402

log = logging.getLogger("behaviors")

BOARD_SCALE = float(os.environ.get("BOARD_SCALE", "1.4"))
DEBUG_DIR = Path.home() / "board_look"  # what Gemini's last look saw: gemini_*.jpg
CANT_SEE = ("Ask the person to lay the board flat with its near edge 15-20 cm in front of Pupper's"
            " face (Pupper standing), the \"Pupper's side\" edge towards Pupper.")
CANT_SEE_USB = ("Ask the person to check the board is flat, lit, and fully in view of the board camera"
                " (all four corner markers visible, nothing covering them).")
SETTLE_S = 0.8  # after a move, stand still this long before looking (the body stops swaying)
MIN_BURST_S = 0.3  # the walking policy doesn't take a step for a shorter burst (0.14 s moved nothing)
STUCK_FRACTION = 0.3  # a move that covers less than this of what it asked for didn't happen
STEP_BACK_CM = 8.0  # lost the board after a sideways step or a turn: move this much and look again
CLOSE_GAP_CM = 12.0  # nearer than this to the board's near edge, a lost board is too close (step back); further, too far
APPROACH_GAP_CM = 15.0  # further than this from the board's near edge, first walk in to HOME_GAP_CM...
HOME_GAP_CM = 10.0  # ... where Pupper reads the board best, before lining up
SURPRISE_CM, SURPRISE_DEG = 4.0, 8.0  # moving this much more than asked (or the wrong way): stop at once

_readers = {}  # "usb" / "own" -> BoardReader
_last_frame = {}  # "usb" / "own" -> the last frame read (for the trip log)
_usb_lock = __import__("threading").Lock()  # one grab from the USB camera at a time
game: ttt.Game | None = None  # the game being played, if any
stamping = {"running": False, "cell": None, "result": None}  # the last stamping job
STAMP_MAX_S = 150.0  # a stamping job is over by then (one cancelled before it started never clears "running")


# ============================================================ reading the board
def _layout():
    from layout import Layout
    return Layout(scale=BOARD_SCALE)


def _read(jpeg: bytes | None):
    """Read the board from one of Pupper's own camera frames (JPEG), or, with
    None, from a fresh USB camera frame (slow: run it off the event loop)."""
    import cv2
    import numpy as np
    import usb_camera
    from board_vision import CAMERA_PARAMS, BoardReader

    which = "own" if jpeg is not None else "usb"
    if which not in _readers:
        _readers[which] = BoardReader(_layout(), CAMERA_PARAMS if which == "own" else usb_camera.PARAMS)
    if jpeg is None:
        with _usb_lock:
            frame = usb_camera.grab()
    else:
        frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return None
    _last_frame[which] = frame
    reading = _readers[which].read(frame)
    try:
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        prefix = "gemini_own_" if which == "own" and usb_board_camera() else "gemini_"  # keep the board camera's
        cv2.imwrite(str(DEBUG_DIR / f"{prefix}frame.jpg"), frame)
        for name, img in reading.debug.items():
            cv2.imwrite(str(DEBUG_DIR / f"{prefix}{name}.jpg"), img)
    except Exception:
        pass  # debug images are a convenience
    return reading


def usb_board_camera() -> bool:
    import usb_camera
    return usb_camera.available()


async def observe(robot: Robot, own: bool = False):
    """(reading, "") from the board camera, or (None, why not). own: Pupper's own
    camera even when the USB board camera is plugged in (stamping steers by it)."""
    if not own and usb_board_camera():
        reading = await asyncio.to_thread(_read, None)
        if reading is None:
            return None, "no frame from the USB board camera (unplugged, or busy in another program)"
        if not reading.found:
            return None, f"can't read the board: {reading.message}"
        return reading, ""
    try:
        msg = robot.latest_image.get_nowait()
    except Exception:
        return None, "no camera frame yet: the camera isn't running"
    try:
        robot.latest_image.put_nowait(msg)  # keep it for the next look (and the web app)
    except Exception:
        pass
    reading = await asyncio.to_thread(_read, bytes(msg.data))
    if reading is None:
        return None, "the camera frame could not be decoded"
    if not reading.found:
        return None, f"can't read the board: {reading.message}"
    return reading, ""


async def read_board(robot: Robot) -> tuple[str | None, str]:
    """(board as 9 characters, "") from the board camera, or (None, why not). A cell
    the USB camera can't read (out of its frame, or covered) is read with Pupper's
    own camera instead, when that one reads it."""
    reading, why = await observe(robot)
    board = None if reading is None else "".join(m for row in reading.grid for m in row)
    if usb_board_camera() and reading is not None:
        hidden = [c.in_view < 0.98 for row in reading.cells for c in row]  # out of its frame
        if any(hidden):  # (a "?" it can see, ink of no clear shape, stays "?": never trust a blur over it)
            own, _ = await observe(robot, own=True)
            if own is not None:
                mine = "".join(m for row in own.grid for m in row)
                board = "".join(o if h else u for u, o, h in zip(board, mine, hidden))
    if board is None:
        if why.startswith("can't read") and usb_board_camera():
            return None, f"{why}. {CANT_SEE_USB}"
        return None, f"{why}. {CANT_SEE}" if why.startswith("can't read") else why
    return board, ""


def describe(board: str) -> str:
    """The grid as Gemini reads it."""
    rows = [f"  row {r + 1}: " + " ".join(board[r * 3:r * 3 + 3]) for r in range(3)]
    empty = [str(i + 1) for i in ttt.empty_cells(board)]
    text = (
        "The board from Pupper's side (row 1 is the far row; each row starts on Pupper's left;"
        " cells 1 to 9 the same way, 5 = center):\n" + "\n".join(rows)
        + f"\nEmpty cells: {', '.join(empty) or 'none'}."
    )
    unsure = [str(i + 1) for i, m in enumerate(board) if m == "?"]
    if unsure:
        text += (f" Cells {', '.join(unsure)} can't be read (blurry, covered, or outside the camera's view):"
                 " don't guess them. Ask the person to check nothing covers them and the whole board is in"
                 " view, then look again.")
    return text


def referee(board: str) -> str:
    """What the game makes of this board; a consistent board becomes the game's state."""
    if game is None:
        return "No game is going: call new_game to start one."
    if "?" in board:
        return "Can't check the game until every cell can be read."
    problems = game.problems(board)
    if problems:
        return ("Something doesn't add up: " + "; ".join(problems)
                + ". Tell the person in a friendly way and ask them to fix the board, then look again.")
    game.accept(board)
    return game.status(board)


# ============================================================ stamping
# The longest timed move board/calibrate_walk.py measures, per axis: a longer walk is
# split into steps no longer than this, so each stays where the calibration is true
# (a long burst goes less far than the short ones extrapolate to).
LONGEST_CALIBRATED_S = {"forward": 0.55, "sideways": 1.0, "turn": 0.5}


async def drive(robot: Robot, axis: str, amount: float, cal: sp.WalkCalibration) -> Result | None:
    """Walk `amount` (cm forward or left, or degrees left) the calibrated way, in
    steps no longer than the calibration measured. None when it went; a Result if it fell."""
    a = cal.backward if axis == "forward" and amount < 0 and cal.backward else getattr(cal, axis)
    step_max = a.rate * LONGEST_CALIBRATED_S[axis] + a.offset  # what the longest measured move covers
    steps = max(1, math.ceil(abs(amount) / step_max)) if step_max > 1.0 else 1
    for _ in range(steps):
        fell = await _drive_once(robot, axis, amount / steps, a)
        if fell:
            return fell
    return None


async def _drive_once(robot: Robot, axis: str, amount: float, a: sp.Axis) -> Result | None:
    """One timed move, then stand still and let the body settle."""
    seconds = max(a.seconds_for(amount), MIN_BURST_S) if abs(amount) > 0.05 else 0.0
    if seconds > 0:
        v = a.speed if amount > 0 else -a.speed
        vx, vy, wz = (v, 0.0, 0.0) if axis == "forward" else (0.0, v, 0.0) if axis == "sideways" else (0.0, 0.0, v)
        end = robot.now() + seconds
        while robot.now() < end - 1e-6:
            if robot.fallen:
                robot.send_speed(0, 0, 0)
                return Result(False, FELL_MESSAGE)
            robot.send_speed(vx, vy, wz)
            await robot.sleep(1.0 / LLM_CMD_RATE)
        robot.send_speed(0, 0, 0)
    await robot.sleep(SETTLE_S)
    return Result(False, FELL_MESSAGE) if robot.fallen else None


async def crouch(robot: Robot, cfg: sp.StampConfig) -> Result:
    """Press the stamp down: the crouch trick, which ends standing in locomotion."""
    trick = robot.tricks.get(cfg.crouch_trick)
    if trick is None:
        return Result(False, f"the crouch trick {cfg.crouch_trick} isn't installed")
    return await play_policy_trick(robot, trick)


# ------------------------------------------------------------ the trip log
TRIPS_LOG = DEBUG_DIR / "trips.jsonl"  # one line per stamping trip; survives restarts
TRIPS_DIR = DEBUG_DIR / "trips"  # each trip's looks, as images
KEEP_TRIPS = 30  # image folders kept (the log keeps every trip)


def _r(v, nd=1):
    return None if v is None else round(float(v), nd)


class Trip:
    """What one stamping trip saw and did, for board/trips.py (and for working out what goes wrong)."""

    def __init__(self, cell: int, target, cfg: sp.StampConfig, cal: sp.WalkCalibration):
        self.id = time.strftime("%Y%m%d-%H%M%S")
        self.dir = TRIPS_DIR / self.id
        self.n = 0
        self.rec = {"id": self.id, "cell": cell + 1, "target_cm": [_r(target[0]), _r(target[1])],
                    "stamp_behind_cm": cfg.stamp_behind_camera_cm, "stamp_left_cm": cfg.stamp_left_of_camera_cm,
                    "camera_yaw_deg": cal.camera_yaw_deg, "steer_gain": cfg.steer_gain, "chunk_cm": cfg.chunk_cm,
                    "final_cm": cfg.final_cm, "steps": [], "ok": None, "result": None, "landing": None}

    def image(self, name: str, img) -> None:
        if img is None:
            return
        try:
            import cv2
            self.dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(self.dir / f"{self.n:02d}_{name}.jpg"), img)
            self.n += 1
        except Exception:
            pass

    def step(self, **kw) -> None:
        self.rec["steps"].append({k: (_r(v) if isinstance(v, float) else v) for k, v in kw.items()})

    def save(self) -> None:
        try:
            TRIPS_LOG.parent.mkdir(parents=True, exist_ok=True)
            with TRIPS_LOG.open("a") as f:
                f.write(json.dumps(self.rec) + "\n")
            if TRIPS_DIR.exists():
                import shutil
                for old in sorted(TRIPS_DIR.iterdir())[:-KEEP_TRIPS]:
                    shutil.rmtree(old, ignore_errors=True)
        except Exception as e:
            log.warning("couldn't save the trip log: %s", e)


# ------------------------------------------------------------ steering
MIN_CHUNK_CM = 6.0  # a chunk shorter than this is folded into the final stretch (a burst under 0.3 s doesn't step)
MAX_STEER_WZ = 25.0  # deg/s: the most turning mixed into a forward walk
MAX_STEER_VY = 0.12  # m/s: the most sideways drift mixed into a forward walk


def _turn_shift(forward_cm: float, turn_deg: float) -> float:
    """How far left a forward walk ends up when it turns turn_deg (left +) evenly on the way."""
    return forward_cm * math.sin(math.radians(turn_deg) / 2)


def _errors(pose: sp.Pose, target, cfg: sp.StampConfig) -> tuple[float, float, float]:
    """(forward cm, left cm) from the stamp to the target along Pupper's heading, and the
    degrees to turn (left +) to face the board squarely."""
    e = np.asarray(target, float) - sp.stamp_point(pose, cfg)
    return float(e @ pose.heading), float(e @ sp.left_of(pose.heading)), sp.turn_between(pose.heading, sp.FACING)


async def drive_steered(robot: Robot, forward_cm: float, left_cm: float, turn_deg: float,
                        cal: sp.WalkCalibration) -> Result | None:
    """Walk forward_cm, drifting left_cm and turning turn_deg (left +) on the way: small
    corrections ride on the forward walk, so they aren't held back by the shortest step."""
    a = cal.forward
    step_max = a.rate * LONGEST_CALIBRATED_S["forward"] + a.offset
    steps = max(1, math.ceil(abs(forward_cm) / step_max)) if step_max > 1.0 else 1
    turn_ratio = cal.turn.rate / cal.turn.speed if cal.turn.speed else 1.0  # deg turned per deg/s commanded, per s
    side_ratio = cal.sideways.rate / (cal.sideways.speed * 100) if cal.sideways.speed else 1.0
    for _ in range(steps):
        f, l, t = forward_cm / steps, left_cm / steps, turn_deg / steps
        seconds = max(a.seconds_for(f), MIN_BURST_S)
        wz = float(np.clip(t / seconds / max(0.3, turn_ratio), -MAX_STEER_WZ, MAX_STEER_WZ))
        vy = float(np.clip(l / seconds / max(0.3, side_ratio) / 100, -MAX_STEER_VY, MAX_STEER_VY))
        end = robot.now() + seconds
        while robot.now() < end - 1e-6:
            if robot.fallen:
                robot.send_speed(0, 0, 0)
                return Result(False, FELL_MESSAGE)
            robot.send_speed(a.speed, vy, wz)
            await robot.sleep(1.0 / LLM_CMD_RATE)
        robot.send_speed(0, 0, 0)
        await robot.sleep(SETTLE_S)
        if robot.fallen:
            return Result(False, FELL_MESSAGE)
    return None


def smallest_step(a: sp.Axis) -> float:
    """What the shortest burst the walking policy takes a step for covers."""
    return max(0.0, a.rate * MIN_BURST_S + a.offset)


# ------------------------------------------------------------ the stamping job
async def stamp_cell(robot: Robot, cell: int, cfg: sp.StampConfig, cal: sp.WalkCalibration) -> Result:
    """The stamping job: walk the stamp over `cell` (0-8), crouch, back off, check where it
    landed. Every trip goes into the trip log (~/board_look/trips.jsonl)."""
    target = _layout().cell_center(cell // 3, cell % 3)
    trip = Trip(cell, target, cfg, cal)
    try:
        res = await _stamp_trip(robot, cell, target, cfg, cal, trip)
    except asyncio.CancelledError:
        trip.rec.update(ok=False, result="cancelled (stop)")
        trip.save()
        raise
    trip.rec.update(ok=res.ok, result=res.message)
    trip.save()
    return res


async def _stamp_trip(robot: Robot, cell: int, target, cfg: sp.StampConfig, cal: sp.WalkCalibration,
                      trip: Trip) -> Result:
    from landing import find_landing

    res = await robot.switch_to(LOCOMOTION_CONTROLLER)  # standing, ready to walk
    if not res.ok:
        return res
    first, _ = await observe(robot)  # the board before the stamp (board camera if plugged in), to find the new X after
    before_top = first.top if first is not None else None
    trip.image("before_top", before_top)

    board_h = _layout().size[1]
    walked = 0.0  # forward cm walked on this trip, to walk back afterwards
    pose, asked, last, stuck = None, None, None, 0  # last pose, last move asked for, what it was, moves that didn't
    lost_steps = 0  # times it stepped to find the board again
    gap = None  # the board's near edge from the lens (cm): at the last look, then as the last move should leave it
    rest = None  # (forward, left, turn) still to do, as the last look and move leave it: for the blind finish

    for _ in range(cfg.max_steps):
        reading, why = await observe(robot, own=True)
        frame = _last_frame.get("own")
        trip.image("look", reading.debug.get("raw") if reading is not None else frame)
        if reading is None or reading.board_R is None:
            trip.step(kind="look", lost=why or "no pose")
            if rest is not None and last == "chunk" and gap is not None and gap < CLOSE_GAP_CM:
                break  # over the board, which went out of view under Pupper: finish blind from the last look
            if lost_steps < 2:  # (a sideways step or a turn took it out of view): step and look again
                lost_steps += 1
                # Close in, the board is under its face: step back. Far off, it's too small: step in.
                step = -STEP_BACK_CM if gap is None or gap < CLOSE_GAP_CM else STEP_BACK_CM
                trip.step(kind="find", forward=step)
                fell = await drive(robot, "forward", step, cal)
                if fell:
                    return fell
                walked += step
                if gap is not None:
                    gap -= step
                pose, asked, last, rest = None, None, None, None  # a fresh start from wherever it is now
                continue
            return await _back_off(robot, walked, cal, f"lost sight of the board ({why or 'no pose'})")

        before, pose = pose, sp.pose_from_board(reading.board_R, reading.board_t, cal.camera_yaw_deg)
        gap = pose.y - board_h
        f, l, th = _errors(pose, target, cfg)
        entry = dict(kind="look", x=pose.x, y=pose.y, heading_deg=sp.turn_between(sp.FACING, pose.heading),
                     gap=gap, height=pose.height_cm, markers=len(reading.markers), forward=f, left=l, turn=th)
        if before is not None and asked is not None:  # did the last move do what it was asked?
            fwd, lft, turned = sp.moved(before, pose)
            entry.update(asked=asked, moved={"forward": _r(fwd), "left": _r(lft), "turn": _r(turned)})
            axis = "turn" if last == "turn" else "sideways" if last == "sideways" else "forward"
            amount = asked[axis]
            got = {"forward": fwd, "sideways": lft, "turn": turned}[axis]
            limit = SURPRISE_DEG if axis == "turn" else SURPRISE_CM
            if abs(got - amount) > abs(amount) + limit or got * amount < 0 and abs(got) > limit:
                trip.step(**entry)
                robot.send_speed(0, 0, 0)
                return Result(False, f"stopped: asked to move {axis} {amount:+.1f}, Pupper moved {fwd:+.1f} cm"
                              f" forward, {lft:+.1f} cm left and turned {turned:+.1f} deg. Something is wrong"
                              " (slipping, the stamp catching, or the legs sagging): check Pupper before stamping"
                              " again")
            if abs(got) < STUCK_FRACTION * abs(amount):
                stuck += 1
                if stuck >= 2:
                    trip.step(**entry)
                    return await _back_off(robot, walked, cal, f"Pupper isn't moving for small corrections"
                                           f" ({axis} {amount:+.1f} moved {got:+.1f}): run board/calibrate_walk.py")
        trip.step(**entry)
        log.info("stamp %s: lens at (%.1f, %.1f) cm, %.0f cm up, %.0f cm from the board; stamp %+.1f cm forward,"
                 " %+.1f cm left, turn %+.1f deg", ttt.cell_name(cell), pose.x, pose.y, pose.height_cm, gap, f, l, th)

        g = cfg.steer_gain
        far = gap > APPROACH_GAP_CM and f > gap - HOME_GAP_CM
        if not far and abs(th) > cfg.big_turn_deg:  # badly turned: turn in place first
            last, asked, rest = "turn", {"turn": th}, None
            fell = await drive(robot, "turn", th, cal)
        elif not far and abs(l) > cfg.big_side_cm:  # far off to the side: step across first
            last, asked, rest = "sideways", {"sideways": l}, None
            fell = await drive(robot, "sideways", l, cal)
        elif f > cfg.final_cm + MIN_CHUNK_CM:  # a chunk forward, steering out part of the error on the way
            d = min(cfg.chunk_cm, f - cfg.final_cm, gap - HOME_GAP_CM if far else cfg.chunk_cm)
            d = max(d, MIN_CHUNK_CM)
            turn = g * th
            left = g * l - _turn_shift(d, turn)
            last, asked = "chunk", {"forward": d, "left": left, "turn": turn}
            rest = (f - d, l * (1 - g), th * (1 - g))
            walked += d
            gap -= d
            fell = await drive_steered(robot, d, left, turn, cal)
        else:
            rest = (f, l, th)  # close enough: walk the rest, correcting what's left on the way
            break
        if fell:
            return fell
        if abs(walked) > cfg.max_walk_cm:
            return await _back_off(robot, walked, cal, "the walk got too long; gave up")
    else:
        return await _back_off(robot, walked, cal, f"couldn't line up with {ttt.cell_name(cell)} in {cfg.max_steps} looks")

    fr, lr, tr = rest
    trip.step(kind="final", forward=fr, left=lr, turn=tr)
    if abs(walked + fr) > cfg.max_walk_cm or fr < -cfg.blind_max_cm or fr > cfg.blind_max_cm + cfg.chunk_cm:
        return await _back_off(robot, walked, cal, f"the plan was off ({fr:+.1f} cm to go); gave up")
    if fr > 0.5:
        fell = await drive_steered(robot, fr, lr - _turn_shift(fr, tr), tr, cal)
    else:
        fell = await drive(robot, "forward", fr, cal) if fr < -0.5 else None
    walked += fr
    if fell:
        return fell
    res = await crouch(robot, cfg)
    stamping["crouched"] = True
    if not res.ok:
        return res
    back = await _back_off(robot, walked, cal, "")
    if not back.ok:
        return back

    reading, why = await observe(robot)  # the board after (board camera if plugged in)
    if reading is None:
        return Result(True, f"stamped {ttt.cell_name(cell)}, but can't see the board to check ({why})")
    trip.image("after_top", reading.top)
    hit = find_landing(before_top, reading.top, _layout(), target)
    if hit is not None:
        trip.rec["landing"] = {"x": _r(hit.x), "y": _r(hit.y), "short_cm": _r(hit.short_cm),
                               "right_cm": _r(hit.right_cm), "ink_cm2": _r(hit.ink_cm2)}
        log.info("stamp %s: %s", ttt.cell_name(cell), hit.describe())
    where = f" ({hit.describe()})" if hit is not None else ""
    mark = reading.grid[cell // 3][cell % 3]
    if mark == "X":
        return Result(True, f"stamped an X in {ttt.cell_name(cell)}{where}")
    return Result(False, f"crouched over {ttt.cell_name(cell)}, but it shows {mark!r}, not an X{where}")


async def _back_off(robot: Robot, walked: float, cal: sp.WalkCalibration, why: str) -> Result:
    """Walk back what this trip walked forward, so Pupper can see the board again."""
    fell = await drive(robot, "forward", -walked, cal) if abs(walked) > 0.5 else None
    if fell:
        return fell
    return Result(not why, why or "backed off")


async def start_stamping(robot: Robot, cell: int) -> Result:
    """Queue the stamping job (the reply says it was accepted, as for any move)."""
    cfg, cal = sp.StampConfig.load(), sp.WalkCalibration.load()  # read fresh: edits apply without a restart
    note = await prepare_to_move(robot)  # stops follow mode

    async def job() -> Result:
        try:
            res = await stamp_cell(robot, cell, cfg, cal)  # (sets stamping["crouched"] once it presses)
            stamping["result"] = res
            return res
        finally:
            stamping["running"] = False

    # Busy from the moment it's queued: a pupper_move before the job starts must wait for it too.
    stamping.update(running=True, cell=cell, result=None, since=time.monotonic(), crouched=False)
    queued = await robot.queue(f"stamp {ttt.cell_name(cell)}", job, 30.0, then=LOCOMOTION)
    if not queued.ok:
        stamping["running"] = False
    return with_note(note, queued)


def stamping_ready(robot: Robot) -> str:
    """"" if Pupper can stamp now, else why not (then the person draws Pupper's X)."""
    cfg = sp.StampConfig.load()
    if not cfg.enabled:
        return "stamping is off (board/stamp.yaml)"
    if not sp.WalkCalibration.load().measured:
        return "the walk isn't calibrated yet: run board/calibrate_walk.py"
    if cfg.crouch_trick not in robot.tricks:
        return f"the crouch trick {cfg.crouch_trick} isn't installed"
    if robot.fallen:
        return "Pupper has fallen over"
    return ""


# ============================================================ what the robot does
async def look_at_board(robot: Robot) -> Result:
    board, why = await read_board(robot)
    if board is None:
        return Result(False, why)
    return Result(True, describe(board) + " " + referee(board))


async def stand_for_game(robot: Robot) -> str:
    """Stand Pupper up (and wait until it stands) unless it already is: "" or why not."""
    if robot.state_now == LOCOMOTION:
        return ""
    res = await stand_up(robot)
    if not res.ok:
        return res.message
    for _ in range(60):  # behind whatever was queued first
        if robot.state_now == LOCOMOTION:
            break
        await robot.sleep(0.25)
    await robot.sleep(WALK_READY_S + SETTLE_S)  # the policy eases in, then the body settles
    return "" if robot.state_now == LOCOMOTION else "Pupper didn't manage to stand up"


async def new_game(robot: Robot, first: str = "person", difficulty: str = "normal",
                   pupper_is: str = "X") -> Result:
    """A game on the board as it is: empty, or one already going (then pupper_is says
    which mark is Pupper's, and the counts say whose turn it is)."""
    global game
    pupper_is = (pupper_is or "X").upper()
    if pupper_is not in ("X", "O"):
        return Result(False, "pupper_is must be X or O")
    why = await stand_for_game(robot)  # Pupper reads and plays standing
    if why:
        return Result(False, why)
    board, why = await read_board(robot)
    if board is None:
        return Result(False, why)
    if "?" in board:
        return Result(False, describe(board))
    me, them = pupper_is, ttt.other(pupper_is)
    stamps = me == "X"
    how = "stamps its moves" if stamps else "can only stamp X, so the person draws Pupper's O for it"
    if board == ".........":
        game = ttt.Game(pupper=me, first=first, difficulty=difficulty)
        who = "Pupper goes first: call pupper_move now." if first == "pupper" else \
            f"The person goes first: ask them to draw an {them} and say when they're done."
        return Result(True, f"New game ({difficulty}): Pupper is {me} and {how}; the person is {them}. {who}")
    # A game already on the board: carry on from here.
    nx, no = board.count("X"), board.count("O")
    if abs(nx - no) > 1:
        return Result(False, describe(board) + f" That's {nx} X and {no} O, which no game can reach: ask"
                      " the person to fix the board or start a fresh one.")
    if nx != no:  # the mark with more went first
        first = "pupper" if ("X" if nx > no else "O") == me else "person"
    # (equal counts: whoever went first moves next, so `first` says whose turn it is)
    game = ttt.Game(pupper=me, first=first, difficulty=difficulty)
    game.accept(board)
    game.their_move = None
    status = game.status(board)
    if status.startswith("Game over"):
        game = None
        return Result(True, describe(board) + " " + status + " Offer a new game on a fresh board.")
    turn = ("It's Pupper's turn: call pupper_move now." if game.turn(board) == "pupper" else
            f"It's the person's turn: ask them to draw their {them} and say when they're done.")
    return Result(True, describe(board) + f" Carrying on this game ({difficulty}): Pupper is {me} and {how};"
                  f" the person is {them}. {turn}")


async def pupper_move(robot: Robot) -> Result:
    if game is None:
        return Result(False, "No game is going: call new_game first (it carries on a game already on"
                      " the board: ask the person which mark is Pupper's).")
    if stamping["running"] and time.monotonic() - stamping.get("since", 0) < STAMP_MAX_S:
        return Result(False, f"Pupper is still stamping its X in {ttt.cell_name(stamping['cell'])}: wait"
                      " until it has backed off the board.")
    board, why = await read_board(robot)
    if board is None:
        return Result(False, why)
    if "?" in board:
        return Result(False, describe(board))
    note = ""
    if game.pending is not None and board[game.pending] == ".":  # Pupper's last X isn't there
        stray = [i for i in range(9) if game.last[i] == "." and board[i] == game.pupper]
        if len(stray) == 1 and stamping["cell"] == game.pending:  # the stamp slipped into another empty cell
            note = (f"Pupper's stamp landed in {ttt.cell_name(stray[0])} instead of"
                    f" {ttt.cell_name(game.pending)}: that counts as Pupper's move. ")
            game.pending = stray[0]
        else:
            ours = stamping["cell"] == game.pending
            if ours and not stamping.get("crouched", True):  # the trip was given up before it pressed
                why_not = stamping["result"].message if stamping["result"] is not None else "it stopped"
                return Result(False, f"Pupper couldn't get to {ttt.cell_name(game.pending)} to stamp it ({why_not})."
                              f" Ask the person to draw Pupper's {game.pupper} there, then to make their move.")
            return Result(False, ("Pupper's stamp didn't leave an X in " if ours else
                                  f"Pupper's {game.pupper} isn't in ")
                          + f"{ttt.cell_name(game.pending)}: ask the person to draw Pupper's {game.pupper} there,"
                          " then to make their move.")
    problems = game.problems(board)
    if problems:
        return Result(False, note + "Can't move yet, something doesn't add up: " + "; ".join(problems)
                      + ". Ask the person to fix the board.")
    game.accept(board)
    # The person's move since Pupper's last, so Gemini can say it without a look_at_board first.
    played = note + (f"They played {ttt.cell_name(game.their_move)}. " if game.their_move is not None else "")
    status = game.status(board)
    if status.startswith("Game over"):
        return Result(True, played + status + " Offer a rematch.")
    if game.turn(board) != "pupper":
        return Result(False, f"It's the person's turn: they haven't drawn their {game.person} yet.")
    cell = ttt.choose(board, game.pupper, game.difficulty)
    game.pending = cell
    game.their_move = None  # reported
    wins = " Once it's stamped, Pupper wins!" if ttt.winner(ttt.place(board, cell, game.pupper)) else ""
    move = f"{played}Pupper plays {ttt.cell_name(cell)}: {ttt.reason(board, cell, game.pupper)}.{wins}"
    not_ready = stamping_ready(robot) if game.pupper == "X" else "its stamp only makes X"
    if not not_ready:
        queued = await start_stamping(robot, cell)
        if queued.ok:
            return Result(True, f"{move} Pupper is walking over to stamp its X now (about half a minute):"
                          " ask the person to keep their hands off the board until Pupper has backed off,"
                          " then to make their move and say when they're done.")
        not_ready = queued.message
    return Result(True, f"{move} Pupper can't stamp right now ({not_ready}): ask the person to draw Pupper's"
                  f" {game.pupper} there, then to make their move.")


# ============================================================ what Gemini reads
@tool(
    "new_game",
    "Start a tic-tac-toe game on the printed board, or carry on one already drawn"
    " on it. Pupper stands up first. On an empty board: Pupper is X and stamps its"
    " moves; ask who goes first, unless they said. On a board with marks already:"
    " ask which mark is Pupper's (pupper_is) unless they said, and, only if there"
    " are as many X as O, whose turn it is (first).",
    first=Param("string", "Who moves first; carrying on with as many X as O: whose turn it is.",
                enum=["person", "pupper"], required=False),
    pupper_is=Param("string", "Pupper's mark: X (it stamps) unless the person says Pupper is O"
                    " in a game already on the board.", enum=["X", "O"], required=False),
    difficulty=Param("string", "How well Pupper plays: easy (often silly moves), normal (sometimes"
                     " slips), unbeatable. Normal unless the person asks.", enum=ttt.DIFFICULTIES, required=False),
)
async def new_game_tool(robot: Robot, first: str = "person", difficulty: str = "normal",
                        pupper_is: str = "X") -> Result:
    return await new_game(robot, first, difficulty, pupper_is)


@tool(
    "look_at_board",
    "Read the tic-tac-toe board with Pupper's camera: each cell X, O, . (empty)"
    " or ? (too blurry), cells 1 to 9 from Pupper's side, and during a game whose"
    " turn it is, who won, or what doesn't add up. Call it when someone asks about"
    " the board; during a game, pupper_move already reads it. Trust it over the"
    " video. Nothing moves.",
)
async def look_at_board_tool(robot: Robot) -> Result:
    return await look_at_board(robot)


@tool(
    "pupper_move",
    "Call it as soon as the person says they played (or right after new_game if"
    " Pupper goes first): it reads the board itself (no look_at_board first),"
    " checks the game, says what they played, picks Pupper's move and walks over"
    " to stamp Pupper's X there, or says the game is over. If Pupper can't stamp,"
    " it says so: then ask the person to draw Pupper's X." + " Moves Pupper; stop cancels it.",
)
async def pupper_move_tool(robot: Robot) -> Result:
    return await pupper_move(robot)
