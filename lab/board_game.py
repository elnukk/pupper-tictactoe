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
import math
import logging
import os
import sys
from pathlib import Path

from part3_follow import prepare_to_move
from part4_tricks import play_policy_trick
from robot_api import FELL_MESSAGE, LLM_CMD_RATE, LOCOMOTION, LOCOMOTION_CONTROLLER, Result, Robot, with_note
from tool_registry import Param, tool

BOARD_DIR = Path(__file__).resolve().parent.parent / "board"
if str(BOARD_DIR) not in sys.path:
    sys.path.append(str(BOARD_DIR))  # last, so it never hides a lab/ or robot/ module
import stamp_plan as sp  # noqa: E402
import tictactoe as ttt  # noqa: E402

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
SURPRISE_CM, SURPRISE_DEG = 4.0, 8.0  # moving this much more than asked (or the wrong way): stop at once

_readers = {}  # "usb" / "own" -> BoardReader
_usb_lock = __import__("threading").Lock()  # one grab from the USB camera at a time
game: ttt.Game | None = None  # the game being played, if any
stamping = {"running": False, "cell": None, "result": None}  # the last stamping job


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
    reading = _readers[which].read(frame)
    try:
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(DEBUG_DIR / "gemini_frame.jpg"), frame)
        for name, img in reading.debug.items():
            cv2.imwrite(str(DEBUG_DIR / f"gemini_{name}.jpg"), img)
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
    """(board as 9 characters, "") from the latest camera frame, or (None, why not)."""
    reading, why = await observe(robot)
    if reading is None:
        if why.startswith("can't read") and usb_board_camera():
            return None, f"{why}. {CANT_SEE_USB}"
        return None, f"{why}. {CANT_SEE}" if why.startswith("can't read") else why
    return "".join(m for row in reading.grid for m in row), ""


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
        text += (f" Cells {', '.join(unsure)} are too blurry to read: don't guess them. Ask the person"
                 " to move the board closer or tilt it towards Pupper, then look again.")
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
    a = getattr(cal, axis)
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


async def stamp_cell(robot: Robot, cell: int, cfg: sp.StampConfig, cal: sp.WalkCalibration) -> Result:
    """The stamping job: walk the stamp over `cell` (0-8), crouch, back off, check."""
    res = await robot.switch_to(LOCOMOTION_CONTROLLER)  # standing, ready to walk
    if not res.ok:
        return res
    target = _layout().cell_center(cell // 3, cell % 3)
    walked = 0.0  # forward cm walked on this trip, to walk back afterwards
    p, since, last = None, 0.0, None  # the latest plan, forward walked since it, the last move made
    pose, asked, stuck = None, None, 0  # the latest pose, the last move asked for (axis, amount), moves that didn't

    for _ in range(cfg.max_steps):
        reading, why = await observe(robot, own=True)
        if reading is None or reading.board_R is None:
            if p is not None and last == "forward":
                break  # the board went out of view as Pupper closed in: finish blind on the last plan
            return await _back_off(robot, walked, cal, f"lost sight of the board ({why or 'no pose'})")
        before, pose = pose, sp.pose_from_board(reading.board_R, reading.board_t, cal.camera_yaw_deg)
        if before is not None and asked is not None:  # did the last move do what it was asked?
            fwd, left, turned = sp.moved(before, pose)
            axis, amount = asked
            got = {"forward": fwd, "sideways": left, "turn": turned}[axis]
            limit = SURPRISE_DEG if axis == "turn" else SURPRISE_CM
            if abs(got - amount) > abs(amount) + limit or got * amount < 0 and abs(got) > limit:
                robot.send_speed(0, 0, 0)
                return Result(False, f"stopped: asked to move {axis} {amount:+.1f}, Pupper moved {fwd:+.1f} cm"
                              f" forward, {left:+.1f} cm left and turned {turned:+.1f} deg. Something is wrong"
                              " (slipping, the stamp catching, or the legs sagging): check Pupper before stamping"
                              " again")
            if abs(got) < STUCK_FRACTION * abs(amount):
                stuck += 1
                if stuck >= 2:
                    return await _back_off(robot, walked, cal, f"Pupper isn't moving for small corrections"
                                           f" ({axis} {amount:+.1f} moved {got:+.1f}): run board/calibrate_walk.py")
        p, since = sp.plan(pose, target, cfg), 0.0
        log.info("stamp %s: lens at (%.1f, %.1f) cm, %.0f cm up; %s", ttt.cell_name(cell), pose.x, pose.y,
                 pose.height_cm, p.describe())
        if p.turn_deg:
            last, asked = "turn", ("turn", p.turn_deg)
            fell = await drive(robot, "turn", p.turn_deg, cal)
        elif abs(p.left_cm) > cfg.lateral_tolerance_cm:
            last, asked = "sideways", ("sideways", p.left_cm)
            fell = await drive(robot, "sideways", p.left_cm, cal)
        elif p.forward_cm > cfg.blind_max_cm:
            step = p.forward_cm - cfg.blind_target_cm
            last, since, walked, asked = "forward", step, walked + step, ("forward", step)
            fell = await drive(robot, "forward", step, cal)
        else:
            break  # lined up, and close enough to walk the rest blind
        if fell:
            return fell
        if abs(walked) > cfg.max_walk_cm:
            return await _back_off(robot, walked, cal, "the walk got too long; gave up")
    else:
        return await _back_off(robot, walked, cal, f"couldn't line up with {ttt.cell_name(cell)} in {cfg.max_steps} looks")

    final = p.forward_cm - since
    if abs(walked + final) > cfg.max_walk_cm or final < -cfg.blind_max_cm:
        return await _back_off(robot, walked, cal, f"the plan was off ({p.describe()}); gave up")
    fell = await drive(robot, "forward", final, cal)
    walked += final
    if fell:
        return fell
    res = await crouch(robot, cfg)
    if not res.ok:
        return res
    back = await _back_off(robot, walked, cal, "")
    if not back.ok:
        return back
    reading, why = await observe(robot)
    if reading is None:
        return Result(True, f"stamped {ttt.cell_name(cell)}, but can't see the board to check ({why})")
    mark = reading.grid[cell // 3][cell % 3]
    if mark == "X":
        return Result(True, f"stamped an X in {ttt.cell_name(cell)}")
    return Result(False, f"crouched over {ttt.cell_name(cell)}, but it shows {mark!r}, not an X")


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
        stamping.update(running=True, cell=cell, result=None)
        try:
            res = await stamp_cell(robot, cell, cfg, cal)
            stamping["result"] = res
            return res
        finally:
            stamping["running"] = False

    return with_note(note, await robot.queue(f"stamp {ttt.cell_name(cell)}", job, 30.0, then=LOCOMOTION))


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


async def new_game(robot: Robot, first: str = "person", difficulty: str = "normal") -> Result:
    global game
    board, why = await read_board(robot)
    if board is None:
        return Result(False, why)
    if board != ".........":
        return Result(False, describe(board) + " A new game needs an empty board: ask the person"
                      " to erase it or put down a fresh sheet, then call new_game again.")
    game = ttt.Game(pupper="X", first=first, difficulty=difficulty)
    who = "Pupper goes first: call pupper_move now." if first == "pupper" else \
        f"The person goes first: ask them to draw an {game.person} and say when they're done."
    return Result(True, f"New game ({difficulty}): Pupper is X and stamps its moves, the person is O. {who}")


async def pupper_move(robot: Robot) -> Result:
    if game is None:
        return Result(False, "No game is going: call new_game first.")
    if stamping["running"]:
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
            stamped = stamping["cell"] == game.pending
            return Result(False, ("Pupper's stamp didn't leave an X in " if stamped else "Pupper's X isn't in ")
                          + f"{ttt.cell_name(game.pending)}: ask the person to draw Pupper's X there,"
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
    not_ready = stamping_ready(robot)
    if not not_ready:
        queued = await start_stamping(robot, cell)
        if queued.ok:
            return Result(True, f"{move} Pupper is walking over to stamp its X now (about half a minute):"
                          " ask the person to keep their hands off the board until Pupper has backed off,"
                          " then to make their move and say when they're done.")
        not_ready = queued.message
    return Result(True, f"{move} Pupper can't stamp right now ({not_ready}): ask the person to draw Pupper's"
                  " X there, then to make their move.")


# ============================================================ what Gemini reads
@tool(
    "new_game",
    "Start a tic-tac-toe game on the printed board. Pupper is always X and stamps"
    " its moves; the person is O. Ask who goes first, unless they said. The board"
    " must be empty.",
    first=Param("string", "Who moves first.", enum=["person", "pupper"], required=False),
    difficulty=Param("string", "How well Pupper plays: easy (often silly moves), normal (sometimes"
                     " slips), unbeatable. Normal unless the person asks.", enum=ttt.DIFFICULTIES, required=False),
)
async def new_game_tool(robot: Robot, first: str = "person", difficulty: str = "normal") -> Result:
    return await new_game(robot, first, difficulty)


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
