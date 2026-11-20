#!/usr/bin/env python3
"""Desk tests for tic-tac-toe: the engine, the referee and Gemini's game tools.

    python3 board/test_tictactoe.py

No robot, camera or Gemini: the tools get boards from a fake camera.
"""

from __future__ import annotations

import asyncio
import itertools
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tictactoe as ttt  # noqa: E402

failures = 0


def check(what: str, ok: bool, detail: str = "") -> None:
    global failures
    failures += not ok
    print(f"  {'ok  ' if ok else 'FAIL'} {what}" + ("" if ok or not detail else f"\n         {detail}"))


# ----------------------------------------------------------------------------- engine
def play_all(board: str, to_move: str, me: str, difficulty: str, rng) -> bool:
    """True if Pupper (me) never loses from here, against every possible opponent move."""
    w = ttt.winner(board)
    if w:
        return w[0] == me  # True: Pupper won; False: the opponent did
    if not ttt.empty_cells(board):
        return True
    if to_move == me:
        return play_all(ttt.place(board, ttt.choose(board, me, difficulty, rng), me), ttt.other(me), me, difficulty, rng)
    return all(play_all(ttt.place(board, i, to_move), me, me, difficulty, rng) for i in ttt.empty_cells(board))


def engine() -> None:
    print("engine")
    check("a full line wins", ttt.winner("XXX.O.O..") == ("X", 0))
    check("a diagonal wins", ttt.winner("O.X.O.X.O") == ("O", 6))
    check("no line, no winner", ttt.winner("XOXXOOOXX") is None)
    rng = random.Random(1)
    never_loses = all(play_all(".........", first, "O", "unbeatable", rng) for first in "XO")
    check("unbeatable never loses, whoever starts, against every opponent", never_loses)
    check("every level takes a win that's right there",
          all(ttt.choose("OO.XX....", "O", d, random.Random(s)) == 2 for d in ttt.DIFFICULTIES for s in range(20)))
    check("unbeatable blocks", ttt.choose("XX..O....", "O", "unbeatable") == 2)
    picks = [ttt.choose("X........", "O", "normal", random.Random(s)) for s in range(200)]
    check("normal answers a corner with the center mostly, but not always",
          0.6 < picks.count(4) / len(picks) < 0.95, f"center {picks.count(4)} of {len(picks)}")
    check("the reason for a block says so", ttt.reason("XX..O....", 2, "O") == "it blocks your line")
    check("the reason for a win says so", ttt.reason("OO.XX....", 2, "O") == "it wins")


# ----------------------------------------------------------------------------- referee
def referee() -> None:
    print("referee")
    g = ttt.Game(pupper="O", first="person")
    check("empty board, person first: person's turn", g.turn(".........") == "person")
    check("after the person's X: Pupper's turn", g.turn("....X....") == "pupper")
    check("a consistent board has no problems", g.problems("....X....") == [])
    check("two X at once is a problem", any("2 cells" in p for p in g.problems("X...X....")))
    check("an O Pupper didn't play is a problem", any("didn't play" in p for p in g.problems("O...X....")))
    g.accept("....X....")
    g.pending = 0
    check("Pupper's O drawn where it played is fine", g.problems("O...X....") == [])
    check("Pupper's O drawn elsewhere is a problem", any("was drawn in cell 3" in p for p in g.problems("..O.X....")))
    g.accept("O...X....")
    check("drawing Pupper's move clears it", g.pending is None)
    check("a mark that disappears is a problem", any("now has nothing" in p for p in g.problems("O........")))
    check("a mark that changes is a problem", any("now has X" in p for p in g.problems("X...X....")))
    check("a win is game over", g.status("OOOXX.X..").startswith("Game over: Pupper won"))
    check("a full board without a line is a draw", "draw" in ttt.Game().status("XOXXOOOXX"))
    g2 = ttt.Game(pupper="X", first="pupper")
    check("Pupper first as X: Pupper's turn on an empty board", g2.turn(".........") == "pupper")


# ----------------------------------------------------------------------------- tools
class FakeImage:
    """robot.latest_image: hands out a frame; the board in it is set by the test."""

    def __init__(self):
        self.msg = type("Msg", (), {"data": b"frame"})()

    def get_nowait(self):
        return self.msg

    def put_nowait(self, msg):
        pass


def tools() -> None:
    print("tools")
    sys.path.insert(0, str(HERE.parent / "robot"))
    import test_tools  # noqa: F401  (a fake ROS, so robot_api imports at a desk)
    sys.path.insert(0, str(HERE.parent / "lab"))
    import board_game as bg

    shown = {"board": "........."}

    def fake_read(jpeg):
        board = shown["board"]
        grid = [list(board[r * 3:r * 3 + 3]) for r in range(3)]
        return type("Reading", (), {"found": True, "grid": grid, "message": "", "board_distance_cm": 30})()

    bg._read = fake_read
    robot = type("Robot", (), {"latest_image": FakeImage(), "tricks": {}, "fallen": None})()
    run = lambda coro: asyncio.run(coro)  # noqa: E731

    # Pupper is X (it stamps), the person O. With board/stamp.yaml's stamping off, Pupper asks the
    # person to draw its X; test_stamp.py tests the stamping itself.
    bg.sp.StampConfig.load = staticmethod(lambda path=None: bg.sp.StampConfig(enabled=False))
    shown["board"] = "O........"
    res = run(bg.new_game(robot))
    check("new_game refuses a board with marks on it", not res.ok and "empty board" in res.message, res.message)
    shown["board"] = "........."
    res = run(bg.new_game(robot, "person", "normal"))
    check("new_game on an empty board: Pupper is X", res.ok and "Pupper is X" in res.message
          and "the person goes first" in res.message.lower(), res.message)
    res = run(bg.pupper_move(robot))
    check("pupper_move refuses on the person's turn", not res.ok and "person's turn" in res.message, res.message)
    shown["board"] = "....O...."
    res = run(bg.look_at_board(robot))
    check("look_at_board after the person's O: Pupper's turn", res.ok and "Pupper's turn" in res.message, res.message)
    res = run(bg.pupper_move(robot))
    cell = bg.game.pending
    check("pupper_move says what the person played",
          res.ok and "They played cell 5" in res.message, res.message)
    check("... picks a cell; stamping off, it asks the person to draw Pupper's X",
          res.ok and cell is not None and "draw Pupper's X" in res.message and f"cell {cell + 1}" in res.message,
          res.message)
    res = run(bg.pupper_move(robot))
    check("pupper_move again before it's drawn: asks for it, doesn't pick again",
          not res.ok and "isn't in" in res.message and bg.game.pending == cell, res.message)
    wrong = next(i for i in range(9) if i not in (4, cell))
    shown["board"] = ttt.place("....O....", wrong, "X")
    res = run(bg.look_at_board(robot))
    check("Pupper's X drawn in the wrong cell is caught", res.ok and "doesn't add up" in res.message, res.message)
    with_x = ttt.place("....O....", cell, "X")
    nxt = next(i for i in range(9) if with_x[i] == "." and not ttt.winner(ttt.place(with_x, i, "O")))
    shown["board"] = ttt.place(with_x, nxt, "O")
    res = run(bg.pupper_move(robot))
    check("Pupper's X and the person's next O together: one pupper_move handles both",
          res.ok and f"They played cell {nxt + 1}" in res.message and bg.game.pending not in (None, cell), res.message)
    shown["board"] = "?...O...."
    res = run(bg.pupper_move(robot))
    check("a blurry cell stops pupper_move", not res.ok and "too blurry" in res.message, res.message)

    # A whole game against unbeatable Pupper, one pupper_move per turn as Gemini plays it:
    # Pupper's last X appears, the person adds their O (the first empty cell), then says "your turn".
    shown["board"] = "........."
    run(bg.new_game(robot, "person", "unbeatable"))
    board, res = ".........", None
    for _ in range(9):
        if bg.game.pending is not None:
            board = ttt.place(board, bg.game.pending, "X")
        if not ttt.winner(board) and ttt.empty_cells(board):
            board = ttt.place(board, ttt.empty_cells(board)[0], "O")
        shown["board"] = board
        res = run(bg.pupper_move(robot))
        if "Game over" in res.message or not res.ok:
            break
    check("a whole game ends, and unbeatable Pupper doesn't lose",
          res.ok and "Game over" in res.message and "the person won" not in res.message, res.message)

    # The tools as Gemini sees them.
    import tool_registry
    game_tools = {"new_game", "look_at_board", "pupper_move"}
    check("Gemini gets new_game, look_at_board and pupper_move",
          game_tools <= set(tool_registry.TOOLS), str(sorted(tool_registry.TOOLS)))
    check("... each with a real description",
          all(len(tool_registry.TOOLS[n].description) >= 20 for n in game_tools))


if __name__ == "__main__":
    engine()
    referee()
    tools()
    print("\nall checks passed" if not failures else f"\n{failures} check(s) FAILED")
    sys.exit(1 if failures else 0)
