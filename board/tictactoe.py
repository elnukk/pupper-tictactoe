"""Tic-tac-toe for Pupper: the rules, a move chooser and a referee. Plain Python.

A board is 9 characters, cells 1 to 9 as Pupper sees them (row 1 the far row,
each row from Pupper's left): "X", "O", "." (empty) or "?" (the camera couldn't
read it). lab/board_game.py turns this into Gemini's tools.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from functools import lru_cache

LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
LINE_NAMES = ["the far row", "the middle row", "the near row", "the left column", "the middle column",
              "the right column", "the diagonal from far left", "the diagonal from far right"]
# Where each cell is, from Pupper's side ("far" = away from Pupper, "left" = Pupper's left).
CELL_NAMES = ["far left corner", "far middle", "far right corner", "middle left", "center",
              "middle right", "near left corner", "near middle", "near right corner"]
DIFFICULTIES = ["easy", "normal", "unbeatable"]
SLIP = {"easy": 0.6, "normal": 0.2, "unbeatable": 0.0}  # chance of a random move instead of the best


def other(mark: str) -> str:
    return "O" if mark == "X" else "X"


def cell_name(i: int) -> str:
    """'cell 5 (center)' for index 4."""
    return f"cell {i + 1} ({CELL_NAMES[i]})"


def winner(board: str) -> tuple[str, int] | None:
    """(mark, line index) of a completed line, else None."""
    for n, (a, b, c) in enumerate(LINES):
        if board[a] in "XO" and board[a] == board[b] == board[c]:
            return board[a], n
    return None


def empty_cells(board: str) -> list[int]:
    return [i for i, m in enumerate(board) if m == "."]


def place(board: str, i: int, mark: str) -> str:
    return board[:i] + mark + board[i + 1:]


@lru_cache(maxsize=None)
def score(board: str, to_move: str) -> int:
    """Best result for to_move with perfect play on both sides: +1 win, 0 draw, -1 loss."""
    w = winner(board)
    if w:
        return 1 if w[0] == to_move else -1
    cells = empty_cells(board)
    if not cells:
        return 0
    return max(-score(place(board, i, to_move), other(to_move)) for i in cells)


def best_moves(board: str, me: str) -> list[int]:
    cells = empty_cells(board)
    results = {i: -score(place(board, i, me), other(me)) for i in cells}
    top = max(results.values())
    return [i for i in cells if results[i] == top]


def choose(board: str, me: str, difficulty: str = "normal", rng: random.Random | None = None) -> int:
    """Pupper's move. Every level takes a win that's right there; normal and easy
    sometimes play a random move instead of the best one, so people can win."""
    rng = rng or random.Random()
    cells = empty_cells(board)
    if not cells:
        raise ValueError("the board is full")
    wins = [i for i in cells if winner(place(board, i, me))]
    if wins:
        return rng.choice(wins)
    if rng.random() < SLIP.get(difficulty, 0.2):
        return rng.choice(cells)
    best = best_moves(board, me)
    for prefer in (4, 0, 2, 6, 8):  # the center, then corners: a familiar opening
        if prefer in best and board.count(".") >= 8:
            return prefer
    return rng.choice(best)


def reason(board: str, i: int, me: str) -> str:
    """A few words on why the move is good, for Pupper to say."""
    after = place(board, i, me)
    if winner(after):
        return "it wins"
    if winner(place(board, i, other(me))):
        return "it blocks your line"
    threats = sum(1 for line in LINES if i in line
                  and [after[j] for j in line].count(me) == 2 and [after[j] for j in line].count(".") == 1)
    if threats >= 2:
        return "it sets up two ways to win"
    if threats == 1:
        return "it threatens a line"
    if i == 4:
        return "it takes the center"
    if i in (0, 2, 6, 8):
        return "it takes a corner"
    return "it keeps the game going"


# ----------------------------------------------------------------------------- the referee
@dataclass
class Game:
    pupper: str = "O"  # Pupper's mark
    first: str = "person"  # who moved first: "person" or "pupper"
    difficulty: str = "normal"
    last: str = "........."  # the last board the referee accepted
    pending: int | None = None  # Pupper's move, waiting for the person to draw it
    their_move: int | None = None  # the person's latest move, until Pupper reports it
    over: str | None = None  # how it ended, once it has
    history: list[str] = field(default_factory=list)

    @property
    def person(self) -> str:
        return other(self.pupper)

    @property
    def first_mark(self) -> str:
        return self.pupper if self.first == "pupper" else self.person

    def turn(self, board: str) -> str:
        """'pupper' or 'person': whose move it is on this (consistent) board."""
        firsts, seconds = board.count(self.first_mark), board.count(other(self.first_mark))
        mover = self.first_mark if firsts == seconds else other(self.first_mark)
        return "pupper" if mover == self.pupper else "person"

    def problems(self, board: str) -> list[str]:
        """What doesn't add up on this board, given the game so far (empty: all fine)."""
        out = []
        firsts, seconds = board.count(self.first_mark), board.count(other(self.first_mark))
        if not 0 <= firsts - seconds <= 1:
            out.append(f"the counts don't add up: {board.count('X')} X and {board.count('O')} O, "
                       f"but {self.first_mark} went first, so it should have the same number or one more")
        for i, (was, now) in enumerate(zip(self.last, board)):
            if was in "XO" and now != was:
                out.append(f"{cell_name(i)} had {was} and now has {'nothing' if now == '.' else now}")
        new = [i for i, (was, now) in enumerate(zip(self.last, board)) if was == "." and now in "XO"]
        new_pupper = [i for i in new if board[i] == self.pupper]
        new_person = [i for i in new if board[i] == self.person]
        if len(new_person) > 1:
            out.append(f"the person's mark {self.person} appeared in {len(new_person)} cells at once")
        if self.pending is None and new_pupper:
            out.append(f"Pupper's mark {self.pupper} appeared in {cell_name(new_pupper[0])}, "
                       "but Pupper didn't play there")
        elif self.pending is not None and any(i != self.pending for i in new_pupper):
            wrong = next(i for i in new_pupper if i != self.pending)
            out.append(f"Pupper's {self.pupper} was drawn in {cell_name(wrong)}, "
                       f"but Pupper played {cell_name(self.pending)}")
        return out

    def accept(self, board: str) -> None:
        """Take a consistent board as the game's state."""
        if self.pending is not None and board[self.pending] == self.pupper:
            self.pending = None
        new = [i for i, (was, now) in enumerate(zip(self.last, board)) if was == "." and now == self.person]
        if new:
            self.their_move = new[0]
        if board != self.last:
            self.history.append(board)
        self.last = board

    def status(self, board: str) -> str:
        """Whose turn, or how the game ended, for Gemini."""
        w = winner(board)
        if w:
            mark, line = w
            who = "Pupper" if mark == self.pupper else "the person"
            self.over = f"{who} won with {LINE_NAMES[line]}"
            return f"Game over: {who} won ({mark} on {LINE_NAMES[line]})."
        if not empty_cells(board):
            self.over = "a draw"
            return "Game over: it's a draw."
        if self.pending is not None:
            return (f"Waiting for the person to draw Pupper's {self.pupper} in {cell_name(self.pending)}.")
        if self.turn(board) == "pupper":
            return "It's Pupper's turn: call pupper_move."
        return f"It's the person's turn: they draw an {self.person}."
