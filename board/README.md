# Board vision and stamping

Pupper plays tic-tac-toe on a printed board on the floor. It reads the board
with its own camera (four ArUco markers in the corners, then each cell's ink),
and plays X by walking a stamp under its chest over a cell and crouching.

| file | what it is |
| --- | --- |
| `layout.py` | the board's geometry in cm, shared by the printer and the reader |
| `make_board.py` | writes the printable PDF |
| `board_vision.py` | `BoardReader().read(frame)`: the reading (grid, cells, the board's pose) |
| `look.py` | grab a frame from the board camera and print what it reads |
| `usb_camera.py`, `usb_camera.yaml` | the USB board camera (ELP 16MP): grabbing a frame, and its lens |
| `tictactoe.py` | the rules, Pupper's move chooser and the referee |
| `stamp_plan.py` | where Pupper stands on the board, and how to walk the stamp over a cell |
| `stamp.yaml` | where the stamp is mounted, and stamping on/off |
| `calibrate_walk.py` | measures how far Pupper's timed walks go, into `walk_calibration.yaml` |
| `test_board_vision.py`, `test_tictactoe.py`, `test_stamp.py` | desk tests (no robot) |

Gemini's tools (`new_game`, `look_at_board`, `pupper_move`) are in `lab/board_game.py`.

## Print it

    python3 board/make_board.py --paper a3 --scale 1.4         # the A3 board (board_a3_x1.4.pdf): use this one
    python3 board/make_board.py                                # Letter, cells 5.3 cm (no stamping)

Print at 100% (never "fit to page") and measure a corner square: 5.9 cm on the
A3 board (4.2 cm times the scale). Everything defaults to the A3 board (scale
1.4); for another, pass `--scale` to `look.py` and `calibrate_walk.py` and start
the server with `BOARD_SCALE=...`. Matte paper, taped down flat (Pupper walks on
it), markers unmarked. Lay it with **"Pupper's side" towards the robot**.

## The board camera

A USB camera (ELP 16MP, plain lens) reads the board whenever it's plugged in;
without it, Pupper's own fisheye camera does. It can sit anywhere, at any
angle: on a stand, taped up high, off to one side. The cells are numbered from
the board's printed "Pupper's side" edge, not from the camera, so the camera
can even look from the opposite side. For a good reading, all four corner
markers in view, and a cell at least ~40 px wide in its frame: from straight
above about 60-100 cm up, or at a slant from up to about 1 m away (the higher
it looks down, the better). Stamping still steers by Pupper's own camera.

## Look at it

    python3 board/look.py            # one reading
    python3 board/look.py --watch    # keep reading while you place the board
    python3 board/look.py --json     # every cell: ink, mark, camera-frame position, bearing

It uses the USB board camera when plugged in; else Pupper's camera (the robot
stack's camera topic when the stack runs, else `rpicam-still`).
Debug images go to `~/board_look/`; when a reading is wrong, open
`topdown.jpg` first. Gemini's own looks are saved there too, as `gemini_*.jpg`.

Marks: `X` (ink crossing at its middle), `O` (a ring), `.` (empty) and `?`: ink
of neither shape, or a cell too small in the raw frame (under 11 px) to tell.
The reader would rather say `?` or "no board" than guess.

**Home spot:** Pupper standing, the board's near edge **15-20 cm in front of
its face**. From there it reads every cell. It can tell where the board is out
to about 35 cm, but the far cells blur beyond about 20 cm.

## Stamping

The stamp sits on Pupper's centreline under its chest, just behind the front
legs: the point under the body nearest the camera (the shortest blind walk) and
the lowest point of the `push_up` trick, which is the crouch.

Each of Pupper's moves (`pupper_move`, one queued job; "stop" cancels it):

1. Look from home: the board's pose gives the cell's place in cm, and the
   camera's height and tilt.
2. Turn to face the board squarely, then step sideways until the stamp is in
   line with the cell, looking again after each correction.
3. Walk closer while the board is in view, then the last stretch (up to
   `blind_max_cm`) blind: by then the board is under Pupper.
4. Crouch (`push_up`), back off the same distance, and look: the X must be in
   the cell. A stamp that slipped into another empty cell counts as Pupper's
   move there; one that left no X, Pupper asks the person to draw.

To set it up:

1. Mount the stamp. Measure, on the standing robot, from the lens straight back
   along the centreline to the middle of the stamp's face, and put it in
   `stamp.yaml` (`stamp_behind_camera_cm`, and `stamp_left_of_camera_cm` if
   it's off-centre).
2. Check the crouch touches: stand Pupper over scrap paper, say "do a push
   up", and adjust the stamp's height until it leaves an X.
3. Calibrate the walk (robot stack and server running, the board's near edge
   20-30 cm in front of Pupper):

       python3 board/calibrate_walk.py --dry-run   # only looks: where Pupper is
       python3 board/calibrate_walk.py             # walks short moves and back (2-3 minutes)

   It prints how much the moves vary: about the error of a blind stamp. Up to
   about 2.5 cm lands inside an A3 cell.
4. Set `enabled: true` in `stamp.yaml`. It is read on every move: no restart.

Until then Pupper asks the person to draw its X.

## Range and accuracy (desk tests, not yet measured on the robot)

The desk tests render the board through the real lens model with the camera 15
cm up and pitched 10 degrees up (guesses: `calibrate_walk.py --dry-run` prints
the real height). `test_stamp.py` walks a simulated Pupper that covers only
80-90% of what it is told and drifts, with the floor close under it hidden by
its body; with the walk calibrated, every stamp lands within 2.5 cm of the
cell's middle.

The limit is the lens: its image circle is only ~760 px across for 220
degrees (about 3.5 px per degree), so on the floor, seen at a grazing angle, a
far cell is a few pixels tall. What helps reading most: a bigger board, a
higher camera (standing), dark ink, and good light.
