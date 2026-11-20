#!/usr/bin/env python3
"""Print the tic-tac-toe board Pupper reads (layout.py describes it).

    python3 board/make_board.py                    # board/board_letter.pdf, US Letter
    python3 board/make_board.py --paper a4         # board/board_a4.pdf
    python3 board/make_board.py --paper tabloid --scale 1.35   # bigger board: Pupper reads it from further
    python3 board/make_board.py --paper custom --scale 1.5     # page sized to the board (poster / plotter)

Print at 100% ("actual size", never "fit to page"), then measure a corner
square: it must be 4.2 cm (times --scale). Matte paper, and keep the markers
flat and unmarked: the camera finds the board by them. Tape the board down
with the "Pupper's side" line towards the robot.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from reportlab.lib.pagesizes import A3, A4, LETTER, TABLOID
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas

from layout import MARKER_IDS, Layout

HERE = Path(__file__).resolve().parent
PAPERS = {"letter": LETTER, "a4": A4, "tabloid": TABLOID, "a3": A3}


def make(path: Path, layout: Layout, paper: str) -> None:
    w, h = layout.size
    if paper == "custom":  # a page just bigger than the board (poster / plotter)
        page_w, page_h = (w + 2) * cm, (h + 2) * cm
    else:
        page_w, page_h = PAPERS[paper]
    if (w + 0.8) * cm > page_w or (h + 0.8) * cm > page_h:  # printers leave ~4 mm blank at the edges
        raise SystemExit(f"a {w:.1f} x {h:.1f} cm board does not fit on {paper}; "
                         f"use a smaller --scale or --paper custom")
    ox, oy = (page_w - w * cm) / 2, (page_h - h * cm) / 2  # centre the board

    c = canvas.Canvas(str(path), pagesize=(page_w, page_h))
    c.setTitle("Pupper tic-tac-toe board")

    def rect(x, y, rw, rh):  # layout cm, y down -> PDF points, y up
        c.rect(ox + x * cm, page_h - oy - (y + rh) * cm, rw * cm, rh * cm, stroke=0, fill=1)

    c.setFillGray(0)
    for r in layout.grid_lines():
        rect(*r)
    for mid in MARKER_IDS:
        for r in layout.marker_modules(mid):
            rect(*r)

    # Notes between the markers, well away from the grid.
    m = layout.s(layout.marker)
    c.setFillGray(0.35)
    size = 9 * layout.scale
    c.setFont("Helvetica", size)
    top = page_h - oy - m / 2 * cm
    c.drawCentredString(page_w / 2, top, "far side")
    c.setFont("Helvetica-Bold", 1.3 * size)
    bottom = oy + m / 2 * cm
    c.drawCentredString(page_w / 2, bottom + 0.4 * cm * layout.scale, "Pupper's side")
    c.setFont("Helvetica", 0.8 * size)
    c.drawCentredString(page_w / 2, bottom - 0.4 * cm * layout.scale,
                        f"print at 100%: each corner square measures {m:.1f} cm")
    c.showPage()
    c.save()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paper", choices=[*PAPERS, "custom"], default="letter")
    ap.add_argument("--scale", type=float, default=1.0, help="board size, 1 = fits Letter/A4 (cells 5.3 cm)")
    ap.add_argument("-o", "--out", type=Path, help="PDF to write (default board/board_<paper>.pdf)")
    args = ap.parse_args()
    layout = Layout(scale=args.scale)
    name = f"board_{args.paper}" + ("" if args.scale == 1.0 else f"_x{args.scale:g}")
    out = args.out or HERE / f"{name}.pdf"
    make(out, layout, args.paper)
    print(f"wrote {out}: {layout.size[0]:.1f} x {layout.size[1]:.1f} cm board, "
          f"cells {layout.cell:.1f} cm, markers {layout.s(layout.marker):.1f} cm (scale {args.scale:g})")


if __name__ == "__main__":
    main()
