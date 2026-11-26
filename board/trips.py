#!/usr/bin/env python3
"""What Pupper's stamping trips did: where each stamp landed, and how its moves went.

Every stamping trip (lab/board_game.py) appends a line to ~/board_look/trips.jsonl
and saves what it saw in ~/board_look/trips/<id>/.

    python3 board/trips.py            # the last 10 trips, and the average miss
    python3 board/trips.py -n 30      # more
    python3 board/trips.py --steps    # every look and move of the last trip
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

LOG = Path.home() / "board_look" / "trips.jsonl"


def load() -> list[dict]:
    if not LOG.exists():
        return []
    out = []
    for line in LOG.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def show_steps(t: dict) -> None:
    print(f"trip {t['id']}: cell {t['cell']}, target {t['target_cm']} cm  ->  {t['result']}")
    for s in t["steps"]:
        if s["kind"] == "look" and "lost" in s:
            print(f"  look: LOST ({s['lost']})")
        elif s["kind"] == "look":
            moved = ""
            if "asked" in s:
                a, m = s["asked"], s["moved"]
                moved = (f"   after asking {', '.join(f'{k} {v:+.1f}' for k, v in a.items())}:"
                         f" moved fwd {m['forward']:+.1f}, left {m['left']:+.1f}, turn {m['turn']:+.1f}")
            print(f"  look: {s['gap']:+5.1f} cm from the board, {s['markers']} markers;"
                  f" stamp needs fwd {s['forward']:+5.1f}, left {s['left']:+5.1f}, turn {s['turn']:+5.1f}{moved}")
        else:
            rest = ", ".join(f"{k} {v:+.1f}" for k, v in s.items() if k != "kind" and isinstance(v, (int, float)))
            print(f"  {s['kind']}: {rest}")
    if t.get("landing"):
        L = t["landing"]
        print(f"  landed: {L['short_cm']:+.1f} cm short, {L['right_cm']:+.1f} cm right ({L['ink_cm2']} cm2 of ink)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", type=int, default=10, help="how many recent trips")
    ap.add_argument("--steps", action="store_true", help="every look and move of the last trip")
    args = ap.parse_args()
    trips = load()
    if not trips:
        print(f"no trips yet ({LOG})")
        return 0
    if args.steps:
        show_steps(trips[-1])
        return 0
    recent = trips[-args.n:]
    print(f"{'trip':16s} cell  looks  lost  short   right   result")
    for t in recent:
        looks = [s for s in t["steps"] if s["kind"] == "look"]
        lost = sum(1 for s in looks if "lost" in s)
        L = t.get("landing") or {}
        sh = f"{L['short_cm']:+5.1f}" if L else "   - "
        rt = f"{L['right_cm']:+5.1f}" if L else "   - "
        print(f"{t['id']:16s}  {t['cell']}    {len(looks):3d}   {lost:3d}   {sh}   {rt}   {str(t['result'])[:60]}")
    hits = [t["landing"] for t in recent if t.get("landing")]
    if len(hits) >= 2:
        sh = [h["short_cm"] for h in hits]
        rt = [h["right_cm"] for h in hits]
        print(f"\n{len(hits)} measured stamps: on average {statistics.mean(sh):+.1f} cm short"
              f" (spread {statistics.stdev(sh):.1f}), {statistics.mean(rt):+.1f} cm right (spread {statistics.stdev(rt):.1f})")
        print("(a steady average miss is a bias to correct; the spread is what's left after that)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
