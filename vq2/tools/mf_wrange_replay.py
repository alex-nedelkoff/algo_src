"""Open-loop replay of the width-as-range correction over banked corpora.

For each gate-2 leg: take the logged MF_DR trajectory (`mapfollow` p rows) as
the uncorrected baseline, and re-run the correction against the logged widths
at the true update rate (a detection persists for CMD_HZ*dt updates between
`fg` rows). Report whether the believed range would have tracked the vision
range better, and how much authority the correction used.

HARD LIMIT ON WHAT THIS PROVES. It is OPEN LOOP. In flight the correction
changes `_mf_p`, which changes the carrot, which changes where the drone
goes and therefore every subsequent measurement. This replay cannot predict
the closed-loop outcome and is not an acceptance test -- it answers only:
"does the correction move the believed range toward the imagery, using
bounded authority, without firing on the corpus where DR and vision already
agreed?" A tick is decided in the sim, at n>=3.

Usage: python -m vq2.tools.mf_wrange_replay
"""
from __future__ import annotations

import json
import math

from vq2.live import mf_wrange as W

CMD_HZ = 50.0
GATE = {"vq2_mig2": (11.41, 5.27), "vq2_mig3": (11.37, 5.29),
        "vq2_mig4": (11.39, 5.29), "vq2_mig8": (11.34, 5.29)}


def rows_of(name):
    return [json.loads(l) for l in
            open(rf"C:\Users\Administrator\{name}\livelog.jsonl") if l.strip()]


def replay(name, gate):
    rows = rows_of(name)
    ticks = [r for r in rows if r.get("kind") == "gate_tick"]
    if not ticks:
        return None
    t1 = ticks[0]["t"]
    mf = [(r["t"], list(r["p"])) for r in rows
          if r.get("kind") == "mapfollow" and r.get("p") and r["t"] >= t1]
    fg = [r for r in rows if r.get("kind") in ("fg", "settle")
          and isinstance(r.get("w"), (int, float)) and r["w"] > 0
          and r["t"] > t1]
    if len(mf) < 3 or len(fg) < 4:
        return None

    def base_p(t):
        tt, p = min(mf, key=lambda e: abs(e[0] - t))
        return list(p) if abs(tt - t) <= 0.6 else None

    cal = None
    off = [0.0, 0.0]          # accumulated correction applied to the DR
    fired = used = 0
    guards = {}
    before = after = None
    prev_t = None
    for r in fg:
        p0 = base_p(r["t"])
        if p0 is None:
            continue
        p = [p0[0] + off[0], p0[1] + off[1]]
        if cal is None:
            cal, why = W.calibrate(p, gate, r["w"], 0.05)
            prev_t = r["t"]
            continue
        n = max(1, int(round((r["t"] - prev_t) * CMD_HZ))) if prev_t else 1
        prev_t = r["t"]
        r_vis = cal / r["w"]
        r_dr0 = math.hypot(gate[0] - p0[0], gate[1] - p0[1])
        if before is None:
            before, after = [], []
        for _ in range(min(n, int(W.MAX_AGE * CMD_HZ))):
            dx, dy, d = W.step(p, gate, r["w"], 0.05, cal)
            if d["why"] != "ok":
                guards[d["why"]] = guards.get(d["why"], 0) + 1
                break
            p[0] += dx
            p[1] += dy
            off[0] += dx
            off[1] += dy
            used += math.hypot(dx, dy)
            fired += 1
        r_dr1 = math.hypot(gate[0] - p[0], gate[1] - p[1])
        before.append(abs(r_dr0 - r_vis))
        after.append(abs(r_dr1 - r_vis))
    if not before:
        return None
    return dict(name=name, n=len(before),
                before=sum(before) / len(before),
                after=sum(after) / len(after),
                used=used, fired=fired, guards=guards,
                off=math.hypot(*off))


print("OPEN LOOP -- shows tracking + authority only, not whether it ticks.\n")
print(f'{"corpus":<10}{"n":>4}{"|resid| before":>16}{"|resid| after":>15}'
      f'{"total corr":>12}{"net offset":>12}  guards')
print("-" * 82)
for name, gate in GATE.items():
    res = replay(name, gate)
    if not res:
        print(f"{name:<10} (insufficient rows)")
        continue
    print(f'{res["name"]:<10}{res["n"]:>4}{res["before"]:>15.2f}m'
          f'{res["after"]:>14.2f}m{res["used"]:>11.2f}m{res["off"]:>11.2f}m  '
          + (", ".join(f"{k}x{v}" for k, v in res["guards"].items()) or "-"))
