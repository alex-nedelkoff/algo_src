"""mine_collisions.py -- map the gate-3 obstacle from today's vq_course collision endpoints.

Each collision run's final position is a measured point ON (or at) the obstacle surface.
All afternoon sessions used the track-aligned chart, so endpoints compare directly to the
TRACK_INFO gate coords. Prints endpoints relative to gate 3 and to the gate2->3 chord.
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path.home() / "Documents" / "vq_data"
G2 = np.array([-74.59375, 1.20001948, 13.66804695])
G3 = np.array([-111.4937439, -5.09998035, 24.56804657])
CH = (G3 - G2)[:2]; CH /= np.linalg.norm(CH)          # chord direction gate2->gate3
PERP = np.array([-CH[1], CH[0]])

runs = sorted(p for p in ROOT.glob("20260611T1*_vq_course") if (p / "data.npz").exists())
print(f"{len(runs)} vq_course runs with data")
pts = []
for p in runs:
    d = np.load(p / "data.npz")
    pos = d["pos"]
    end = pos[-1]
    # use the position at the LAST collision-seq increment if present (cleaner than final row)
    cs = d["coll_seq"]
    inc = np.nonzero(np.diff(cs.astype(float)) != 0)[0]
    if len(inc):
        end = pos[min(inc[-1] + 1, len(pos) - 1)]
    # keep only endpoints in the gate-3 approach region
    if -118 < end[0] < -95 and abs(end[1] - G3[1]) < 15:
        rel = end - G3
        along = float((end[:2] - G3[:2]) @ CH)         # - = before the gate plane
        lat = float((end[:2] - G3[:2]) @ PERP)         # + = left of chord (PERP convention)
        pts.append((p.name, end, along, lat, rel[2]))
print(f"\n{len(pts)} endpoints in the gate-3 region (gate3 = {np.round(G3,1).tolist()}, "
      f"chord dir {np.round(CH,2).tolist()}):")
for name, end, along, lat, dz in pts:
    print(f"  {name[-22:]}: end=[{end[0]:7.1f},{end[1]:6.1f},{end[2]:6.1f}]  "
          f"along={along:+6.1f}  lat={lat:+5.1f}  dz={dz:+5.1f}")
if pts:
    a = np.array([[p[2], p[3], p[4]] for p in pts])
    print(f"\nspread: along [{a[:,0].min():+.1f},{a[:,0].max():+.1f}] "
          f"lat [{a[:,1].min():+.1f},{a[:,1].max():+.1f}] dz [{a[:,2].min():+.1f},{a[:,2].max():+.1f}]")
