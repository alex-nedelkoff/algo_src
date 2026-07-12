"""Locate the judge aperture from DPVO truth trajectories.

Per corpus: DPVO trajectory (trimmed frames, spawn-anchored) + judge
verdict (gidx2 semantics on mavlink.jsonl). Align each run's DPVO frame
to the spawn frame: origin = first pose (pad vicinity), scale from the
climb height (commanded ~1.9 m in-est, sim-true equal since IMU
noiseless at rest->hover), yaw from the known course direction (+x =
mean cruise direction). Extract the y/z where each trajectory crosses
the gate plane (x = pad-lock range, ~6.3 m). Ticked runs' crossings
lie inside the judge aperture; misses bound it from outside.

Usage: python -m vq2.truth_crossings <out_dir_with_trajs> <corpora_root>
"""
import json
import os
import struct
import sys

import numpy as np

from vq2.build_map import quat_to_R

CLIMB_TRUE = 1.87   # est climb magnitude (noiseless-IMU DR, pad->hover)
GATE_X = 6.3

def judge_verdict(mav_path):
    prev = None
    reset_i = 0
    rows = []
    for line in open(mav_path):
        m = json.loads(line)
        if m.get('mavpackettype') == 'ENCAPSULATED_DATA':
            d = bytes.fromhex(m['data'])
            if len(d) >= 37 and d[0] == 1:
                f = struct.unpack_from('<BQqqIq', d, 0)
                rows.append((f[1], f[4]))
    for i in range(1, len(rows)):
        if rows[i][0] < rows[i-1][0] - 5000:
            reset_i = i
    post = rows[reset_i:]
    ticked = any(idx > post[0][1] for _, idx in post)
    return ticked

def align_traj(T):
    """Deterministic DPVO->spawn: DPVO world = first camera frame, and the
    spawn camera pose is KNOWN (camera +20 deg up on the drone, drone at
    -17.8 deg pad pitch, facing course +x). Scale from the climb magnitude
    (est/true 1.87 m, noiseless IMU at rest->hover)."""
    import math
    P = T[:, 1:4] - T[0, 1:4]
    # camera net pitch above horizontal at spawn
    th = math.radians(20.0 - 17.8)
    # camera axes in spawn/FRD coords: z_cam = forward-up, x_cam = right, y_cam = down-ish
    zc = np.array([math.cos(th), 0.0, -math.sin(th)])
    xc = np.array([0.0, 1.0, 0.0])
    yc = np.cross(zc, xc)          # completes right-handed cam frame
    # DPVO coords are in cam0 frame (x right, y down, z forward)
    R = np.column_stack([xc, yc, zc])   # cam->spawn
    Ps = (R @ P.T).T
    # scale from climb: max altitude gain (spawn -z) in the first 40%%
    n4 = max(int(len(Ps) * 0.4), 10)
    climb = -(Ps[:n4, 2].min())
    s = CLIMB_TRUE / max(climb, 1e-6)
    return Ps * s

def crossing(Ps):
    for i in range(1, len(Ps)):
        if Ps[i-1][0] < GATE_X <= Ps[i][0]:
            f = (GATE_X - Ps[i-1][0]) / (Ps[i][0] - Ps[i-1][0])
            return Ps[i-1] + f * (Ps[i] - Ps[i-1])
    return None

def main():
    out_dir, root = sys.argv[1], sys.argv[2]
    print(f'{"corpus":<14} {"tick":<5} {"cross y":>8} {"cross z":>8}')
    for f in sorted(os.listdir(out_dir)):
        if not f.endswith('_traj.txt'):
            continue
        c = f[:-9]
        T = np.loadtxt(os.path.join(out_dir, f))
        if T.ndim != 2 or len(T) < 50:
            print(f'{c:<14} traj too short'); continue
        Ps = align_traj(T)
        x = crossing(Ps)
        mav = os.path.join(root, c, 'mavlink.jsonl')
        tick = judge_verdict(mav) if os.path.exists(mav) else None
        if x is None:
            print(f'{c:<14} {str(tick):<5} no gate-plane crossing (max x {Ps[:,0].max():.1f})')
        else:
            print(f'{c:<14} {str(tick):<5} {x[1]:8.2f} {x[2]:8.2f}')

if __name__ == '__main__':
    main()
