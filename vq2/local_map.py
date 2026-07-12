"""Segment-local gate mapping: gates 1-2 only, drift-free window.

Global monocular map fails (scale drift); gate 2 needs only the
gate-1 -> gate-2 relative vector from the local trajectory window around
the gate-1 crossing. Metric scale comes from triangulated gate corner
quads vs official gate dimensions; rotation from crossing velocity
(-> spawn +x) and camera-down (-> spawn +z); translation anchors the
crossed gate at the judge-calibrated G1_TRUE.

Usage:
  python -m vq2.local_map <traj.txt> <detections.jsonl> <offset> [f_end]
"""
import json
import sys

import numpy as np

from vq2.build_map import FX, FY, CX, CY, G1_TRUE, quat_to_R

GATE_OUTER = 2.7   # m, official outer square
GATE_INNER = 1.5   # m, official inner aperture

def load_full(traj_path, det_path, det_offset):
    T = np.loadtxt(traj_path)
    dets = []
    for i, line in enumerate(open(det_path)):
        fi = i - det_offset
        if fi < 0 or fi >= len(T):
            continue
        m = json.loads(line)
        for inst in m.get('insts', []):
            if not inst.get('solved') or inst.get('center_score', 0) < 0.3:
                continue
            dets.append((fi, np.asarray(inst['center_xy'], float),
                         np.asarray(inst['corner_xy'], float)))
    return T, dets

def ray(T, fi, uv):
    row = T[fi]
    p, q = row[1:4], row[4:8]
    d = np.array([(uv[0] - CX) / FX, (uv[1] - CY) / FY, 1.0])
    d /= np.linalg.norm(d)
    return p, quat_to_R(q) @ d

def tri(rays):
    A = np.zeros((3, 3)); b = np.zeros(3)
    for o, d in rays:
        P = np.eye(3) - np.outer(d, d)
        A += P; b += P @ o
    return np.linalg.solve(A, b)

def track(T, dets, ang_tol=0.06, gap=15, min_obs=15):
    tracks = []
    for fi, uv, corners in dets:
        o, d = ray(T, fi, uv)
        for tr in tracks:
            _, ld, lfi, _ = tr[-1]
            if fi - lfi <= gap and np.arccos(np.clip(d @ ld, -1, 1)) < ang_tol:
                tr.append((o, d, fi, corners))
                break
        else:
            tracks.append([(o, d, fi, corners)])
    return [t for t in tracks if len(t) >= min_obs]

def analyze_track(T, tr):
    pt = tri([(o, d) for o, d, _, _ in tr])
    pos = T[:, 1:4]
    dist = np.linalg.norm(pos - pt, axis=1)
    fc = int(np.argmin(dist))
    # corner quads: triangulate each of the 8 corners across the track
    cpts = []
    for k in range(8):
        rays = [ray(T, fi, c[k]) for _, _, fi, c in tr if len(c) > k]
        cpts.append(tri(rays))
    cpts = np.array(cpts)
    # quad side length: mean edge of each 4-corner loop
    def side(q):
        return np.mean([np.linalg.norm(q[i] - q[(i + 1) % 4]) for i in range(4)])
    s1, s2 = side(cpts[:4]), side(cpts[4:])
    return pt, fc, dist, (s1 + s2) / 2, cpts

def main():
    traj_path, det_path, off = sys.argv[1], sys.argv[2], int(sys.argv[3])
    f_end = int(sys.argv[4]) if len(sys.argv) > 4 else 560
    T, dets = load_full(traj_path, det_path, off)
    dets = [d for d in dets if d[0] <= f_end]
    tracks = track(T, dets)
    print(f'window [0,{f_end}]: {len(tracks)} tracks')
    infos = []
    for tr in tracks:
        pt, fc, dist, quad, cpts = analyze_track(T, tr)
        infos.append({'tr': tr, 'pt': pt, 'fc': fc, 'dist': dist,
                      'quad': quad, 'f0': tr[0][2], 'f1': tr[-1][2],
                      'n': len(tr)})
    for i in infos:
        scl_o, scl_i = GATE_OUTER / i['quad'], GATE_INNER / i['quad']
        print(f"trk f[{i['f0']:4d},{i['f1']:4d}] n={i['n']:3d} "
              f"pt=[{i['pt'][0]:7.2f},{i['pt'][1]:7.2f},{i['pt'][2]:7.2f}] "
              f"cross@{i['fc']:4d} d={i['dist'][i['fc']]:6.3f} "
              f"quad={i['quad']:6.4f}u -> scale {scl_o:6.1f}(outer) {scl_i:6.1f}(inner)")
    # distance-to-point profile from trajectory start, per major track
    print('\ndrone->pt distance profile (every 40 frames):')
    for i in infos:
        if i['n'] < 50:
            continue
        prof = ' '.join(f'{i["dist"][f]:.2f}' for f in range(0, f_end, 40))
        print(f"  trk@{i['f0']:4d}: {prof}")

if __name__ == '__main__':
    main()
