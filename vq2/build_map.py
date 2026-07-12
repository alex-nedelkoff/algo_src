"""Build the course gate map from a DPVO trajectory + GateNet detections.

Bearings-only multi-view triangulation (no PnP ranges -- sidesteps the
gate-size/origin-offset scale swamp), then similarity alignment to the
spawn frame using the judge-calibrated gate-1 position and the
floor-standing planarity of the official gates.

Usage:
  python -m vq2.build_map <traj.txt> <detections.jsonl> <out.json>
"""
import json
import sys

import numpy as np

FX = FY = 320.0
CX, CY = 320.0, 180.0
# judge-calibrated truth (VQ2-AIM-01): gate 1 aperture, spawn frame
G1_TRUE = np.array([6.3, 0.0, -1.35])

def quat_to_R(q):
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])

def load(traj_path, det_path, det_offset=0):
    T = np.loadtxt(traj_path)
    dets = []
    for i, line in enumerate(open(det_path)):
        fi = i - det_offset
        if fi < 0 or fi >= len(T):
            continue
        m = json.loads(line)
        for inst in m.get('insts', []):
            if not inst.get('solved'):
                continue
            dets.append((fi, np.asarray(inst['center_xy'], float),
                         float(inst.get('center_score', 0))))
    return T, dets

def bearing_world(T, fi, uv):
    row = T[fi]
    p, q = row[1:4], row[4:8]
    d_cam = np.array([(uv[0] - CX) / FX, (uv[1] - CY) / FY, 1.0])
    d_cam /= np.linalg.norm(d_cam)
    return p, quat_to_R(q) @ d_cam

def track_gates(T, dets, ang_tol=0.06, gap=15):
    """Greedy temporal tracking on world-frame bearings."""
    tracks = []
    for fi, uv, score in dets:
        if score < 0.3 or fi >= len(T):
            continue
        o, d = bearing_world(T, fi, uv)
        placed = False
        for tr in tracks:
            lo, ld, lfi = tr[-1]
            if fi - lfi <= gap and np.arccos(np.clip(d @ ld, -1, 1)) < ang_tol:
                tr.append((o, d, fi))
                placed = True
                break
        if not placed:
            tracks.append([(o, d, fi)])
    return [t for t in tracks if len(t) >= 15]

def triangulate(track):
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for o, d, _ in track:
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ o
    pt = np.linalg.solve(A, b)
    # parallax quality: baseline span / mean range
    os_ = np.array([o for o, _, _ in track])
    base = np.linalg.norm(os_.max(0) - os_.min(0))
    rng = np.mean([np.linalg.norm(pt - o) for o in os_])
    return pt, base / max(rng, 1e-6), len(track)

def align(traj_xyz, gates_dpvo):
    """Similarity transform DPVO->spawn: pad(pose of min flight start)=origin,
    nearest-early gate -> G1_TRUE, gate-plane normal -> z."""
    pad = traj_xyz[0]
    # gravity/plane: fit plane normal to gate points
    G = np.array(gates_dpvo)
    c = G.mean(0)
    _, _, vt = np.linalg.svd(G - c)
    n = vt[2]
    # gate 1 = gate closest to pad
    d = np.linalg.norm(G - pad, axis=1)
    g1 = G[np.argmin(d)]
    # scale from pad->g1 distance vs true
    s = np.linalg.norm(G1_TRUE - np.zeros(3)) / max(np.linalg.norm(g1 - pad), 1e-9)
    # rotation: map (g1-pad) -> G1_TRUE direction, n -> z(down or up resolve later)
    a1 = (g1 - pad) / np.linalg.norm(g1 - pad)
    b1 = G1_TRUE / np.linalg.norm(G1_TRUE)
    # choose n sign so that it points to the same side as spawn z relative course
    def frame(u, v):
        w1 = u
        w2 = v - (v @ u) * u
        w2 /= np.linalg.norm(w2)
        return np.column_stack([w1, w2, np.cross(w1, w2)])
    best = None
    for sn in (n, -n):
        Fa = frame(a1, sn)
        Fb = frame(b1, np.array([0, 0, 1.0]))
        R = Fb @ Fa.T
        # score: gates should land at z ~ -1.35 after transform
        Gp = (s * (R @ (G - pad).T)).T
        err = np.abs(Gp[:, 2] - G1_TRUE[2]).mean()
        if best is None or err < best[0]:
            best = (err, R)
    err, R = best
    return R, s, pad, err

def main():
    traj_path, det_path, out_path = sys.argv[1:4]
    off = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    T, dets = load(traj_path, det_path, off)
    print(f'poses {len(T)}, detections {len(dets)}')
    tracks = track_gates(T, dets)
    print(f'tracks >=15 obs: {len(tracks)}')
    pts = []
    for tr in tracks:
        pt, par, n = triangulate(tr)
        if par > 0.05:
            pts.append((pt, par, n))
    print(f'triangulated with parallax: {len(pts)}')
    G = [p for p, _, _ in pts]
    R, s, pad, zerr = align(T[:, 1:4], G)
    print(f'alignment: scale {s:.4f}, gate-plane z err {zerr:.2f} m')
    gates = []
    for (pt, par, n) in pts:
        w = s * (R @ (pt - pad))
        gates.append({'pos': [round(float(x), 2) for x in w],
                      'obs': n, 'parallax': round(par, 3)})
    gates.sort(key=lambda g: g['pos'][0])
    traj_w = (s * (R @ (T[:, 1:4] - pad).T)).T
    out = {'gates': gates, 'scale': s,
           'traj_span': [round(float(x), 1) for x in (traj_w.max(0) - traj_w.min(0))]}
    json.dump(out, open(out_path, 'w'), indent=1)
    np.save(out_path.replace('.json', '_traj.npy'), traj_w)
    for g in gates:
        print('GATE', g['pos'], f"obs={g['obs']} par={g['parallax']}")

if __name__ == '__main__':
    main()
