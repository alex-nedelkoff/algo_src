"""Merged course map: multi-run survey logs, IMU-placed, gate-0 anchored."""
import json, math, bisect, sys
import numpy as np

CAM_TILT = math.radians(20.0)
ct, st = math.cos(CAM_TILT), math.sin(CAM_TILT)
C_Z = np.array([-ct, 0, -st]); C_Y = np.array([-st, 0, ct]); C_X = np.cross(C_Y, C_Z)
M = np.stack([C_X, C_Y, C_Z], axis=1)

def load_run(path):
    poses, obs = [], []
    for line in open(path):
        d = json.loads(line)
        if d['kind'] == 'pose':
            p = np.array(d['p'], float)   # bled integral: compressed scale, consistent across runs
            poses.append((d['t'], p, d['yaw']))
        elif d['kind'] == 'multi_obs':
            obs.append(d)
    return poses, obs

def place(poses, obs):
    ts = [p[0] for p in poses]
    pts = []
    for d in obs:
        i = bisect.bisect(ts, d['f_wall'])
        if i == 0 or i >= len(poses):
            continue
        t_, p_, yaw = poses[i - 1]
        if abs(d['f_wall'] - t_) > 0.2:
            continue
        r, p = d['roll'], d['pitch']
        sr, cr = math.sin(r), math.cos(r)
        sp, cp = math.sin(-p), math.cos(p)
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        cy, sy = math.cos(yaw), math.sin(yaw)
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        R_wb = Rz @ Ry @ Rx
        for inst in d['insts']:
            if inst['lc'] or inst['R'] is None:
                continue
            t_cam = np.array(inst['t_cam'])
            rng = float(np.linalg.norm(t_cam))
            if not (4.0 < rng < 40.0):
                continue
            g_w = p_ + R_wb @ (M @ t_cam)
            Rcg = np.array(inst['R']).reshape(3, 3)
            n_c = Rcg @ np.array([0, 0, 1.0])
            if np.dot(n_c, t_cam) > 0:
                n_c = -n_c
            n_w = R_wb @ (M @ n_c)
            w = 1.0 / (rng * rng)
            pts.append((g_w, n_w, w, rng))
    return pts

def anchor_shift(pts):
    """Anchor: densest near cluster in the first metres = gate 0 -> origin."""
    near = [p for (p, n, w, rng) in pts if rng < 15]
    if not near:
        return np.zeros(3)
    near = np.stack(near)
    med = np.median(near, axis=0)
    sel = near[np.linalg.norm(near - med, axis=1) < 3.0]
    return sel.mean(axis=0) if len(sel) else med

all_pts = []
for path in sys.argv[1:]:
    poses, obs = load_run(path)
    pts = place(poses, obs)
    sh = anchor_shift(pts)
    pts = [(p - sh, n, w, rng) for (p, n, w, rng) in pts]
    print(f'{path}: {len(pts)} placements, anchor {sh.round(1)}')
    all_pts += pts

# greedy weighted clustering
clusters = []
for (p, n, w, rng) in sorted(all_pts, key=lambda x: -x[2]):
    for c in clusters:
        if np.linalg.norm(p - c['pos']) < 4.0:
            c['pos'] = (c['pos'] * c['w'] + p * w) / (c['w'] + w)
            nn = n if np.dot(c['nrm'], n) >= 0 else -n
            c['nrm'] += nn * w
            c['w'] += w; c['n'] += 1
            break
    else:
        clusters.append({'pos': p.copy(), 'nrm': n * w, 'w': w, 'n': 1})

out = []
for c in clusters:
    if c['n'] < 6:
        continue
    nrm = c['nrm'] / max(1e-9, np.linalg.norm(c['nrm']))
    out.append({'pos': [round(float(x), 2) for x in c['pos']],
                'normal': [round(float(x), 3) for x in nrm], 'n_obs': c['n']})
out.sort(key=lambda o: -o['pos'][0])   # course runs along -x: sort by distance downcourse
for i, o in enumerate(out):
    o['id'] = i
    facing = 'COURSE-FACING' if o['normal'][0] > 0.5 else ('side' if abs(o['normal'][0]) < 0.5 else 'back')
    print(f"id {i:2d} pos {o['pos']} n_obs {o['n_obs']:3d} normal {o['normal']} {facing}")
json.dump(out, open('course_map_v4.json', 'w'), indent=1)
row = [o for o in out if o['normal'][0] > 0.5]
print(f'total clusters {len(out)}; COURSE-FACING gates: {len(row)}')
