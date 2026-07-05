"""Course map v3: detections placed by the IMU trajectory (no VO)."""
import json, math, bisect
import numpy as np

poses = []      # (t, p(3), yaw, roll, pitch)
obs = []        # (f_wall, roll, pitch, insts)
for line in open('survey2_log.jsonl'):
    d = json.loads(line)
    if d['kind'] == 'pose':
        poses.append((d['t'], np.array(d['p']), d['yaw'], d['roll'], d['pitch']))
    elif d['kind'] == 'multi_obs':
        obs.append(d)
print('poses', len(poses), 'obs frames', len(obs))
ts = [p[0] for p in poses]

CAM_TILT = math.radians(20.0)
ct, st = math.cos(CAM_TILT), math.sin(CAM_TILT)
C_Z = np.array([-ct, 0, -st]); C_Y = np.array([-st, 0, ct]); C_X = np.cross(C_Y, C_Z)
M = np.stack([C_X, C_Y, C_Z], axis=1)

gates = []
for d in obs:
    # pose at frame time
    i = bisect.bisect(ts, d['f_wall'])
    if i == 0 or i >= len(poses):
        continue
    t_, p_, yaw, roll_l, pitch_l = poses[i - 1]
    if abs(d['f_wall'] - t_) > 0.15:
        continue
    r, p = d['roll'], d['pitch']       # attitude at frame time (from det epoch buffer)
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
        rng = np.linalg.norm(t_cam)
        if not (2.5 < rng < 30.0):
            continue
        g_b = M @ t_cam
        g_w = p_ + R_wb @ g_b
        Rcg = np.array(inst['R']).reshape(3, 3)
        n_c = Rcg @ np.array([0, 0, 1.0])
        if np.dot(n_c, t_cam) > 0:
            n_c = -n_c
        n_b = M @ n_c
        n_w = R_wb @ n_b
        w = 1.0 / max(1.0, rng / 5.0)     # near observations weigh more
        for g in gates:
            if np.linalg.norm(g_w - g['pos']) < 3.0:
                g['pos'] = (g['pos'] * g['w'] + g_w * w) / (g['w'] + w)
                if np.dot(g['normal'], n_w) < 0:
                    n_w = -n_w
                g['normal'] = g['normal'] + n_w * w
                g['w'] += w; g['n'] += 1
                break
        else:
            gates.append({'pos': g_w, 'normal': n_w * w, 'w': w, 'n': 1,
                          'first_t': d['f_wall']})

out = []
for g in sorted(gates, key=lambda g: g['first_t']):
    if g['n'] < 6:
        continue
    n_ = g['normal'] / max(1e-9, np.linalg.norm(g['normal']))
    out.append({'pos': [round(float(x), 2) for x in g['pos']],
                'normal': [round(float(x), 3) for x in n_],
                'n_obs': g['n']})
for i, o in enumerate(out):
    o['id'] = i
    print(o)
json.dump(out, open('course_map_v3.json', 'w'), indent=1)
if len(out) > 1:
    ps = [np.array(o['pos']) for o in out]
    print('spacing:', [round(float(np.linalg.norm(b - a)), 1) for a, b in zip(ps, ps[1:])])
