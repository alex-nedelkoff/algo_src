"""Course map v2: consecutive-frame gate chaining (image-space association).

T_c1_c2 from a gate seen in both frames (Z4-resolved); camera pose chained;
gates accumulated in world. Robust to detection gaps (chain coasts on last pose).
"""
import json, math
import numpy as np

def T_from(R, t):
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
    return T

def inv(T):
    Ti = np.eye(4); Ti[:3, :3] = T[:3, :3].T; Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti

RZ90 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)

def rot_ang(R):
    return math.acos(max(-1, min(1, (np.trace(R) - 1) / 2)))

frames = []
for line in open('survey_detections.jsonl'):
    d = json.loads(line)
    insts = []
    for i in d['insts']:
        if not (i.get('solved') and i.get('t_cam') and i.get('R_cam_gate')):
            continue
        rng = float(np.linalg.norm(i['t_cam']))
        if not (2.0 < rng < 35.0):
            continue
        insts.append({
            'c': np.array(i['center_xy'], float),
            't': np.array(i['t_cam'], float),
            'R': np.array(i['R_cam_gate'], float).reshape(3, 3),
            'lc': bool(i.get('low_confidence')),
            'rng': rng,
        })
    frames.append(insts)

T_wc = np.eye(4)                     # world = first camera pose
gates = []                            # dicts: pos, normal_sum, n, first_idx
prev = None                           # (insts, T_wc_at_that_frame)
n_chained = n_coast = 0

def upd_gates(T_wc, insts, fidx):
    for i in insts:
        if i['lc']:
            continue
        p_w = (T_wc @ np.append(i['t'], 1))[:3]
        n_c = i['R'] @ np.array([0, 0, 1.0])
        if np.dot(n_c, i['t']) > 0:
            n_c = -n_c
        n_w = T_wc[:3, :3] @ n_c
        for g in gates:
            if np.linalg.norm(p_w - g['pos']) < 3.0:
                g['pos'] = (g['pos'] * g['n'] + p_w) / (g['n'] + 1)
                if np.dot(g['normal_sum'], n_w) < 0:
                    n_w = -n_w
                g['normal_sum'] += n_w
                g['n'] += 1
                break
        else:
            gates.append({'pos': p_w, 'normal_sum': n_w, 'n': 1, 'first_idx': fidx})

for fidx, insts in enumerate(frames):
    if not insts:
        continue
    if prev is not None:
        p_insts, T_wc_prev = prev
        # image-space association to previous frame
        best = None
        for a in insts:
            for b in p_insts:
                d = np.linalg.norm(a['c'] - b['c'])
                if d < 80 and abs(a['rng'] - b['rng']) < 3.0 and (best is None or d < best[0]):
                    best = (d, a, b)
        if best is not None:
            _, a, b = best
            # relative camera motion from the shared gate; resolve Z4 on 'a'
            T_cb_g = T_from(b['R'], b['t'])
            cands = []
            Ra = a['R'].copy()
            for k in range(4):
                T_ca_g = T_from(Ra, a['t'])
                T_cb_ca = T_cb_g @ inv(T_ca_g)     # prev_cam <- cur_cam
                cands.append((rot_ang(T_cb_ca[:3, :3]), T_cb_ca))
                Ra = Ra @ RZ90
            cands.sort(key=lambda c: c[0])
            ang, T_cb_ca = cands[0]
            if ang < 0.6 and np.linalg.norm(T_cb_ca[:3, 3]) < 4.0:  # sane inter-frame motion
                T_wc = T_wc_prev @ T_cb_ca
                n_chained += 1
            else:
                n_coast += 1
        else:
            n_coast += 1
    upd_gates(T_wc, insts, fidx)
    prev = (insts, T_wc)

print(f'chained {n_chained} coasted {n_coast} raw gates {len(gates)}')
out = []
for gi, g in enumerate(sorted(gates, key=lambda g: g['first_idx'])):
    if g['n'] < 8:
        continue
    n_w = g['normal_sum'] / max(1e-9, np.linalg.norm(g['normal_sum']))
    out.append({'id': gi, 'pos': [round(float(x), 2) for x in g['pos']],
                'normal': [round(float(x), 3) for x in n_w],
                'n_obs': int(g['n']), 'first_idx': g['first_idx']})
for o in out:
    print(o)
json.dump(out, open('course_map.json', 'w'), indent=1)
print('wrote course_map.json,', len(out), 'gates (n_obs >= 8)')
# course stats
if len(out) > 1:
    ps = [np.array(o['pos']) for o in out]
    d = [round(float(np.linalg.norm(b - a)), 1) for a, b in zip(ps, ps[1:])]
    print('inter-gate spacing (visit order):', d)
