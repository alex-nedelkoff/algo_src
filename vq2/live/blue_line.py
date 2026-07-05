"""Extract the rendered cyan course line from survey frames -> world polyline.

Per frame: HSV-segment cyan, back-project pixels through the nose camera
(fx=fy=320, c=(320,180), +x boresight 20 deg up) onto the floor plane using
the logged pose/attitude; accumulate world points ordered by frame time.
"""
import json, math, bisect, glob, os, sys
import numpy as np
import cv2

LOG = r'C:\Users\alexj\vq2_survey\log.jsonl'
FRAMES = r'C:\Users\alexj\vq2_survey\frames'
OUT = r'C:\Users\alexj\blue_line.jsonl'

FX = FY = 320.0; CX = 320.0; CY = 180.0
CAM_TILT = math.radians(20.0)
ct, st = math.cos(CAM_TILT), math.sin(CAM_TILT)
C_Z = np.array([ct, 0, -st]); C_Y = np.array([st, 0, ct]); C_X = np.cross(C_Y, C_Z)
M = np.stack([C_X, C_Y, C_Z], axis=1)
Z_FLOOR = 1.4   # floor below the takeoff origin (pad ~0.2 + survey alt ~1.2) -- refined by calibration pass

poses = []
frame_meta = {}
for line in open(LOG):
    d = json.loads(line)
    if d['kind'] == 'pose':
        poses.append((d['t'], np.array(d['p']), -d['yaw'], d['roll'], d['pitch']))
    elif d['kind'] == 'multi_obs':
        frame_meta[d['ns']] = d['f_wall']
ts = [p[0] for p in poses]
print('poses', len(poses), 'frame meta', len(frame_meta), flush=True)

out = open(OUT, 'w')
n_pts = 0
for fp in sorted(glob.glob(os.path.join(FRAMES, '*.jpg'))):
    ns = int(os.path.basename(fp)[:-4])
    fw = frame_meta.get(ns)
    if fw is None:
        continue
    i = bisect.bisect(ts, fw)
    if i == 0 or i >= len(poses):
        continue
    t_, p_, yaw, roll, pitch = poses[i - 1]
    if abs(fw - t_) > 0.2:
        continue
    img = cv2.imread(fp)
    if img is None:
        continue
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # cyan glow: hue ~85-105, decent saturation+value
    mask = cv2.inRange(hsv, (80, 80, 120), (110, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    ys, xs = np.nonzero(mask)
    if len(xs) < 20:
        continue
    # subsample pixels
    idx = np.random.default_rng(0).choice(len(xs), min(120, len(xs)), replace=False)
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    cyw, syw = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[cyw, -syw, 0], [syw, cyw, 0], [0, 0, 1]])
    R_wb = Rz @ Ry @ Rx
    pts = []
    for k in idx:
        u, v = float(xs[k]), float(ys[k])
        d_cam = np.array([(u - CX) / FX, (v - CY) / FY, 1.0])
        d_w = R_wb @ (M @ d_cam)
        if d_w[2] < 0.05:            # must point downward to hit the floor
            continue
        s = (Z_FLOOR - p_[2]) / d_w[2]
        if not (1.5 < s * np.linalg.norm(d_w) < 45.0):
            continue
        w = p_ + s * d_w
        pts.append([round(float(w[0]), 2), round(float(w[1]), 2)])
    if pts:
        out.write(json.dumps({'t': fw, 'drone': p_.round(2).tolist(), 'pts': pts}) + '\n')
        n_pts += len(pts)
out.close()
print('line points:', n_pts, '->', OUT, flush=True)
