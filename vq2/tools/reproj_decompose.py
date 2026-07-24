"""Leg 2: vectorially decompose the ~20deg pad reprojection bias.

Replicates vq2wp's identity-check prediction chain EXACTLY
(rel_ = (_cmap - p_kf) @ (Rz Ry Rx @ M_BODY_CAM); uv = x/z*320+319.5,
y/z*320+179.5) on mig1 pad frames vs of-record GateNet corners, then sweeps
(pitch, tilt, camera z) variants to find which term carries the error.
"""
import json, math
import numpy as np

CM = np.load(r"C:\Users\Administrator\g2rib_corners_world.npy")   # _cmap slots
DET = r"C:\Users\Administrator\algo_src-mig\work\mig1_pad_detections.jsonl"
FX, CX, CY = 320.0, 319.5, 179.5


def m_body_cam(tilt_rad):
    ct, st = math.cos(tilt_rad), math.sin(tilt_rad)
    cz = np.array([ct, 0.0, -st])
    cy = np.array([st, 0.0, ct])
    cx = np.cross(cy, cz)
    return np.stack([cx, cy, cz], axis=1)


def predict(p_kf, roll, pitch, yaw, tilt_deg):
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    cy_, sy_ = math.cos(yaw), math.sin(yaw)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Rz = np.array([[cy_, -sy_, 0], [sy_, cy_, 0], [0, 0, 1]])
    R_wc = (Rz @ Ry @ Rx) @ m_body_cam(math.radians(tilt_deg))
    rel = (CM - p_kf) @ R_wc
    uv = np.stack([rel[:, 0] / rel[:, 2] * FX + CX,
                   rel[:, 1] / rel[:, 2] * FX + CY], axis=1)
    return uv, rel


def near_inst(d):
    c = [i for i in d["insts"] if i.get("solved") and i.get("t_cam")
         and not i.get("low_confidence")]
    return min(c, key=lambda i: i["t_cam"][2]) if c else None


frames = []
for line in open(DET):
    d = json.loads(line)
    inst = near_inst(d)
    if inst and inst.get("visibility"):
        frames.append(inst)
print(f"frames with near-gate detections: {len(frames)}")

variants = [
    ("as-flown: pitch -17.8, tilt 20", -17.8, 20.0, 0.0),
    ("pitch 0,    tilt 20          ", 0.0, 20.0, 0.0),
    ("pitch -17.8, tilt 0          ", -17.8, 0.0, 0.0),
    ("pitch 0,    tilt 0           ", 0.0, 0.0, 0.0),
    ("pitch 0,    tilt 2.25 (VP)   ", 0.0, 2.25, 0.0),
    ("as-flown + cam z -0.5        ", -17.8, 20.0, -0.5),
    ("as-flown + cam z +0.5        ", -17.8, 20.0, +0.5),
]

print(f"corner map: {CM.shape[0]} slots, world z range "
      f"[{CM[:,2].min():.2f}, {CM[:,2].max():.2f}], "
      f"x range [{CM[:,0].min():.2f}, {CM[:,0].max():.2f}]")

for label, pitch, tilt, zoff in variants:
    du_all, dv_all, n = [], [], 0
    for inst in frames:
        obs = np.asarray(inst["corner_xy"], float)
        vis = np.asarray(inst["visibility"], float)
        uv, rel = predict(np.array([0.0, 0.0, zoff]), 0.0,
                          math.radians(pitch), 0.0, tilt)
        for k in range(min(len(obs), len(CM))):
            if vis[k] < 0.5 or rel[k, 2] <= 0.2:
                continue
            du_all.append(uv[k, 0] - obs[k, 0])
            dv_all.append(uv[k, 1] - obs[k, 1])
            n += 1
    du, dv = np.array(du_all), np.array(dv_all)
    norm = np.hypot(du, dv)
    print(f"{label}  n={n:3d}  err |med| {np.median(norm):6.1f}px  "
          f"du med {np.median(du):+7.1f}  dv med {np.median(dv):+7.1f}  "
          f"(ang {math.degrees(math.atan(np.median(norm)/FX)):+5.1f} deg)")
