"""G0 focal-length adjudication (evidence only, no approval).

Non-circular calibration of the sim FPV camera focal length from the KNOWN
1.5 m G0 aperture geometry, independent of GateNet's own depth assumption.

Given a directory of fresh-reset stationary G0 frames (640x360 jpgs), this:
  1. Localizes the four aperture corners by red-frame / blue-interior
     segmentation (median over N frames; front-rim and interior brackets).
  2. Feeds them to ``vq2.g0_calibration.compare_hypotheses`` to report, per
     focal hypothesis, reproj RMS + orthogonality/equal-scale residuals +
     the implied camera-to-G0 depth, compared to the certified ~10.6 m datum.

It NEVER edits ``vq2.camera`` and NEVER marks a model approved.

Usage:
  python -m vq2.tools.g0_focal_adjudication <frames_dir> [--n 30]
"""
from __future__ import annotations
import argparse, glob, os, sys
import cv2
import numpy as np
from scipy.optimize import brentq

from vq2.g0_calibration import compare_hypotheses, _OBJECT_3D

CERTIFIED_DEPTH_M = 10.595   # Janahan datum: G0 [10.595, 0.051, -0.2472] reset-NED
CX, CY = 319.5, 179.5        # (W-1)/2,(H-1)/2 pixel-centre convention @640x360


def _redness(im):
    im = im.astype(np.float32)
    b, g, r = im[:, :, 0], im[:, :, 1], im[:, :, 2]
    return r - np.maximum(g, b)


def interior_quad(im):
    """Aperture corners (TL,TR,BR,BL) from the largest non-red interior blob."""
    red = _redness(im)
    roi = np.zeros(red.shape, np.uint8)
    roi[135:200, 290:352] = 1
    interior = ((red < 10) & (roi == 1)).astype(np.uint8)
    interior = cv2.morphologyEx(interior, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(interior, 8)
    best = max(range(1, n), key=lambda i: stats[i, cv2.CC_STAT_AREA])
    mask = (lab == best).astype(np.uint8)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    peri = cv2.arcLength(c, True)
    ap = None
    for eps in np.linspace(0.01, 0.08, 20):
        ap = cv2.approxPolyDP(c, eps * peri, True)
        if len(ap) == 4:
            break
    ap = ap.reshape(-1, 2).astype(float)
    s, d = ap.sum(1), np.diff(ap, axis=1).ravel()
    return np.array([ap[np.argmin(s)], ap[np.argmin(d)], ap[np.argmax(s)], ap[np.argmax(d)]])


def pnp_range(corners, fx, fy=None, cx=CX, cy=CY):
    fy = fy or fx
    k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]])
    ok, rvec, tvec = cv2.solvePnP(_OBJECT_3D, corners.astype(np.float64), k, None,
                                  flags=cv2.SOLVEPNP_ITERATIVE)
    t = tvec.reshape(3)
    return t, float(np.linalg.norm(t))


def adjudicate(frames_dir, n=30):
    files = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    if not files:
        raise SystemExit(f"no jpg frames in {frames_dir}")
    im0 = cv2.imread(files[0])
    print(f"frames: {len(files)}   resolution: {im0.shape[1]}x{im0.shape[0]}")
    quads = np.array([interior_quad(cv2.imread(f)) for f in files[:n]])
    interior = np.median(quads, 0)
    # front rim ~ interior top raised by the frame-thickness band; widen ~4px.
    front = interior + np.array([[-1.7, -7.6], [2.1, -8.2], [1.2, -1.7], [-1.9, -0.2]])

    for name, C in (("interior(mid-rim)", interior), ("front-rim", front)):
        w = ((C[1, 0] - C[0, 0]) + (C[2, 0] - C[3, 0])) / 2
        h = ((C[3, 1] - C[0, 1]) + (C[2, 1] - C[1, 1])) / 2
        print(f"\n=== corner set: {name}  (mean {w:.1f}w x {h:.1f}h px, "
              f"centre {C[:,0].mean():.1f},{C[:,1].mean():.1f}) ===")
        rep = compare_hypotheses(C, cx=CX, cy=CY, certified_depth_m=CERTIFIED_DEPTH_M)
        print(f"  {'model':13s}{'fx':>6}{'rmsPx':>8}{'orth':>8}{'eqscale':>9}"
              f"{'Zopt':>7}{'range':>7}{'dVsCert':>9}")
        for r in rep["models"]:
            if not r.get("available", True):
                print(f"  {r['id']:13s} unavailable ({r['reason'][:34]})")
                continue
            t = np.array(r["t_cam_gate_m"])
            print(f"  {r['id']:13s}{r['fx']:6.0f}{r['reproj_rms_px']:8.2f}"
                  f"{r['orthogonality_residual']:+8.3f}{r['equal_scale_residual']:+9.3f}"
                  f"{t[2]:7.2f}{np.linalg.norm(t):7.2f}{t[2]-CERTIFIED_DEPTH_M:+9.2f}")
        fx10 = brentq(lambda f: pnp_range(C, f)[1] - CERTIFIED_DEPTH_M, 80, 900)
        print(f"  -> focal that reproduces certified {CERTIFIED_DEPTH_M} m: fx={fx10:.0f}")

    print("\nresolution/crop: H2 fx=width/2=320 is defined @640 -> applies directly here.")
    print("VO scale: gate-PnP range prop-to focal; 226->recommended 320 factor = "
          f"{320/226:.3f}; 0.30 m/unit -> {0.30*320/226:.3f} m/unit.")
    print("evidence only -- NOT approved.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("frames_dir")
    ap.add_argument("--n", type=int, default=30)
    a = ap.parse_args()
    adjudicate(a.frames_dir, a.n)
