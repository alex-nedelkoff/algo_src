"""Re-run COR-142 T2a's vanishing-point tilt method on a frames dir.

Method (per T2a calibration report): Canny -> HoughLinesP -> robust
length-weighted least-squares vanishing point of receding oblique lines;
horizon row v_vp implies tilt_up = atan((v_vp - cy) / fy) for a level body.
IRLS (Huber) replaces their exact robust fit; oblique-angle gate excludes
image-horizontal (transverse) and image-vertical (pillar) lines, which do
not constrain / would bias the receding-line VP.
"""
import sys, glob, os
import numpy as np
import cv2

FY, CY, CX = 320.0, 179.5, 319.5


def frame_vp(img):
    edges = cv2.Canny(img, 60, 160)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=40,
                            minLineLength=40, maxLineGap=6)
    if lines is None:
        return None
    rows = []
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
        ang = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        ang = min(ang, 180 - ang)
        if not (6.0 < ang < 80.0):        # oblique (receding) lines only
            continue
        n = np.array([y2 - y1, x1 - x2], float)   # line normal
        ln = np.linalg.norm(n)
        if ln < 1e-9:
            continue
        n /= ln
        c = n @ np.array([x1, y1], float)
        w = np.hypot(x2 - x1, y2 - y1)            # length weight
        rows.append((n[0], n[1], c, w))
    if len(rows) < 8:
        return None
    A = np.array([[r[0], r[1]] for r in rows])
    b = np.array([r[2] for r in rows])
    w = np.array([r[3] for r in rows])
    p = np.array([CX, CY])
    for _ in range(30):                            # IRLS, Huber delta=3px
        r = A @ p - b
        hub = np.where(np.abs(r) < 3.0, 1.0, 3.0 / np.abs(r))
        W = w * hub
        p_new = np.linalg.lstsq(A * W[:, None], b * W, rcond=None)[0]
        if np.linalg.norm(p_new - p) < 1e-6:
            p = p_new
            break
        p = p_new
    inl = np.abs(A @ p - b) < 5.0
    return p, int(inl.sum()), len(rows)


def main(frames_dir, n=12):
    paths = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    idx = np.linspace(int(0.3 * len(paths)), int(0.8 * len(paths)) - 1, n).astype(int)
    vps = []
    for i in idx:
        img = cv2.imread(paths[i], cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        out = frame_vp(img)
        if out is None:
            continue
        (vx, vy), n_in, n_tot = out
        vps.append((vx, vy, n_in, n_tot))
    vps = np.array(vps)
    vy_med, vy_std = np.median(vps[:, 1]), vps[:, 1].std()
    vx_med = np.median(vps[:, 0])
    tilt = np.degrees(np.arctan2(vy_med - CY, FY))
    print(f"frames used {len(vps)}  inlier lines median {int(np.median(vps[:,2]))}/{int(np.median(vps[:,3]))}")
    print(f"VP: x median {vx_med:.1f} (cx {CX}), y median {vy_med:.1f} +- {vy_std:.1f} (cy {CY})")
    # horizon BELOW centre (v_vp > cy) <=> camera tilted UP: tilt_up = +atan((v-cy)/fy)
    print(f"implied tilt_up (level body, fy={FY:.0f}): {tilt:+.2f} deg")
    print("T2a reference: horizon row 288.0 +- 2.7 => tilt 18.6 +- 0.4 deg")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 12)
