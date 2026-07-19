"""Offline validation of GATE-RELATIVE pose from the aperture quad.

Signal 1 (robust): edge-height skew of the hole -> obliquity angle
  sin(theta) ~= 640 * s / w_px   with s = (h_l - h_r)/(h_l + h_r)
Signal 2 (full): approxPolyDP 4-corner quad -> solvePnP(IPPE_SQUARE)
  -> gate-frame yaw + lateral offset.

Truth cases:
  vq2_test76 = THE TICK  (punched at yaw 0.78 ~ on-course -> theta ~ 0)
  vq2_test75 = on-course near miss (theta ~ 0)
  vq2_test90 = oblique miss (punched at yaw 1.21 vs course 0.77 ->
               theta ~ 25 deg, hole slid left at bloom)
"""
import sys, os, json, glob, math
import numpy as np, cv2

sys.path.insert(0, r'C:\Users\Administrator')
import fastgate

FX, CX, CY = 320.0, 320.0, 180.0
W_M = 1.5

def hole_quad_metrics(img):
    """Largest unclipped-ish hole -> (w_px, b0, skew, theta_skew_deg,
    theta_pnp_deg, lat_off_m) or None."""
    b, g, r = [c.astype(np.int16) for c in cv2.split(img)]
    mask = ((r > 120) & (r - g > 25) & (r - b > -30)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((1, 25), np.uint8))
    cnts, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hier is None:
        return None
    hier = hier[0]
    best = None
    for i, c in enumerate(cnts):
        if hier[i][3] != -1 or cv2.contourArea(c) < 150:
            continue
        child = hier[i][2]
        while child != -1:
            ha = cv2.contourArea(cnts[child])
            if ha >= 64 and (best is None or ha > best[0]):
                best = (ha, child)
            child = hier[child][0]
    if best is None:
        return None
    hc = cnts[best[1]][:, 0, :]           # N x (x,y)
    hx, hy, hw, hh = cv2.boundingRect(cnts[best[1]])
    H, W = img.shape[:2]
    if hx <= 1 or hx + hw >= W - 2:
        return None                        # side-clipped: skew invalid
    w_px = float(max(hw, hh))
    u = hx + hw / 2.0
    b0 = (u - CX) / FX
    # edge heights: vertical extent of contour points in the outermost
    # 15% columns each side
    x0, x1 = hx + 0.15 * hw, hx + 0.85 * hw
    left = hc[hc[:, 0] <= x0]; right = hc[hc[:, 0] >= x1]
    if len(left) < 4 or len(right) < 4:
        return None
    h_l = float(left[:, 1].max() - left[:, 1].min())
    h_r = float(right[:, 1].max() - right[:, 1].min())
    if h_l + h_r < 10:
        return None
    s = (h_l - h_r) / (h_l + h_r)
    sin_t = max(-1.0, min(1.0, 640.0 * s / max(w_px, 1.0)))
    th_skew = math.degrees(math.asin(sin_t))
    # full PnP on a 4-corner quad, when clean
    th_pnp = None; lat = None
    peri = cv2.arcLength(cnts[best[1]], True)
    quad = cv2.approxPolyDP(cnts[best[1]], 0.03 * peri, True)
    if len(quad) == 4 and cv2.isContourConvex(quad):
        q = quad[:, 0, :].astype(np.float64)
        # order: tl, tr, br, bl
        srt = q[np.argsort(q[:, 1])]
        top = srt[:2][np.argsort(srt[:2][:, 0])]
        bot = srt[2:][np.argsort(srt[2:][:, 0])]
        img_pts = np.array([top[0], top[1], bot[1], bot[0]])
        h2 = W_M / 2
        obj = np.array([[-h2, -h2, 0], [h2, -h2, 0],
                        [h2, h2, 0], [-h2, h2, 0]], dtype=np.float64)
        K = np.array([[FX, 0, CX], [0, FX, CY], [0, 0, 1]])
        ok, rvec, tvec = cv2.solvePnP(obj, img_pts, K, None,
                                      flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if ok:
            R, _ = cv2.Rodrigues(rvec)
            # gate-plane normal in cam frame = R @ [0,0,1]
            nz = R @ np.array([0, 0, 1.0])
            th_pnp = math.degrees(math.atan2(nz[0], nz[2]))
            # camera position in gate frame: -R^T t -> x = lateral offset
            pg = -R.T @ tvec.ravel()
            lat = float(pg[0])
    return w_px, b0, s, th_skew, th_pnp, lat


def terminal_window(corp):
    """Last punch sequence in livelog -> (t_start-5, t_end+0.5) wall."""
    evs = [json.loads(l) for l in
           open(rf'C:\Users\Administrator\{corp}\livelog.jsonl')
           if l.strip()]
    pts = [e['t'] for e in evs if e.get('kind') == 'punch']
    if not pts:
        # fall back to settle window
        pts = [e['t'] for e in evs if e.get('kind') == 'settle']
    t1 = max(pts); t0g = t1
    for t in sorted(pts, reverse=True):
        if t0g - t < 1.5:
            t0g = t
        else:
            break
    return t0g - 5.0, t1 + 0.5


for corp in ('vq2_test76', 'vq2_test75', 'vq2_test90'):
    try:
        w0, w1 = terminal_window(corp)
    except Exception as e:
        print(corp, 'no window:', e); continue
    fr = sorted(glob.glob(rf'C:\Users\Administrator\{corp}\frames\*.jpg'))
    print(f'\n=== {corp} terminal window {w1-w0:.1f}s ===')
    rows = []
    for f in fr:
        ts = int(os.path.basename(f)[:-4]) / 1e9
        if not (w0 <= ts <= w1):
            continue
        img = cv2.imread(f)
        if img is None:
            continue
        m = hole_quad_metrics(img)
        if m is None:
            continue
        w_px, b0, s, th_s, th_p, lat = m
        if w_px < 40 or w_px > 260:
            continue
        rows.append((ts - w0, w_px, b0, s, th_s, th_p, lat))
    for rw in rows[::2]:
        ts, w_px, b0, s, th_s, th_p, lat = rw
        tp = f'{th_p:+6.1f}' if th_p is not None else '  ----'
        lt = f'{lat:+5.2f}' if lat is not None else ' ----'
        print(f' t+{ts:5.2f} w{w_px:4.0f} b0 {b0:+5.2f} skew {s:+5.2f} '
              f'thSkew {th_s:+6.1f} thPnP {tp} lat {lt}')
    if rows:
        th_all = [r[4] for r in rows]
        thp = [r[5] for r in rows if r[5] is not None]
        print(f' -> median thSkew {np.median(th_all):+5.1f} deg over '
              f'{len(rows)} frames; PnP on {len(thp)} '
              f'(median {np.median(thp):+5.1f})' if thp else
              f' -> median thSkew {np.median(th_all):+5.1f} deg, no PnP')
