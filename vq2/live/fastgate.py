"""FASTGATE: millisecond classical gate detector.
Orange emissive frame + dark interior hole -> aperture CENTER (the hole itself,
not the PnP origin) + range from known hole width (1.5 m). Pure CPU/cv2.

detect(img) -> list of (t_cam[x right, y down, z fwd], hole_w_px, score)
Pinhole: fx=fy=320, cx=320, cy=180 (sim intrinsics INTR=[320,320,320,180]).
"""
import numpy as np, cv2

FX = FY = 320.0
CX, CY = 320.0, 180.0
HOLE_W_M = 1.5          # inner aperture width (course truth)


def detect(img, min_hole_px=8):
    """Find orange gate frames with a dark hole; return camera-frame vectors to
    each hole center, sorted by hole size (largest = nearest first).
    Each det: (t_cam, hole_w_px, area, clipped) -- clipped=True when the hole
    touches the image border (bearing usable, range unreliable)."""
    b, g, r = [c.astype(np.int16) for c in cv2.split(img)]
    # emissive orange INCLUDING the washed pink-white close-range core, which
    # picks up blue from the cyan trace glow (measured r246 g183 b220): key on
    # red-over-green; allow blue up to r+30. White ceiling lights (r~g~b) and
    # the cyan trace (r<g) stay excluded.
    mask = ((r > 120) & (r - g > 25) & (r - b > -30)).astype(np.uint8) * 255
    # 9x9 close: the white text/checker bands on the gate face fail the color
    # test and BREAK the orange ring (5x5 left the fg9 chute hole un-enclosed
    # -> undetected while plainly visible); 9x9 seals them.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    # horizontal close bridges THIN VERTICAL occluders (the start-light pole,
    # ~10-20 px, bisects the ring AND the blob -> blind approach, fg10/fg12)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((1, 25), np.uint8))
    cnts, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    if hier is None:
        return out
    hier = hier[0]
    for i, c in enumerate(cnts):
        if hier[i][3] != -1:        # want OUTER contours (gate frame)
            continue
        area = cv2.contourArea(c)
        if area < 150:
            continue
        # must have a child contour = the dark hole
        child = hier[i][2]
        best_hole = None
        while child != -1:
            ha = cv2.contourArea(cnts[child])
            if best_hole is None or ha > best_hole[0]:
                best_hole = (ha, child)
            child = hier[child][0]
        if best_hole is None or best_hole[0] < min_hole_px ** 2:
            # FRAME-CENTROID FALLBACK (07-14 fg10/fg12): the start-light pole
            # bisects the gate ring -> no enclosed hole -> blind approach. A
            # pole can't erase the big orange blob: for a LARGE frame with no
            # hole, emit the outer bbox center (~hole center by symmetry).
            if area > 2500:
                fx_, fy_, fw_, fh_ = cv2.boundingRect(c)
                u, v = fx_ + fw_ / 2.0, fy_ + fh_ / 2.0
                # hole ~= 0.62x the outer frame width (1.5 m hole / ~2.4 m frame)
                hw_est = 0.62 * max(fw_, fh_)
                z = FX * HOLE_W_M / max(hw_est, 1.0)
                t_cam = np.array([(u - CX) / FX * z, (v - CY) / FY * z, z])
                H, W = img.shape[:2]
                clipped = fx_ <= 1 or fy_ <= 1 or fx_ + fw_ >= W - 2 or fy_ + fh_ >= H - 2
                out.append((t_cam, float(hw_est), float(area), clipped))
            continue
        hx, hy, hw, hh = cv2.boundingRect(cnts[best_hole[1]])
        # hole must be roughly square-ish and a sane fraction of the frame blob
        if hw < min_hole_px or hh < min_hole_px:
            continue
        ar = hw / max(hh, 1)
        # oblique views: a square hole seen at 70 deg reads ar ~0.3 (fg9 gap:
        # the chute hole was plainly visible but tall-narrow and got rejected
        # here while far bay gates passed). Keep only a sliver floor.
        if ar > 3.5 or ar < 0.15:
            continue
        H, W = img.shape[:2]
        clipped = hx <= 1 or hy <= 1 or hx + hw >= W - 2 or hy + hh >= H - 2
        u, v = hx + hw / 2.0, hy + hh / 2.0
        # range from the larger visible hole dimension (less clipped side)
        z = FX * HOLE_W_M / max(hw, hh)
        t_cam = np.array([(u - CX) / FX * z, (v - CY) / FY * z, z])
        out.append((t_cam, float(hw), float(area), clipped))
    out.sort(key=lambda o: -o[1])                   # largest hole first
    return out


if __name__ == '__main__':
    import sys, time, json, math
    corp = sys.argv[1] if len(sys.argv) > 1 else 'vq2_servo_nt13'
    rows = sorted(json.loads(l)['sim_ns'] for l in
                  open(f'C:/Users/alexj/{corp}/frames_dedup.jsonl') if l.strip())
    lat = []
    det_by_range = {}
    n_det = 0; n_tot = 0
    for ns in rows[::3]:
        img = cv2.imread(f'C:/Users/alexj/{corp}/frames/{ns}.jpg')
        if img is None:
            continue
        n_tot += 1
        t0 = time.perf_counter()
        dets = detect(img)
        lat.append((time.perf_counter() - t0) * 1000)
        if dets:
            n_det += 1
            rng = float(np.linalg.norm(dets[0][0]))
            det_by_range.setdefault(int(rng), 0)
            det_by_range[int(rng)] += 1
    lat = np.array(lat)
    print(f'{corp}: {n_tot} frames, detected in {n_det} ({100*n_det/max(n_tot,1):.0f}%)')
    print(f'latency ms: median {np.median(lat):.2f}  p90 {np.percentile(lat,90):.2f}  max {lat.max():.2f}')
    print('detections by integer range (m):', dict(sorted(det_by_range.items())[:14]))
