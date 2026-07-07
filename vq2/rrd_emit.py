"""Emit a per-flight Rerun recording (.rrd) from a corpus + report.npz.

Run under the rerun-0.33 venv (NOT the system 3.9, whose rerun is 0.26):

    ~/.rerun33-venv/bin/python vq2/rrd_emit.py <corpus_dir> [out.rrd]

Deliberately self-contained: no vq2 imports (that package lives in the 3.9
env). Reads mavlink.jsonl / frames_dedup.jsonl / detections.jsonl / frames/
plus the report.npz written by `python3 -m vq2.flight_report`.

Layout on a single 'boot_s' timeline (scrub the FPV stream, everything
follows):
    camera/fpv           raw jpegs (no re-encode)
    camera/detections    2D boxes colored by waterfall stage
    world/est_path, world/gates
    plots/*              rates, vz, z, |a_w_xy| leak, obs miss, NIS
    events               TextLog: rejections w/ reason, phase markers
"""
from __future__ import annotations

import glob
import json
import os
import statistics
import sys

import numpy as np
import rerun as rr

W, H = 640, 360
NIS_GATE = 11.34
STAGE_COLOR = {
    "accepted": (0, 220, 0),
    "policy_reject": (255, 200, 0),
    "maha_reject": (255, 60, 60),
}


def load_mavlink(root):
    imu, last = [], None
    segs = [[]]
    with open(os.path.join(root, "mavlink.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            if d.get("mavpackettype") != "HIGHRES_IMU":
                continue
            t = d["time_usec"]
            if last is not None and t < last - 1_000_000:
                segs.append([])
            last = t
            segs[-1].append((t, d["xgyro"], d["ygyro"], d["zgyro"],
                             d.get("_rx_wall", 0.0)))
    return segs[-1]  # flight segment = last stamp epoch (corpus.py convention)


def clock_offset(imu, root):
    """camera epoch-s -> boot-s (median rx_wall bridge, as vq2/corpus.py)."""
    rows = []
    with open(os.path.join(root, "frames_dedup.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            if d.get("rx_wall", 0) > 0:
                rows.append(d["rx_wall"] - d["sim_ns"] / 1e9)
    if not rows or not imu:
        return None
    imu_off = statistics.median(w - t / 1e6 for t, _, _, _, w in imu[::50] if w)
    return statistics.median(rows) - imu_off


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    root = argv[0].rstrip("/")
    out = argv[1] if len(argv) > 1 else os.path.join(root, "flight.rrd")
    npz = np.load(os.path.join(root, "report.npz"), allow_pickle=False)

    rr.init("vq2_flight", spawn=False)
    rr.save(out)

    imu = load_mavlink(root)
    off = clock_offset(imu, root)
    t0 = imu[0][0] / 1e6 if imu else float(npz["t_s"][0])

    # --- IMU-rate plots (decimate 2x: ~72 Hz plenty for the eye)
    for t_us, gx, gy, gz, _ in imu[::2]:
        tb = t_us / 1e6
        rr.set_time("boot_s", duration=tb - t0)
        rr.log("plots/rate_max_deg_s",
               rr.Scalars(float(np.degrees(max(abs(gx), abs(gy), abs(gz))))))

    # --- estimator timeline from report.npz
    t_s = npz["t_s"]; p = npz["p"]; v = npz["v"]; awxy = npz["a_w_xy"]
    step = max(1, len(t_s) // 4000)
    for i in range(0, len(t_s), step):
        rr.set_time("boot_s", duration=float(t_s[i]) - t0)
        rr.log("plots/z_m", rr.Scalars(float(p[i][2])))
        rr.log("plots/vz", rr.Scalars(float(v[i][2])))
        rr.log("plots/leak_a_w_xy", rr.Scalars(
            float(np.hypot(awxy[i][0], awxy[i][1]))
            if i < len(awxy) else 0.0))
        rr.log("world/drone", rr.Points3D([p[i].tolist()], radii=0.1,
                                          colors=[80, 255, 80]))
    rr.log("world/est_path", rr.LineStrips3D(
        [p[::step].tolist()], colors=[80, 255, 80]), static=True)
    rr.log("world/gates", rr.Points3D(
        npz["gates"].tolist(), radii=0.3, colors=[255, 60, 60]), static=True)

    # --- obs waterfall
    for i in range(len(npz["obs_t"])):
        tb = float(npz["obs_t"][i])
        stage = str(npz["obs_stage"][i])
        rr.set_time("boot_s", duration=tb - t0)
        rr.log("plots/obs_miss_m", rr.Scalars(float(npz["obs_miss"][i])))
        nis = float(npz["obs_nis"][i])
        if np.isfinite(nis):
            rr.log("plots/nis", rr.Scalars(nis))
        if stage != "accepted":
            rr.log("events", rr.TextLog(
                f"obs {stage} miss={npz['obs_miss'][i]:.2f}",
                level=rr.TextLogLevel.WARN))

    # --- camera stream + detection overlays
    dets = {}
    dpath = os.path.join(root, "detections.jsonl")
    if os.path.exists(dpath):
        with open(dpath) as f:
            for line in f:
                d = json.loads(line)
                dets[d["sim_ns"]] = d["insts"]
    frames = sorted(glob.glob(os.path.join(root, "frames", "*.jpg")))
    obs_t = npz["obs_t"]; obs_stage = npz["obs_stage"]
    for fp in frames:
        stem = os.path.basename(fp)[:-4]
        if not stem.isdigit():
            continue
        ns = int(stem)
        if off is None:
            continue
        tb = ns / 1e9 + off
        if not (t0 - 1 <= tb <= float(t_s[-1]) + 1):
            continue  # other-session frame in an accumulated dir
        rr.set_time("boot_s", duration=tb - t0)
        with open(fp, "rb") as fh:
            rr.log("camera/fpv", rr.EncodedImage(contents=fh.read(),
                                                 media_type="image/jpeg"))
        insts = dets.get(ns) or []
        if insts:
            # color each box by the waterfall stage of the nearest obs event
            centers, sizes, colors = [], [], []
            for inst in insts:
                cx, cy = inst.get("center_xy", (0, 0))
                corners = inst.get("corner_xy") or []
                if corners:
                    xs = [c[0] for c in corners]; ys = [c[1] for c in corners]
                    w, h = max(xs) - min(xs), max(ys) - min(ys)
                else:
                    w = h = 40.0
                k = int(np.argmin(np.abs(obs_t - tb))) if len(obs_t) else -1
                stage = str(obs_stage[k]) if (
                    k >= 0 and abs(float(obs_t[k]) - tb) < 0.1) else "none"
                centers.append([cx, cy]); sizes.append([w, h])
                colors.append(STAGE_COLOR.get(stage, (140, 140, 140)))
            rr.log("camera/detections", rr.Boxes2D(
                centers=centers, sizes=sizes, colors=colors))
        else:
            rr.log("camera/detections", rr.Clear(recursive=False))

    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
