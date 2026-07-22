"""Real-time replay validator for LiveVO (COR-147, Step A).

Feeds recorded frames at their TRUE sim cadence into LiveVO's front-end, so we
can prove — without touching the live sim — that:
  * the front-end hot path (submit/track) stays real-time (no growing backlog),
  * the back-end keeps up with keyframe solves (few/no drops),
  * OdomDeltas emit live via the on_step callback.

This is the offline stand-in for the in-sim run; the only remaining piece for
true live is the vq2wp frame tap (state['frame']), which reuses this exact
LiveVO.submit() call.

Usage: python -m vq2.tools.vo_live_replay <corpus> [--speed 1.0] [--max N]
"""
import argparse
import json
import os
import time

import numpy as np
import cv2

from vq2.live.vo_live import LiveVO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--speed", type=float, default=1.0, help="playback x realtime")
    ap.add_argument("--max", type=int, default=100000)
    args = ap.parse_args()

    fd = os.path.join(args.corpus, "frames_dedup.jsonl")
    ns = [json.loads(l)["sim_ns"] for l in open(fd)][: args.max]
    files = [os.path.join(args.corpus, "frames", f"{n}.jpg") for n in ns]

    emitted = []
    last_emit = {"t": None}
    def on_step(step):
        emitted.append(step)
        last_emit["t"] = time.perf_counter()

    vo = LiveVO(on_step=on_step, kf_flow_px=9.0)

    # DECODE-AHEAD: in the live system cam_loop decodes jpeg on its own thread
    # and hands VO an already-decoded frame, so decode is NOT on the VO path.
    # Pre-decode here to time the VO path faithfully (not jpeg decode).
    print(f"{os.path.basename(args.corpus)}: decoding {len(files)} frames...")
    imgs = [(n, cv2.imread(f, cv2.IMREAD_GRAYSCALE)) for n, f in zip(ns, files)]
    imgs = [(n, im) for n, im in imgs if im is not None]
    print(f"feeding {len(imgs)} frames at {args.speed}x realtime")
    t_wall0 = time.perf_counter()
    ns0 = imgs[0][0]
    behind = []
    for n, img in imgs:
        # sleep until this frame's scheduled wall time (sim cadence / speed)
        sched = t_wall0 + (n - ns0) * 1e-9 / args.speed
        dt = sched - time.perf_counter()
        if dt > 0:
            time.sleep(dt)
        else:
            behind.append(-dt * 1000)          # front-end fell behind schedule
        vo.submit(n * 1e-9, img)

    vo.close(drain=True)
    s = vo.stats
    print(s.summary())
    behind = np.array(behind) if behind else np.array([0.0])
    print(f"frames behind schedule: {len(behind)} / {s.submitted}  "
          f"(max lag {behind.max():.0f} ms, p90 {np.percentile(behind,90):.0f} ms)")
    print(f"OdomDeltas emitted live: {len(emitted)}")
    fwd = np.array([st.t_body_unit for st in emitted]) if emitted else np.zeros((1, 3))
    print(f"emitted +x-forward dominant: {np.mean(np.argmax(np.abs(fwd),axis=1)==0):.0%}")
    drop_rate = s.dropped / max(s.keyframes, 1)
    verdict = ("REAL-TIME OK" if (np.percentile(behind, 90) < 15 and drop_rate < 0.15)
               else "STRESSED - see lag/drops")
    print(f"VERDICT: {verdict} (keyframe drop rate {drop_rate:.0%})")


if __name__ == "__main__":
    main()
