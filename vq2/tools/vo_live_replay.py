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
    ap.add_argument("--gpu", action="store_true", help="CUDA-graph GPU optical flow")
    args = ap.parse_args()

    fd = os.path.join(args.corpus, "frames_dedup.jsonl")
    ns = [json.loads(l)["sim_ns"] for l in open(fd)][: args.max]
    files = [os.path.join(args.corpus, "frames", f"{n}.jpg") for n in ns]

    emitted = []
    last_emit = {"t": None}
    def on_step(step):
        emitted.append(step)
        last_emit["t"] = time.perf_counter()

    vo = LiveVO(on_step=on_step, kf_flow_px=9.0, gpu=args.gpu)
    print(f"optical flow backend: {'GPU (CUDA graph)' if args.gpu else 'CPU (cv2)'}")

    # DECODE-AHEAD: in the live system cam_loop decodes jpeg on its own thread
    # and hands VO an already-decoded frame, so decode is NOT on the VO path.
    # Pre-decode here to time the VO path faithfully (not jpeg decode).
    print(f"{os.path.basename(args.corpus)}: decoding {len(files)} frames...")
    imgs = [(n, cv2.imread(f, cv2.IMREAD_GRAYSCALE)) for n, f in zip(ns, files)]
    imgs = [(n, im) for n, im in imgs if im is not None]
    # pre-warm off the clock (GPU CUDA-graph capture is a one-time ~1-4s cost)
    for n, img in imgs[:3]:
        vo.submit(n * 1e-9, img)
    time.sleep(0.2)
    vo.stats = type(vo.stats)()               # reset stats after warmup

    print(f"feeding {len(imgs)} frames at {args.speed}x realtime")
    submit_ms = []
    ns0 = imgs[3][0]
    t_wall0 = time.perf_counter()
    for n, img in imgs[3:]:
        # sleep until this frame's scheduled wall time (sim cadence / speed)
        sched = t_wall0 + (n - ns0) * 1e-9 / args.speed
        dt = sched - time.perf_counter()
        if dt > 0:
            time.sleep(dt)
        t0 = time.perf_counter()
        vo.submit(n * 1e-9, img)
        submit_ms.append((time.perf_counter() - t0) * 1000)   # front-end hot-path cost

    vo.close(drain=True)
    s = vo.stats
    print(s.summary())
    sm = np.array(submit_ms)
    BUDGET = 33.0 * args.speed
    over = float(np.mean(sm > BUDGET))
    print(f"FRONT-END submit() ms: p50={np.percentile(sm,50):.1f} p90={np.percentile(sm,90):.1f} "
          f"p99={np.percentile(sm,99):.1f} | over {BUDGET:.0f}ms budget: {over:.0%}")
    drop_rate = s.dropped / max(s.keyframes, 1)
    print(f"BACK-END keep-up: solved={s.solved} dropped={s.dropped} (drop rate {drop_rate:.0%})")
    print(f"OdomDeltas emitted live: {len(emitted)}")
    fwd = np.array([st.t_body_unit for st in emitted]) if emitted else np.zeros((1, 3))
    print(f"emitted +x-forward dominant: {np.mean(np.argmax(np.abs(fwd),axis=1)==0):.0%}")
    fe = "OK" if over < 0.05 else "STRESSED"
    be = "OK" if drop_rate < 0.15 else "STRESSED"
    print(f"VERDICT: front-end {fe} (real-time hot path) | back-end {be} (solve keep-up)")


if __name__ == "__main__":
    main()
