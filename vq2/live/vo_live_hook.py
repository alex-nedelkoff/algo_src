"""VO_LIVE hook (COR-147 Step A): live VO consumer that taps vq2wp's live
camera stream (state['frame'], populated by cam_loop) and logs OdomDeltas +
latency. READ-ONLY on state — no control impact, so it is safe to run during
any flight. Gated by VO_LIVE=1; backend GPU by default (VO_GPU=1) with a CPU
fallback if CUDA-graph capture fails mid-flight (e.g. GateNet CUDA contention).

Wire-in (deployed vq2wp.py, after cam_loop is started):
    if os.environ.get('VO_LIVE') == '1':
        sys.path.insert(0, r'C:\\Users\\Administrator\\algo_src-vo')
        from vq2.live.vo_live_hook import start_vo_live
        start_vo_live(state)
"""
from __future__ import annotations

import json
import os
import threading
import time

import numpy as np

try:
    import cv2
except Exception:                      # pragma: no cover
    cv2 = None

from .vo_live import LiveVO


def start_vo_live(state, log_path: str | None = None, gpu: bool | None = None,
                  kf_flow_px: float | None = None):
    """Spawn the live VO thread. Returns the thread (daemon)."""
    if gpu is None:
        gpu = os.environ.get("VO_GPU", "1") == "1"
    if kf_flow_px is None:
        # higher live default (fewer keyframes -> fewer CPU solves) than the
        # offline 9px; wider baselines are also better-conditioned.
        kf_flow_px = float(os.environ.get("VO_KFFLOW", "13"))
    log_path = log_path or os.environ.get("VO_LOG", r"C:\Users\Administrator\vo_live.jsonl")
    t = threading.Thread(target=_run, args=(state, log_path, gpu, kf_flow_px),
                         name="vo-live", daemon=True)
    t.start()
    print(f"[VO_LIVE] started (gpu={gpu}) -> {log_path}", flush=True)
    return t


def _run(state, log_path, gpu, kf_flow_px):
    logf = open(log_path, "w")

    def jlog(**k):
        logf.write(json.dumps(k) + "\n")
        logf.flush()

    count = [0]

    def on_step(step):
        count[0] += 1
        jlog(kind="vo", t0=round(step.t0, 4), t1=round(step.t1, 4),
             dp=[round(float(x), 4) for x in step.t_body_unit],
             dyaw=round(float(step.dyaw), 4), ni=int(step.n_inliers),
             flow=round(float(step.median_flow_px), 1))

    try:
        vo = LiveVO(on_step=on_step, kf_flow_px=kf_flow_px, gpu=gpu)
    except Exception as e:                      # CUDA-graph capture can fail
        jlog(kind="vo_init_fail", gpu=gpu, err=str(e))
        vo = LiveVO(on_step=on_step, kf_flow_px=kf_flow_px, gpu=False)
        jlog(kind="vo_fallback_cpu")

    jlog(kind="vo_ready", gpu=gpu)
    last_ns = 0
    t_stat = time.time()
    while not state.get("stop"):
        img = state.get("frame")
        ns = state.get("frame_ns")
        if img is None or ns == last_ns:
            time.sleep(0.004)
            continue
        last_ns = ns
        gray = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                if getattr(img, "ndim", 2) == 3 else img)
        vo.submit(ns * 1e-9, gray)
        if time.time() - t_stat > 2.0:
            jlog(kind="vo_stat", s=vo.stats.summary())
            t_stat = time.time()

    vo.close()
    jlog(kind="vo_done", n_deltas=count[0], summary=vo.stats.summary())
    logf.close()
