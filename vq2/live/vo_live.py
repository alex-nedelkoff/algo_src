"""Live threaded VO consumer (COR-147, Phase A / Step A).

Front-end/back-end split so the real-time hot path stays cheap:
  * FRONT-END  (caller's thread, ~10ms): submit(ns, gray) -> MonoVO.track().
    KLT tracking + keyframe management only. Never blocks on the solve.
  * BACK-END   (worker thread, ~50ms): drains a bounded queue of track jobs,
    runs the essential-matrix solve, and calls on_step(VoStep) + optionally
    lowers to an OdomDelta.

Latency measured (test46 motion window, this box): tracking-only ~10ms/frame,
keyframe solve 47-70ms. The sim stream is ~30Hz (33ms), so the solve MUST be
off the hot path — hence this split. If the back-end falls behind, jobs are
dropped (logged) rather than stalling the front-end; a dropped keyframe just
widens the next baseline.

Source-agnostic: drive submit() from vq2wp's cam_loop (state['frame']) for a
live flight, or from a real-time replay feeder for offline validation.
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from .vo_cv import MonoVO, VoStep


@dataclass
class LiveStats:
    submitted: int = 0
    keyframes: int = 0
    solved: int = 0
    dropped: int = 0
    track_ms: list = field(default_factory=list)
    solve_ms: list = field(default_factory=list)

    def summary(self) -> str:
        def pct(a, p):
            return float(np.percentile(a, p)) if a else 0.0
        return (f"submitted={self.submitted} keyframes={self.keyframes} "
                f"solved={self.solved} dropped={self.dropped} | "
                f"track ms p50/p90={pct(self.track_ms,50):.1f}/{pct(self.track_ms,90):.1f} "
                f"solve ms p50/p90={pct(self.solve_ms,50):.1f}/{pct(self.solve_ms,90):.1f}")


class LiveVO:
    """Threaded live VO. Call submit(ns, gray) per frame from the hot path;
    on_step(VoStep) fires from the back-end thread as keyframes resolve."""

    def __init__(self, on_step=None, kf_flow_px: float = 9.0,
                 queue_max: int = 3, **mono_kw):
        self._vo = MonoVO(kf_flow_px=kf_flow_px, **mono_kw)
        self._on_step = on_step
        self._q: queue.Queue = queue.Queue(maxsize=queue_max)
        self._stop = threading.Event()
        self.stats = LiveStats()
        self._t = threading.Thread(target=self._backend, name="vo-backend", daemon=True)
        self._t.start()

    def submit(self, ns: float, gray: np.ndarray) -> None:
        """Front-end hot path. ns in seconds. Non-blocking: enqueues a solve
        job on keyframe close, drops it if the back-end is saturated."""
        t0 = time.perf_counter()
        job = self._vo.track(ns, gray)
        self.stats.track_ms.append((time.perf_counter() - t0) * 1000)
        self.stats.submitted += 1
        if job is None:
            return
        self.stats.keyframes += 1
        try:
            self._q.put_nowait(job)
        except queue.Full:
            # back-end behind: evict the OLDEST (stale) job and keep this fresh
            # one, so emitted deltas stay low-latency rather than seconds stale.
            try:
                self._q.get_nowait()
                self.stats.dropped += 1
                self._q.put_nowait(job)
            except (queue.Empty, queue.Full):
                self.stats.dropped += 1

    def _backend(self) -> None:
        while not self._stop.is_set():
            try:
                job = self._q.get(timeout=0.1)
            except queue.Empty:
                continue
            t0 = time.perf_counter()
            step = self._vo.solve_job(job)
            self.stats.solve_ms.append((time.perf_counter() - t0) * 1000)
            if step is not None:
                self.stats.solved += 1
                if self._on_step is not None:
                    try:
                        self._on_step(step)
                    except Exception:
                        pass               # a bad consumer must not kill VO

    def close(self, drain: bool = True, timeout: float = 2.0) -> None:
        if drain:
            t_end = time.perf_counter() + timeout
            while not self._q.empty() and time.perf_counter() < t_end:
                time.sleep(0.01)
        self._stop.set()
        self._t.join(timeout=timeout)
