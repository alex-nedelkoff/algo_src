"""livelog.jsonl capture-time join (COR-147).

The blocker: in `livelog.jsonl`, estimator POSITION and frame CAPTURE TIME never
co-occur in one record. Position lives in `kf_upd`/`mapfollow`/`tick_fix`/
`mfdr_seed` rows (3-vector `p`, world/reset-NED); those rows carry only the loop
WALL clock `t` (float seconds), NOT the camera capture time. The camera capture
time (`ns`, sim nanoseconds, SAME namespace as the frame jpg filenames /
frames_dedup.jsonl `sim_ns`) is logged ONLY on `obs*` rows.

Old approach (naive carry-forward, `vo_replay.load_reference`): stamp each
position row with the LAST-SEEN `ns`. Because the loop logs the position update
and its obs in the same iteration but the last obs seen while reading a position
row is the PREVIOUS iteration's frame, carry-forward is off by ~one loop
iteration (measured median 0.37-0.67 s across test95-99/46/140, max several s in
fast segments). That is larger than a keyframe interval, so the estimator track
cannot be reliably aligned to VO keyframe capture times.

The linkage (measured, see COR-147 notes): within ONE estimator loop iteration
the loop reads a frame captured at `ns`, runs the obs, updates the KF, and logs
`kf_upd`(position) and `obs`(ns) sharing an essentially identical `t` (<1 ms
apart). So `obs*` rows are the only place the loop wall clock `t` and the frame
capture clock `ns` co-occur -> they calibrate a monotone t->ns transform. Each
position row is then placed on the capture-time axis via that transform:
  - when a same-iteration obs exists (16/26 rows in test95), the position row's
    `t` coincides with an anchor `t` and the transform returns that frame's exact
    `ns` (exact per-iteration pairing);
  - otherwise (mapfollow/tick_fix iterations with no co-logged obs) it linearly
    interpolates `ns(t)` between bracketing frames (best available estimate).
No extrapolation outside the anchor span (returns None) -> the prelude and any
post-log tail are excluded by construction.

This module owns the join; `vo_replay` imports it.
"""
from __future__ import annotations

import json
import os

import numpy as np

# livelog record kinds whose `p` field is a 3-vector estimator position.
# (att.p_est / att.pb are SCALARS -- an x-axis projection, NOT a position -- so
#  `att` is deliberately excluded.)
POS_KINDS = ("kf_upd", "mapfollow", "tick_fix", "mfdr_seed")


def read_rows(livelog_path: str):
    """Parse a livelog.jsonl into a list of dicts (bad lines skipped)."""
    rows = []
    with open(livelog_path) as fh:
        for line in fh:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def build_capture_clock(rows):
    """Monotone loop-wall -> capture-time anchors from `obs*` rows.

    Returns (t_anchors[K], ns_anchors[K]) as float seconds, both strictly
    increasing, or None if fewer than 2 usable anchors. `obs*` rows are the only
    ones carrying BOTH the loop clock `t` and the frame capture clock `ns`.
    Duplicate/tie frames (obs_wronggate + obs_nofix logging the same frame) and
    the occasional stale out-of-order frame are removed by keeping a strictly
    increasing subsequence in BOTH axes.
    """
    pairs = {}
    for d in rows:
        k = d.get("kind", "")
        if isinstance(k, str) and k.startswith("obs"):
            t = d.get("t")
            ns = d.get("ns")
            if isinstance(t, (int, float)) and isinstance(ns, (int, float)):
                pairs[float(t)] = float(ns) * 1e-9      # dedup by exact t
    if len(pairs) < 2:
        return None
    items = sorted(pairs.items())
    # greedy strictly-increasing filter on ns (drops ties / stale frames)
    ts, ns = [], []
    last_ns = -np.inf
    for t, n in items:
        if n > last_ns:
            ts.append(t)
            ns.append(n)
            last_ns = n
    if len(ts) < 2:
        return None
    return np.asarray(ts), np.asarray(ns)


def map_t_to_capture_s(t, clock):
    """Map a loop wall time `t` (s) to capture time (s) via the anchor clock.

    Returns None outside the anchor span (no extrapolation). At an anchor `t`
    (same-iteration obs) this returns that frame's exact capture time.
    """
    t_anchors, ns_anchors = clock
    if t < t_anchors[0] or t > t_anchors[-1]:
        return None
    return float(np.interp(t, t_anchors, ns_anchors))


# livelog record kinds that mark a gate tick / position reset. The approach leg
# BEFORE the first of these is the clean straight forward transit; AFTER it comes
# the gate-1 right turn (breaks a single-rotation alignment) and mapfollow
# position resets (`tick_fix` teleports the estimate). Scale/direction scoring
# against p_est is only valid on the pre-tick straight leg.
TICK_KINDS = ("tick_fix", "gate_tick", "success")


def first_tick_capture_ns(rows, clock):
    """Capture sim_ns of the first gate tick / position reset, or None.

    Marks the end of the clean straight approach leg. `rows`/`clock` as from
    `read_rows` / `build_capture_clock`.
    """
    for d in rows:
        if d.get("kind") in TICK_KINDS and isinstance(d.get("t"), (int, float)):
            cap_s = map_t_to_capture_s(float(d["t"]), clock)
            if cap_s is not None:
                return int(round(cap_s * 1e9))
    return None


def load_estimator_reference(corpus: str, pre_tick_only: bool = False):
    """Estimator position reference on the CAPTURE-TIME axis (sim_ns).

    Reads `livelog.jsonl`, builds the t->capture clock from `obs*` rows, and maps
    each POS_KINDS position row's loop time `t` to a capture sim_ns. Returns
    (ns[M] int64, xyz[M,3] float) with strictly increasing unique ns (last
    position wins per ns), or None if the log is absent / too sparse. ns is in
    the SAME namespace as VO keyframe capture times, so the two are directly
    comparable.

    `pre_tick_only=True` clips the reference to the straight approach leg (before
    the first gate tick / position reset), where a single 2D rotation alignment
    of VO to p_est is valid; past the gate-1 turn the fit is ill-posed.
    """
    lp = os.path.join(corpus, "livelog.jsonl")
    if not os.path.exists(lp):
        return None
    rows = read_rows(lp)
    clock = build_capture_clock(rows)
    if clock is None:
        return None
    hi_ns = first_tick_capture_ns(rows, clock) if pre_tick_only else None
    ns_list, pos_list = [], []
    for d in rows:
        if d.get("kind") not in POS_KINDS:
            continue
        p = d.get("p")
        t = d.get("t")
        if not (isinstance(p, list) and len(p) >= 3 and isinstance(t, (int, float))):
            continue
        cap_s = map_t_to_capture_s(float(t), clock)
        if cap_s is None:
            continue
        cap_ns = int(round(cap_s * 1e9))
        if hi_ns is not None and cap_ns > hi_ns:
            continue
        ns_list.append(cap_ns)
        pos_list.append([float(x) for x in p[:3]])
    if len(ns_list) < 3:
        return None
    ns_arr = np.asarray(ns_list, dtype=np.int64)
    pos_arr = np.asarray(pos_list, dtype=float)
    # collapse to unique capture ns, keeping the LAST position per ns
    keep = {}
    for i, n in enumerate(ns_arr):
        keep[int(n)] = i
    uniq = np.array(sorted(keep), dtype=np.int64)
    idx = np.array([keep[int(n)] for n in uniq])
    return uniq, pos_arr[idx]


def load_kf_pose(corpus: str, pre_tick_only: bool = False):
    """Frame-cadence KF pose reference, keyed DIRECTLY by capture sim_ns.

    Reads ``kf_pose`` rows — logged once per recorded frame by the vq2wp cam
    thread (COR-147 g1 DR-bridge plumbing, 2026-07-23). Each row carries the
    frame's own ``ns``, so unlike ``load_estimator_reference`` no t->ns clock
    fit is needed and the reference covers the NOFIX blind leg where
    ``kf_upd`` rows stop. Returns (ns[M] int64 strictly increasing,
    xyz[M,3] float), or None when the log is absent, the corpus predates the
    row kind (all corpora banked before 2026-07-23), or — with
    ``pre_tick_only=True`` — no tick can be located to clip against.
    """
    lp = os.path.join(corpus, "livelog.jsonl")
    if not os.path.exists(lp):
        return None
    rows = read_rows(lp)
    hi_ns = None
    if pre_tick_only:
        clock = build_capture_clock(rows)
        hi_ns = first_tick_capture_ns(rows, clock) if clock is not None else None
        if hi_ns is None:
            return None
    ns_list, pos_list = [], []
    for d in rows:
        if d.get("kind") != "kf_pose":
            continue
        p, n = d.get("p"), d.get("ns")
        if not (isinstance(p, list) and len(p) >= 3 and isinstance(n, (int, float))):
            continue
        n = int(n)
        if hi_ns is not None and n > hi_ns:
            continue
        ns_list.append(n)
        pos_list.append([float(x) for x in p[:3]])
    if len(ns_list) < 2:
        return None
    ns_arr = np.asarray(ns_list, dtype=np.int64)
    pos_arr = np.asarray(pos_list, dtype=float)
    order = np.argsort(ns_arr, kind="stable")
    ns_arr, pos_arr = ns_arr[order], pos_arr[order]
    keep = np.ones(len(ns_arr), bool)
    keep[1:] = ns_arr[1:] != ns_arr[:-1]
    return ns_arr[keep], pos_arr[keep]
