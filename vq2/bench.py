"""Acceptance-policy replay bench: run recorded corpora through candidate
obs-acceptance policies and compare filter health side by side. No policy
flies before it wins here.

Usage:
    python3 -m vq2.bench <corpus_dir> [policy ...]
"""
from __future__ import annotations

import sys

import numpy as np

from .fusion import POLICIES, FusionConfig, run_fusion


# bench-level combos: policy name -> FusionConfig kwargs
COMBOS = {
    "radius": dict(accept_policy="radius"),
    "huber": dict(accept_policy="huber"),
    "huber_area": dict(accept_policy="huber", range_gate=True),
    "radius_area": dict(accept_policy="radius", range_gate=True),
}


def run_bench(root: str, policies=("radius", "huber", "huber_area")) -> dict:
    out = {}
    for name in policies:
        if name not in COMBOS:
            raise ValueError(f"unknown policy {name!r} (have {list(COMBOS)})")
        res = run_fusion(root, FusionConfig(use_flow=False, use_vision_pos=True,
                                            **COMBOS[name]))
        acc = [e for e in res.obs_events if e["stage"] == "accepted"]
        rej = [e for e in res.obs_events if e["stage"] != "accepted"]
        errs = [a["err_xy"] for a in res.anchors]
        # close-range anchors = the tick-deciding regime; far anchors on
        # runaway-tail segments measure junk-chasing, not policy quality
        close = [a["err_xy"] for a in res.anchors if a["range"] < 12.0]
        nis = [e["nis"] for e in acc
               if "nis" in e and np.isfinite(e["nis"])]
        p = np.asarray(res.p)
        dist = float(np.sum(np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1))) \
            if len(p) > 1 else 0.0
        out[name] = {
            "accepted": len(acc),
            "rejected": len(rej),
            "reject_stages": {s: sum(1 for e in rej if e["stage"] == s)
                              for s in {e["stage"] for e in rej}},
            "anchor_p50": round(float(np.percentile(errs, 50)), 3) if errs else None,
            "anchor_p90": round(float(np.percentile(errs, 90)), 3) if errs else None,
            "close_p50": round(float(np.percentile(close, 50)), 3) if close else None,
            "close_p90": round(float(np.percentile(close, 90)), 3) if close else None,
            "close_n": len(close),
            "nis_p50": round(float(np.percentile(nis, 50)), 2) if nis else None,
            "nis_frac_gt_gate": round(float(np.mean([n > 11.34 for n in nis])), 3)
            if nis else None,
            "dist_traveled_m": round(dist, 1),
        }
    return out


def main(argv: list) -> int:
    if not argv:
        print(__doc__)
        return 2
    root = argv[0]
    policies = tuple(argv[1:]) or ("radius", "huber", "huber_area")
    out = run_bench(root, policies)
    cols = ["accepted", "rejected", "close_n", "close_p50", "close_p90",
            "anchor_p90", "nis_p50", "dist_traveled_m"]
    print(f"{'policy':10s} " + " ".join(f"{c:>16s}" for c in cols))
    for name, row in out.items():
        print(f"{name:10s} " + " ".join(f"{str(row[c]):>16s}" for c in cols))
        if row["rejected"]:
            print(f"{'':10s}   reject stages: {row['reject_stages']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
