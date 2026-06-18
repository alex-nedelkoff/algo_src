"""Orchestrate high-tilt rate-loop sysID: mine recordings -> fit -> holdout -> refit
model + envelope + go/no-go report. Pure numpy/scipy + stdlib."""
from __future__ import annotations
import argparse, json, os
import numpy as np
from dataclasses import asdict
from scripts.sysid.ring_loader import load_npz, bin_coverage
from scripts.sysid.ring_fit import fit_axis, holdout_r2, envelope

AXES = ["roll", "pitch", "yaw"]


def run_id(npz_paths, keymap, out_dir, tilt_edges, amp_edges, holdout_frac=0.3) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    runs = [load_npz(p, keymap) for p in npz_paths]
    dt = float(np.median([r.dt for r in runs]))
    n_hold = max(1, int(round(len(runs) * holdout_frac)))
    train, hold = runs[n_hold:], runs[:n_hold]
    if not train:                       # tiny dataset: reuse for both
        train = runs
    params, env, r2s, cov = {}, {}, {}, {}
    for ax in range(3):
        p = fit_axis(train, ax, dt)
        r2 = holdout_r2(p, hold or train, ax, dt, tilt_edges)
        params[AXES[ax]] = asdict(p)
        r2s[AXES[ax]] = [None if np.isnan(v) else round(float(v), 3) for v in r2]
        env[AXES[ax]] = envelope(r2, tilt_edges, thresh=0.9)
        cov[AXES[ax]] = bin_coverage(runs, ax, tilt_edges, amp_edges).tolist()
    max_tilts = [env[a]["max_tilt_deg"] for a in AXES]
    verdict = "go" if min(max_tilts) >= 55.0 else ("partial" if max(max_tilts) >= 45.0 else "no-go")
    with open(os.path.join(out_dir, "vq_model_hightilt.json"), "w") as f:
        f.write(json.dumps({"rate_loop": params, "dt": dt}, indent=2))
    env_out = {a: {**env[a], "r2_by_tilt_bin": r2s[a], "coverage": cov[a]} for a in AXES}
    with open(os.path.join(out_dir, "ring_envelope.json"), "w") as f:
        f.write(json.dumps(env_out, indent=2))
    report = {"verdict": verdict, "max_tilt_deg": dict(zip(AXES, max_tilts)),
              "tilt_edges": list(tilt_edges), "r2": r2s, "n_runs": len(runs)}
    md = [f"# Ring sysID report\n", f"**Verdict: {verdict}**  (per-axis max valid tilt: "
          + ", ".join(f"{a}={env[a]['max_tilt_deg']:.0f}°" for a in AXES) + ")\n",
          "## Per-tilt holdout R²\n", "| axis | " + " | ".join(
              f"{tilt_edges[i]:.0f}-{tilt_edges[i+1]:.0f}°" for i in range(len(tilt_edges) - 1)) + " |",
          "|" + "---|" * (len(tilt_edges)) ]
    for a in AXES:
        md.append(f"| {a} | " + " | ".join("—" if v is None else f"{v:.2f}" for v in r2s[a]) + " |")
    with open(os.path.join(out_dir, "ring_report.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", nargs="+", required=True)
    ap.add_argument("--out", default="sysid")
    a = ap.parse_args()
    from scripts.sysid.ring_loader import DEFAULT_KEYMAP
    rep = run_id(a.npz, DEFAULT_KEYMAP, a.out, tilt_edges=[0, 30, 45, 52, 60, 90],
                 amp_edges=[0, 2, 4, 6, 12])
    print(json.dumps(rep, indent=2))
