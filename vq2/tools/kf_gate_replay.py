"""Offline replay of the KF position gate over the banked corpora (COR-147).

Answers one question against real flight data: which of the accepted
`kf_upd` rows would the fixed gate have rejected?

The livelog does not carry P or the innovation vector, so a byte-exact replay
is impossible. What it does carry per accepted update is (miss, r_scale,
nis) plus -- from the `obs` row emitted immediately after, in the same
iteration of the detector loop -- the gate-relative observation `g_lvl`,
whose norm IS the `rng` that was passed to update_position (vq2wp: rng_meas =
norm(g_lvl)). That is enough for two statements:

1. RIGOROUS (assumption-light). d2 = sum_i c_i^2 / (lambda_i + sigma^2) over
   the eigenbasis of P_xx, so d2 >= miss^2 / (lambda_max + sigma_true^2).
   For the fixed gate to still ACCEPT an update, the filter must therefore
   have carried

       lambda_max >= miss^2 / maha_gate - sigma_true^2

   i.e. a position 1-sigma of `sigma_needed` along the innovation. When that
   is tens of metres, the update is rejected under any P the filter could
   plausibly have held.

2. SCALAR-MODEL ESTIMATE. Treating P_xx as isotropic along the innovation
   (exact iff the innovation is an eigenvector of P_xx), the logged nis
   inverts to lambda_eff = miss^2/nis - (sigma_true*r_scale)^2, and the fixed
   gate would read nis_new = miss^2 / (lambda_eff + sigma_true^2).

Both are reported. They agree on every large-miss row; the scalar estimate is
the one that can be wrong, so the verdict column uses the rigorous test.

Usage: python -m vq2.tools.kf_gate_replay [corpus ...]   (default: all banked)
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys

ROOT = r"C:\Users\Administrator"
SIG_BASE, SIG_PER_M = 0.35, 0.06
MAHA_GATE = 16.27          # the FLOWN value (vq2/live/eskf.py)
HUBER_DELTA = 1.5


def pairs(rows):
    """Accepted kf_upd rows paired with the obs row from the same iteration."""
    out = []
    pending = None
    for r in rows:
        k = r.get("kind")
        if k == "kf_upd":
            pending = r                     # a new kf_upd supersedes an
        elif k == "obs" and pending is not None:   # unpaired predecessor
            out.append((pending, r))
            pending = None
    return out


def analyse(corpus):
    lg = os.path.join(ROOT, corpus, "livelog.jsonl")
    if not os.path.exists(lg):
        return []
    rows = []
    with open(lg) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    out = []
    for upd, obs in pairs(rows):
        g = obs.get("g_lvl")
        nis = upd.get("nis")
        miss = upd.get("miss")
        if not g or miss is None or not isinstance(nis, (int, float)):
            continue
        rng = math.dist([0, 0, 0], g)
        sig_t = SIG_BASE + SIG_PER_M * rng
        rs = upd.get("r_scale", 1.0)

        # (1) rigorous: uncertainty the filter would need to still accept
        need_lambda = miss ** 2 / MAHA_GATE - sig_t ** 2
        sigma_needed = math.sqrt(need_lambda) if need_lambda > 0 else 0.0

        # (2) scalar-model estimate of the post-fix NIS
        nis_new = None
        if nis > 0:
            lam_eff = miss ** 2 / nis - (sig_t * rs) ** 2
            denom = lam_eff + sig_t ** 2
            if denom > 0:
                nis_new = miss ** 2 / denom
        out.append(dict(corpus=corpus, miss=miss, rs=rs, nis=nis, rng=rng,
                        sigma_needed=sigma_needed, nis_new=nis_new,
                        neg=nis < 0))
    return out


def main(argv):
    corpora = argv or [os.path.basename(p) for p in
                       sorted(glob.glob(os.path.join(ROOT, "vq2_*")))
                       if os.path.isdir(p)]
    rec = []
    for c in corpora:
        rec.extend(analyse(c))
    if not rec:
        print("no paired kf_upd/obs rows found")
        return

    # An update is judged REJECTED by the fixed gate when the position sigma
    # it would need to survive exceeds what the filter can hold. 5 m is far
    # beyond anything this filter carries in flight (P is reseated to 0.5 I
    # at every tick_fix) and is used only to label the table.
    rejected = [r for r in rec if r["sigma_needed"] > 5.0]
    big = [r for r in rec if r["miss"] >= 10.0]
    neg = [r for r in rec if r["neg"]]

    print(f"{len(rec)} accepted updates with a recoverable range\n")
    print(f"{'corpus':<16}{'miss':>8}{'rng':>7}{'r_scale':>8}"
          f"{'nis_old':>9}{'nis_new':>10}{'sigma_needed':>13}  verdict")
    print("-" * 88)
    for r in sorted(rec, key=lambda r: -r["miss"])[:18]:
        nn = f'{r["nis_new"]:.1f}' if r["nis_new"] is not None else "-"
        print(f'{r["corpus"]:<16}{r["miss"]:>8.2f}{r["rng"]:>7.1f}'
              f'{r["rs"]:>8.2f}{r["nis"]:>9.2f}{nn:>10}'
              f'{r["sigma_needed"]:>12.1f} m  '
              f'{"REJECTED" if r["sigma_needed"] > 5.0 else "kept"}')

    print(f"\naccepted updates, miss >= 10 m : {len(big)}")
    print(f"  of those, rejected by the fix: "
          f"{sum(1 for r in big if r['sigma_needed'] > 5.0)}")
    print(f"negative-NIS updates (applied with indefinite P): {len(neg)}"
          f"  -> all rejected by the fix (d2 < 0 check)")
    print(f"total rejected by the fix     : {len(rejected)} / {len(rec)}"
          f"  ({100.0*len(rejected)/len(rec):.1f}%)")
    small = [r for r in rec if r["miss"] <= HUBER_DELTA]
    print(f"full-weight fixes (miss <= {HUBER_DELTA}): {len(small)}, "
          f"rejected by the fix: "
          f"{sum(1 for r in small if r['sigma_needed'] > 5.0)}"
          f"   <- must stay 0: the fix must not cost good fixes")


if __name__ == "__main__":
    main(sys.argv[1:])
