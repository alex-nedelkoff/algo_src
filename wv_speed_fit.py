"""wv_speed_fit.py -- refit the camera-forward weathervane vs FORWARD AIRSPEED across the new v0->8
cam-forward data (collect_vq_ramp) + the v0-4 baseline (collect_vq). Reuses fit_model.prep() so ALL
frame conventions (qfix / wfix / body-velocity / t_us-dedup / kinematic accel) are the SOLVED ones,
not re-derived.

Question: the prior directional roll-weathervane (~ -0.019 per m/s, fit on v0-4) -- does it stay
LINEAR to v~8, or STEEPEN near the wall (~8 m/s)? If |g| grows super-linearly at v6-8, that is the
wall mechanism (a model fit on v0-4 under-predicts the high-speed instability -> control/RL trained
on it tumbles at speed). Per-speed-bin rate-residual regression, SAME structure as
fit_model.fit_weathervane: dW_axis = a*wcmd_axis - b*omega_axis + g*v_body_component.

Usage: python wv_speed_fit.py
"""
import numpy as np
from aigp.recorder import load_index
from fit_model import prep

AX = {"roll": 0, "pitch": 1, "yaw": 2}


def fit_wv_axis(dW_axis, wcmd_axis, omega_axis, v_body):
    """Weathervane coeff on one axis: dW = a*wcmd - b*omega + wv*v_body (the fit_model.fit_weathervane
    structure). Returns wv (the moment-per-sideslip) + R^2. Pure -> unit-testable."""
    M = np.vstack([wcmd_axis, -omega_axis, v_body]).T
    sol, *_ = np.linalg.lstsq(M, dW_axis, rcond=None)
    r2 = 1.0 - np.var(dW_axis - M @ sol) / np.var(dW_axis)
    return {"a": float(sol[0]), "b": float(sol[1]), "wv": float(sol[2]), "r2": float(r2), "n": int(len(dW_axis))}


def refit_dynamic(P):
    """Roll & yaw weathervane vs v_body_y, binned by forward speed |v_body_x|, across the crab data."""
    vb = np.concatenate([p["vb"] for p in P]); W = np.concatenate([p["W"] for p in P])
    C = np.concatenate([p["C"] for p in P]); dW = np.concatenate([p["dW"] for p in P])
    spd = np.abs(vb[:, 0])
    for axname, ax in (("roll", 0), ("yaw", 2)):
        print(f"\n== {axname} weathervane vs v_body_y, binned by |v_body_x| ==", flush=True)
        for lo, hi in [(0, 2), (2, 3.5), (3.5, 5), (5, 7), (7, 10)]:
            mm = (spd >= lo) & (spd < hi) & (np.abs(vb[:, 1]) > 0.3)   # require real sideslip
            if int(mm.sum()) < 200:
                print(f"   v[{lo},{hi}) n={int(mm.sum())} (skip)", flush=True); continue
            o = fit_wv_axis(dW[mm, ax], C[mm, ax], W[mm, ax], vb[mm, 1])
            print(f"   v[{lo},{hi}) n={o['n']:5d}  wv={o['wv']:+.4f}  R2={o['r2']:.2f}", flush=True)


runs = [r for r in load_index() if r["script"] in ("collect_vq", "collect_vq_ramp", "collect_vq_crab", "collect_vq_lateral")]
P = [p for p in (prep(r) for r in runs) if p]
vb = np.concatenate([p["vb"] for p in P])
W = np.concatenate([p["W"] for p in P])
C = np.concatenate([p["C"] for p in P])
dW = np.concatenate([p["dW"] for p in P])
spd = np.abs(vb[:, 0])                          # forward airspeed = |v_body_x| (cam-forward => v_body_x<0)
print(f"runs={len(P)} pooled_rows={len(vb)}  |v_body_x| {spd.min():.1f}..{spd.max():.1f} m/s "
      f"(n@v>4 = {int((spd>4).sum())}, n@v>6 = {int((spd>6).sum())})", flush=True)

BINS = [(0, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 10)]


def fit_axis(axname, vcol):
    ax = AX[axname]
    print(f"\n== {axname.upper()} weathervane: dW_{axname} = a*wcmd - b*omega + g*v_body_{'xyz'[vcol]} ==", flush=True)
    print(f"   {'v-bin':>9} {'n':>6} {'g(wv)':>9} {'R2':>5}", flush=True)
    gs = []
    for lo, hi in BINS:
        m = (spd >= lo) & (spd < hi)
        if int(m.sum()) < 200:
            print(f"   [{lo},{hi}) {int(m.sum()):>6}   (skip <200)", flush=True); continue
        M = np.vstack([C[m, ax], -W[m, ax], vb[m, vcol]]).T
        sol, *_ = np.linalg.lstsq(M, dW[m, ax], rcond=None)
        g = float(sol[2]); r2 = 1 - np.var(dW[m, ax] - M @ sol) / np.var(dW[m, ax])
        gs.append((0.5 * (lo + hi), g))
        print(f"   [{lo},{hi}) {int(m.sum()):>6} {g:>+9.4f} {r2:>5.2f}", flush=True)
    if len(gs) >= 3:                              # is g(v) linear, or steepening? fit g vs v
        vmid = np.array([x[0] for x in gs]); gv = np.array([x[1] for x in gs])
        A = np.vstack([np.ones_like(vmid), vmid]).T
        (off, slope), *_ = np.linalg.lstsq(A, gv, rcond=None)
        print(f"   -> g(v) linear fit: g = {off:+.4f} {slope:+.4f}*v   (low-v g={gv[0]:+.4f}, "
              f"high-v g={gv[-1]:+.4f}, ratio {gv[-1]/gv[0] if gv[0] else float('nan'):.1f}x)", flush=True)


fit_axis("roll", 0)                              # cam-forward destabilizer (roll vs forward speed)
fit_axis("yaw", 1)                               # classic weathervane (yaw vs sideslip), for comparison
print("\nNOTE: judge the WALL by whether |g_roll| grows with v (esp. v6-8) vs the v0-4 baseline.", flush=True)

refit_dynamic(P)
