#!/usr/bin/env python3
"""Export the RL training/deploy racing course as a map for validity inspection.

Replicates the course generator in `scripts/rl/rl_finetune.py:make_env` (CURVE
branch, lines ~158-168) and `sysid/vq_deploy4_canonical.py` (lines ~88-94) VERBATIM
so we can SEE what geometry the pod policy is trained + deployed against, and check
it against the REAL VQ1 sim course (TRACK_INFO).

Outputs (to --out dir, default /tmp/rl_course):
  rl_course_map.png   top-down XY + along-track altitude profile + stats panel
  rl_course.json      gate coords + per-leg spacing/turn/dz stats (machine-readable)

Real-VQ1 overlay: pass --real path/to/real_gates.json (list of [x,y,z] NED or ENU,
6 gates) to overlay the actual sim course and print the capability gap. Without it
the script prints the known TRACK_INFO facts (6 gates, 2.72x2.72 m, descends ~26 m
over ~136 m) as the reference to compare by eye.

Usage:
  python3 scripts/rl/export_course_map.py --mode curve --seeds 0 1 2 3 --vdes 16
  python3 scripts/rl/export_course_map.py --mode deploy --turn 0.2 --space 14
"""
import argparse, json, os
import numpy as np


def _yaw_quat(th):                       # rl_finetune.py:149
    return np.array([np.cos(th / 2), 0.0, 0.0, np.sin(th / 2)])


def gen_curve_course(seed, ng=6):
    """rl_finetune.py:154-168 CURVE branch, verbatim. Returns (NG,3) gate xyz, (NG,) yaw."""
    rng = np.random.default_rng(seed)
    sp = rng.uniform(9, 18)                                  # spacing (m)
    amp = rng.uniform(1.0, 3.0)                              # (unused in CURVE; for parity)
    ph = rng.uniform(0, 6.28)
    hd = 0.0
    pos = np.array([8.0, 0.0, 1.0])
    turn = rng.uniform(0.12, 0.28) * rng.choice([-1, 1])     # rad/gate, CONSTANT sign
    gates, yaws = [], []
    for i in range(ng):
        hd += turn
        pos = pos + sp * np.array([np.cos(hd), np.sin(hd), 0.0])
        z = 1.0 + 0.6 * np.sin(i * 0.7 + ph)                # FLAT +/-0.6 m wobble
        gates.append([pos[0], pos[1], z])
        yaws.append(hd)
    return np.array(gates), np.array(yaws), dict(sp=sp, turn=turn, ph=ph)


def gen_deploy_course(turn=0.2, space=14.0, ng=6):
    """vq_deploy4_canonical.py:88-94 verbatim. Fixed turn + fixed spacing."""
    hd = 0.0
    p = np.array([8.0, 0.0, 1.0])
    gates, yaws = [], []
    for i in range(ng):
        hd += turn
        p = p + space * np.array([np.cos(hd), np.sin(hd), 0.0])
        gates.append(p.copy())
        yaws.append(hd)
    return np.array(gates), np.array(yaws), dict(sp=space, turn=turn)


def leg_stats(spawn, gates):
    pts = np.vstack([spawn, gates])
    legs = np.diff(pts, axis=0)
    seg = np.linalg.norm(legs[:, :2], axis=1)               # horizontal spacing
    cum = np.concatenate([[0], np.cumsum(np.linalg.norm(legs, axis=1))])
    hd = np.arctan2(legs[:, 1], legs[:, 0])
    dturn = np.degrees(np.diff(hd))
    dz = legs[:, 2]
    return dict(seg=seg, cum=cum, dturn=dturn, dz=dz,
                total_len=float(cum[-1]),
                z_span=float(gates[:, 2].max() - gates[:, 2].min()),
                z_drop=float(gates[:, 2].min() - spawn[2]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["curve", "deploy"], default="curve")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--turn", type=float, default=0.2)
    ap.add_argument("--space", type=float, default=14.0)
    ap.add_argument("--vdes", type=float, default=16.0)
    ap.add_argument("--ng", type=int, default=6)
    ap.add_argument("--ep_s", type=float, default=15.0)     # EP_STEPS 1080 @ dt 1/72
    ap.add_argument("--real", type=str, default=None)
    ap.add_argument("--out", type=str, default="/tmp/rl_course")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    spawn = np.array([0.0, 0.0, 1.0])

    courses = []
    if args.mode == "curve":
        for s in args.seeds:
            g, y, meta = gen_curve_course(s, args.ng)
            courses.append((f"seed{s}", g, y, meta))
    else:
        g, y, meta = gen_deploy_course(args.turn, args.space, args.ng)
        courses.append((f"deploy t{args.turn} s{args.space}", g, y, meta))

    reach = args.vdes * args.ep_s

    real = None
    if args.real and os.path.exists(args.real):
        rj = json.load(open(args.real))
        real = np.array(rj["gates"] if isinstance(rj, dict) else rj, float)

    # ---- stats + json ----
    export = {"mode": args.mode, "vdes": args.vdes, "reach_m": reach,
              "ep_s": args.ep_s, "courses": []}
    print(f"\n=== RL {args.mode} course ===   vdes={args.vdes}  reach@{args.ep_s}s = {reach:.0f} m")
    for name, g, y, meta in courses:
        st = leg_stats(spawn, g)
        export["courses"].append({
            "name": name, "gates": g.tolist(), "yaws_deg": np.degrees(y).tolist(),
            "spacing_m": st["seg"].tolist(), "turn_deg": st["dturn"].tolist(),
            "dz_m": st["dz"].tolist(), "total_len_m": st["total_len"],
            "z_span_m": st["z_span"], "z_drop_m": st["z_drop"], **{k: float(v) for k, v in meta.items()}})
        print(f"  {name:16s} len={st['total_len']:5.1f}m  spacing {st['seg'].min():.1f}-{st['seg'].max():.1f}m  "
              f"turn/gate {np.degrees(meta['turn']):+.1f}deg  z-span {st['z_span']:.2f}m  z-drop {st['z_drop']:+.2f}m")
    json.dump(export, open(f"{args.out}/rl_course.json", "w"), indent=2)

    # ---- known real-VQ1 reference (TRACK_INFO facts from experiment log) ----
    REAL_REF = dict(n_gates=6, gate_size_m=2.72, clear_half_m=0.95,
                    total_len_m=136.0, descent_m=26.0, spacing_m=136.0 / 5)
    print("\n=== REAL VQ1 sim course (TRACK_INFO) ===")
    print(f"  6 gates, {REAL_REF['gate_size_m']}x{REAL_REF['gate_size_m']} m (clear half ~{REAL_REF['clear_half_m']} m), "
          f"len ~{REAL_REF['total_len_m']} m, DESCENDS ~{REAL_REF['descent_m']} m, ~{REAL_REF['spacing_m']:.0f} m/gate")
    if real is not None:
        st = leg_stats(real[0], real[1:]) if len(real) == args.ng + 1 else leg_stats(spawn, real)
        print(f"  [loaded] len={st['total_len']:.1f}m spacing {st['seg'].min():.1f}-{st['seg'].max():.1f}m "
              f"z-span {st['z_span']:.2f}m")

    # ---- capability gap ----
    print("\n=== CAPABILITY GAP (RL env vs real sim) ===")
    sp_all = np.concatenate([leg_stats(spawn, g)["seg"][1:] for _, g, _, _ in courses])  # drop spawn leg (+8 offset)
    z_all = [leg_stats(spawn, g)["z_span"] for _, g, _, _ in courses]
    real_glide = None
    if real is not None:
        rs = leg_stats(real[0], real[1:]) if len(real) == args.ng + 1 else leg_stats(spawn, real)
        horiz = np.linalg.norm(np.diff(real, axis=0)[:, :2], axis=1)
        real_glide = np.degrees(np.arctan2(rs["dz"], horiz))      # per-leg glideslope (deg, +=down here z_up so neg=down)
        real_turn = rs["dturn"]
        real_sp = rs["seg"]
        print(f"  spacing : RL gate-to-gate {sp_all.min():.1f}-{sp_all.max():.1f} m  |  real {real_sp.min():.1f}-{real_sp.max():.1f} m  -> real OOD-LONG (esp {real_sp.max():.0f}m descent leg)")
        print(f"  descent : RL z-span {min(z_all):.1f}-{max(z_all):.1f} m (FLAT)  |  real DROPS {abs(rs['z_span']):.1f} m, glideslope per-leg {np.round(-real_glide,1).tolist()} deg  -> RL NEVER trains descent")
        print(f"  turns   : RL single-sign arc {np.degrees(0.12):.0f}-{np.degrees(0.28):.0f} deg/gate  |  real |turn| {np.abs(real_turn).max():.1f} deg max -> real is NEAR-STRAIGHT, RL turns HARDER")
        print(f"  gate ap : RL pass-radius ~1.0 m  |  real clear half ~{REAL_REF['clear_half_m']} m (2.72m gate, frame eats ~0.4)  -> close")
    else:
        print(f"  (no --real loaded; using TRACK_INFO reference facts)")
        print(f"  spacing : RL {sp_all.min():.1f}-{sp_all.max():.1f} m  |  real ~{REAL_REF['spacing_m']:.0f} m")
        print(f"  descent : RL FLAT  |  real DROPS {REAL_REF['descent_m']} m")

    # ---- plot ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(20, 6.5))
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(courses)))

    # top-down XY
    for (name, g, y, meta), c in zip(courses, cmap):
        pts = np.vstack([spawn, g])
        ax[0].plot(pts[:, 0], pts[:, 1], "-o", color=c, ms=5, label=name)
        for i, gg in enumerate(g):
            ax[0].annotate(str(i), (gg[0], gg[1]), fontsize=7, color=c)
    ax[0].scatter(*spawn[:2], c="k", marker="*", s=160, zorder=5, label="spawn")
    if real is not None:
        ax[0].plot(real[:, 0], real[:, 1], "-s", color="red", ms=6, lw=2, label="REAL VQ1")
    ax[0].add_artist(plt.Circle((0, 0), reach, fill=False, ls="--", color="gray", alpha=0.6))
    ax[0].set_title(f"Top-down XY (m)  | dashed = v{args.vdes:.0f}x{args.ep_s:.0f}s reach {reach:.0f}m")
    ax[0].set_xlabel("x (m)"); ax[0].set_ylabel("y (m)"); ax[0].axis("equal"); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=7)

    # altitude profile (along-track vs z)
    for (name, g, y, meta), c in zip(courses, cmap):
        st = leg_stats(spawn, g)
        ax[1].plot(st["cum"], np.concatenate([[spawn[2]], g[:, 2]]), "-o", color=c, ms=5, label=name)
    ax[1].axhline(spawn[2], color="k", ls=":", alpha=0.5)
    # real descent reference line
    ax[1].plot([0, REAL_REF["total_len_m"]], [spawn[2], spawn[2] - REAL_REF["descent_m"]],
               "r--", lw=2, label=f"REAL descent ~{REAL_REF['descent_m']}m")
    ax[1].set_title("Altitude vs along-track distance (m)")
    ax[1].set_xlabel("along-track (m)"); ax[1].set_ylabel("z up (m)"); ax[1].grid(alpha=0.3); ax[1].legend(fontsize=7)

    # stats panel
    ax[2].axis("off")
    lines = [f"RL {args.mode} COURSE", "", f"gates: {args.ng}", f"vdes: {args.vdes} m/s",
             f"reach @ {args.ep_s}s: {reach:.0f} m", "",
             f"spacing: {sp_all.min():.1f}-{sp_all.max():.1f} m",
             f"turn/gate: {np.degrees(0.12):.0f}-{np.degrees(0.28):.0f} deg (1 sign)",
             f"z-span: {min(z_all):.1f}-{max(z_all):.1f} m (FLAT)",
             f"pass radius: ~1.0 m", "",
             "--- REAL VQ1 (live TRACK_INFO) ---", f"gates: 6   size: 2.72x2.72 m",
             f"len: ~164 m   spacing: 23-37 m/gate",
             f"DESCENDS 26 m (glideslope to 17deg)",
             f"NEAR-STRAIGHT (lateral +-5 m)",
             f"clear half-aperture: ~0.95 m", "",
             "--- GAPS ---",
             f"* real spacing 23-37m > RL max ~18m",
             f"* RL FLAT; real dives 26m (untrained)",
             f"* real STRAIGHT; RL arcs hard (backwards)"]
    ax[2].text(0.02, 0.98, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=10,
               transform=ax[2].transAxes)
    plt.tight_layout()
    out_png = f"{args.out}/rl_course_map.png"
    plt.savefig(out_png, dpi=110)
    print(f"\nwrote {out_png}\nwrote {args.out}/rl_course.json")


if __name__ == "__main__":
    main()
