# Tilt-budget speed-scheduled trajectory controller (COR-127)

**Date:** 2026-06-09
**Status:** Design approved, pending implementation plan
**Scope:** Full speed-scheduled controller — planner speed law + tracker acquisition limiting + speed-aware cross-track, end-to-end.

## 1. Motivation

This session (exp-log RAMP-WALL / REFIT-01 / REFIT-02 / LATERAL-WALL) characterized the AI-GP
camera-forward "weathervane wall" and found it is **not** a standalone aerodynamic instability. Both
the straight-line speed wall and the lateral-maneuvering wall are the **same constraint: a shared
TILT BUDGET.** The drone's horizontal acceleration authority is `g·tan(tilt_cap)`; forward drag and
lateral correction draw from the same budget.

Measured facts the controller must honor:
- **Straight-line ceiling** scales as `v_max ∝ √(tan(tilt_cap))` — confirmed at 18°→~8, 25°→~9,
  30°→~10.5 m/s. Forward specific force (drag) ≈ `c·v²`, **c ≈ 0.057 /m** (quadratic).
- **Lateral jink capability** shrinks with forward speed: ~5.5 m @v3, ~3 m @v5, ~1.2 m @v7, ~0 @v8.
  At racing speed almost no tilt budget is left for lateral correction.
- The classic yaw weathervane (vs sideslip) breaks `race_cruise`'s heading-hold during lateral
  maneuvers (yaw_err → ±20–28° at divergence) — a secondary contributor that also worsens with speed.

The existing `gate_traj.GateTrajectory` planner already has a curvature-based speed schedule
`v = min(v_cruise, √(a_lat_max/κ))`, but `a_lat_max` is a **fixed** lateral-accel budget
(`g·tan(35°)`) that ignores forward-drag consumption of the budget. That is why `traj_track` diverged
on the translated/curve courses (CRUISE-04→06): on straights it reduces to `race_cruise` and flies,
but any lateral motion over-commits a budget that drag has already partly spent.

## 2. Goal & success criteria

Fly the **translated** and **curve** test courses (the CRUISE-04→06 divergers) on the live VQ sim:
- Reach the end of the course (`s_drone ≥ s_max − ARRIVE`).
- Tilt stays bounded (no tumble; tilt < ~40° sustained, never the 75° abort).
- Competitive peak speed on the straight sections (~6–7 m/s), automatically slowing into lateral/curved
  sections rather than tumbling.
- Direct contrast: the old fixed-budget controller tumbled on these courses; the new one completes them.

## 3. The unifying speed law

Total horizontal accel budget: `A_budget = g·tan(φ_budget)`. Reserve a `margin` fraction for
cross-track correction + disturbance. At speed `v` on curvature `κ`, demand = forward drag `c·v²` +
centripetal `v²·κ`. Require `c·v² + v²·κ ≤ A_budget·margin`, solve for v:

```
v_max(κ) = min( v_cruise, sqrt( g·tan(φ_budget)·margin / (c_drag + κ) ) )
```

- Straight (κ→0): `v_max → √(g·tan(φ_budget)·margin / c_drag)` = the drag-saturation ceiling with margin
  (~6.9 m/s at φ_budget=25°, margin=0.6, c=0.057). Reduces to the REFIT-02 result.
- Curve: the `κ` term lowers v. Unifies both walls in one expression.

This **replaces** the fixed `a_lat_max/κ` law. `c_drag` makes the straight-line ceiling finite (the old
law returned `v_cruise` on straights, ignoring drag).

## 4. Components

### C1 — Planner: `gate_traj.GateTrajectory` (pure, offline-testable)
- Constructor gains new params: `tilt_budget_deg: float = 25.0`, `c_drag: float = 0.057`,
  `margin: float = 0.6`. Keep `v_cruise` as the absolute cap. `phi_max_deg` retained only if needed
  for back-compat; the speed law no longer uses `a_lat_max`.
- `speed_at(κ_abs)` implements the §3 law.
- `sample(s)` unchanged in shape; `v`/`yaw_rate`/`a_lat` now derive from the new `speed_at`.
- No sim dependencies; the `__main__` demo prints the new schedule (straight vs translated vs curve).

### C2 — Tracker: `traj_track.py`
- `TILT_MAX_ACC = g·tan(φ_budget)` — same budget as the planner (was `tan(35°)`). Single source of
  truth: read `φ_budget` from one constant / arg and pass to both planner and tracker.
- **Budget-aware cross-track cap (acquisition limiting):** residual lateral budget at current speed
  `a_ct_max(v) = max(0, g·tan(φ_budget)·margin − c_drag·v²)`. Clamp the cross-track command
  `a_ct = clip(−KP_CT·p_ct − KD_CT·v_ct, −a_ct_max, +a_ct_max)`. This bounds the transient line
  acquisition to whatever tilt is left after holding speed — the CRUISE-04 divergence trigger — and is
  the principled form of "speed-aware cross-track gain" (authority shrinks as speed eats the budget).
- Everything else unchanged: drone-locked `nearest_s + LEAD` reference, nose-follows-tangent yaw,
  ZVD yaw prefilter, race_cruise attitude gains, tilt abort, along-track governor (raise `AL_MAX` so the
  planner's scheduled `ref["v"]` governs, not a 0.6 cap).
- **Durability (hard rule):** add `aigp.recorder.Recorder` + keep the `flog` Rerun dashboard stream so
  every live run is recorded and visualized; never fly blind.

### C3 — Validation
- **Offline (`--dry`):** print the speed schedule for straight / translated / curve; assert straight
  approaches ~7 m/s and curved sections slow (lower v where κ is high).
- **Live VQ sim:** run `translated` then `curve`; success = reach end, tilt bounded, peak speed
  competitive. Kill stray python on udp 14550 first; `python -u > log`, poll the log; watch the Rerun
  dashboard (100.101.13.126:9876).

## 5. Parameters & defaults
| param | default | source / meaning |
|---|---|---|
| `tilt_budget_deg` (φ_budget) | 25° | mid of proven-18 / aggressive-30; tunable live |
| `c_drag` | 0.057 /m | REFIT-02 quadratic drag |
| `margin` | 0.6 | reserve 40% of budget for cross-track + disturbance |
| `v_cruise` | as today | absolute speed cap |

## 6. Acceptance test
1. `python gate_traj.py` (or `traj_track.py curve --dry`) shows the new schedule, straight ~7, curve slows.
2. `traj_track.py translated` on live VQ → REACHED END, max_tilt < ~45, no tumble.
3. `traj_track.py curve` on live VQ → REACHED END, max_tilt < ~45, peak_spd competitive.
4. Recorded run + dashboard present for each.

## 7. Risks & mitigations
- **Yaw weathervane still breaks heading-hold on sharp lateral** even within the accel budget → if
  divergence persists, add a sideslip-aware yaw term or lower `margin`; do NOT redesign the loop first.
- **`c_drag`/budget mis-tuned** → the offline `--dry` + a single live run calibrate; params are exposed.
- **Frame traps** → reuse `race_cruise`/`traj_track`'s existing frame handling verbatim; judge stability
  by TILT, never a hand-rolled β/heading metric (this session's bug).

## 8. Out of scope (YAGNI)
- Centripetal/curvature feedforward (the log's MIMO redesign) — only if budget caps prove insufficient.
- Perception/vision-in-the-loop; gate detection. Uses the planner's known gate poses.
- RL. This is the analytic controller; RL can consume the same tilt-budget speed schedule later.
