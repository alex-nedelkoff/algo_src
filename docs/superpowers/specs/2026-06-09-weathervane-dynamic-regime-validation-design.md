# Weathervane validation in the dynamic regime (COR-127) — design

**Date:** 2026-06-09
**Status:** Design approved, pending implementation plan
**Scope:** Collect controlled high-sideslip data → refit the weathervane in the dynamic (forward-racing) regime → trajectory-validate `VQMatchedDynamics` against live VQ. Deliverable = a matched RL sim validated where it currently is not.

## 1. Motivation — the one root cause behind every stalled approach

All three controller strategies for dynamic racing have been tried and all stalled on the **same** gap (vault 06-07/08 + COR-127 comments):
- **FF/INDI residual RL:** ×4 attempts, all below the FF baseline (gradient death from `clip(FF+residual)`). 3-fail rule hit.
- **Pure RL / DAgger:** DAgger reaches 5.6/6 in the matched sim but PPO erodes it and `ft_dagger_lat.zip` still tumbles on live VQ as v→6.
- **Analytic FF / INDI controller:** built, rejects the weathervane *in-sim* and on camera-*backward*; never confirmed live-VQ forward-at-speed.

Root cause, stated verbatim:
> COR-127 (06-08): *"Remaining gap is the high-sideslip forward-racing regime (needs divergent-regime VQ data), not interface or latency."*
> Vault 06-07: *"closed-loop slalom keeps sideslip small → cannot discriminate the weathervane."*

**The matched sim's weathervane is fit on v0–4 / low-sideslip / closed-loop data → wrong (too weak) in the dynamic regime.** Every approach was validated in that sim, then failed on live VQ. The model couldn't be fixed because **controlled high-sideslip data was uncollectable** — closed-loop control suppresses the sideslip signal; open-loop diverges.

This session changed that: cam-forward is now flyable to ~8 m/s (drag/tilt-saturation understood, REFIT-02) and **fixed-heading `traj_track` flies lateral maneuvers (translated) stably to 82%** (TRACK-01). That stabilizer is the first tool that can collect controlled high-sideslip data on live VQ — the missing "divergent-regime data."

The interface is already consistent: `VQMatchedDynamics` (`sim/dynamics/vq_matched.py`) uses the identical ACRO rate interface `[thr, wx, wy, wz]` and already models both weathervanes (`om_roll += (roll_wv0 + roll_wv1·v_body_x)·v_body_y·dt`, `om_yaw += yaw_wv·v_body_y·dt`). Only the **coefficients in the dynamic regime** are unvalidated.

## 2. Goal & success criteria

Make `VQMatchedDynamics` faithful in the high-sideslip forward-racing regime:
- Collect controlled high-sideslip data on live VQ across forward speeds (v≈3/5/7) up to the divergence wall.
- Refit the weathervane (roll + yaw moment vs sideslip `v_body_y`, modulated by forward speed `v_body_x`), allowing nonlinearity at high sideslip.
- **Trajectory-validate:** replaying held-out live runs through the refit `VQMatchedDynamics` reproduces the live sideslip dynamics + the lateral-wall onset (not just 1-step accel) within tolerance.

Done = a matched sim whose high-sideslip behavior matches live VQ → ready to hand to the existing RL/DAgger pipeline (separate downstream spec).

## 3. The identification maneuver (the crux)

The trap prior sessions hit: closed-loop control suppresses sideslip. The fix decouples *create sideslip* from *stay upright*: a **steady crab-sweep** — hold forward speed + fixed heading, ramp a lateral-velocity crab setpoint so the drone slides sideways relative to its nose → sustained, controlled `v_body_y`. The attitude loop keeps it level; the crab supplies clean steady-state sideslip the oscillatory slalom could not. The weathervane moment is the systematic roll/yaw angular-accel residual at that sideslip. Run at several forward speeds to map the moment vs `(v_body_y, v_body_x)`.

Fallbacks: replay this session's existing lateral runs as a free cross-check (Approach C); if the plain stabilizer cannot hold enough sideslip at high speed (the crab wall at v7 is ~1.2 m, LATERAL-WALL), add the INDI/FF inner loop to push deeper (Approach B), logging + inverting its correction to recover the full moment.

## 4. Components

### C1 — Crab-sweep maneuver: `collect_vq_crab.py`
- Reuse the race_cruise control law VERBATIM + `aigp.recorder.Recorder` + `flog` dashboard + collision/tilt gate (same pattern as `collect_vq`/`collect_vq_lateral` this session).
- Hold forward speed via `--almax` (sets the drag-balance speed, per this session's finding); fixed heading (yaw0, the proven crab — NOT nose-follows-tangent).
- Ramp a **lateral-velocity crab setpoint** `v_ct_ref(t)` from 0 upward over the run (steady crab, not slalom) → sustained sideslip. Cross-track loop tracks `v_ct_ref` (rate setpoint) rather than holding the line.
- Run at `--almax` giving v≈3, 5, 7. Records `vel`, `quat`, `omega`, `cmd` (sideslip derived offline via `prep()`). Tilt-gated; keeps data to the wall.
- Live print: forward v, lateral v, sideslip proxy (v_ct), tilt; mark divergence onset. Judge by TILT.

### C2 — Weathervane refit: extend `wv_speed_fit.py`
- Reuse `fit_model.prep()` (solved frames: qfix/wfix/body-velocity/t_us-dedup). Pool the crab runs (+ this session's `*_collect_vq_lateral`).
- Fit roll & yaw angular-accel residual (after regressing out the rate loop) vs `v_body_y`, binned by `v_body_x` AND by `|v_body_y|` → coefficient surface. Test linear-in-sideslip vs nonlinearity at high `v_body_y`.
- Output: updated `roll_wv(vx)`, `yaw_wv` (and any high-sideslip nonlinear term), with R² and the coefficient-vs-speed curve. Compare to `vq_model.json` v0–4 values.

### C3 — Trajectory validation: extend the replay (`vq_sim`/`match_sim`)
- Replay held-out crab + lateral runs through `VQMatchedDynamics` with (a) the current and (b) the refit coefficients.
- Metric: predicted vs actual **sideslip + attitude trajectory** over multi-step horizons (0.1–1 s), and whether the sim reproduces the **divergence onset** (tilt growth at the same sideslip/speed the live run tumbled). Pass = refit beats current AND reproduces the wall within tolerance.
- On pass: write the refit coefficients into `vq_model.json` + confirm `VQMatchedDynamics` consumes them; keep the existing pytest green.

## 5. Parameters & defaults
| param | default | meaning |
|---|---|---|
| `--almax` | {0.6, 1.5, 3.0} | sets forward speed ≈ {2.8, 4.5, 7} (drag-balance) |
| crab ramp rate | ~0.3 (m/s)/s lateral setpoint | gentle, sustained sideslip build |
| `--dur` | 40 s | sweep duration |
| tilt gate | 75° abort | keep data to the wall |

## 6. Acceptance test
1. `collect_vq_crab.py` runs at 3 forward speeds → recorded runs with sustained sideslip up to divergence, dashboard-streamed.
2. `wv_speed_fit.py` extended → weathervane coefficient surface vs (v_body_y, v_body_x) with R², compared to v0–4.
3. Replay validation → refit `VQMatchedDynamics` reproduces live sideslip dynamics + wall onset, beats the current model on held-out runs.
4. `vq_model.json` updated; `tests/test_sim/test_vq_matched.py` still green.

## 7. Risks & mitigations
- **Stabilizer can't hold enough sideslip at high speed** (crab wall ~1.2 m @v7) → limited high-speed data. Mitigation: collect what the wall allows (still more than closed-loop slalom); escalate to Approach B (INDI inner loop) for the deepest regime.
- **Closed-loop bias still partly suppresses sideslip** → the crab is a velocity setpoint, not line-hold, so it commands sustained lateral motion; but the attitude loop still acts. Mitigation: log the full command; identify the moment from the rate residual (open-loop in the rate axis), per `fit_weathervane`.
- **Frame traps** (this session's β bug) → use `prep()`'s solved conventions; judge by tilt; verify signs on the dashboard; never a hand-rolled β metric.
- **Refit overfits the transient/limit-cycle near the wall** → fit on the quasi-steady crab segment, exclude post-divergence rows (collision/tilt-gated).

## 8. Out of scope (YAGNI)
- RL/DAgger retrain on the validated sim (downstream, separate spec; already built).
- The analytic FF/INDI racing controller as a deliverable (INDI may appear only as a data-collection stabilizer if Approach B is needed).
- Perception/vision; gate detection.
- The straight-line speed controller (TRACK-01, done) and the lateral-acquisition tracker fix (separate open item).
