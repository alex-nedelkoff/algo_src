# Cornering at speed — coordinated turn + weathervane-FF (COR-127) — design

**Date:** 2026-06-09
**Status:** Design approved, pending implementation plan
**Scope:** An analytic coordinated-turn controller that corners at racing speed on live VQ, stabilized by an explicit weathervane feedforward. Validated on ONE corner first (the core physics unlock + the A/B that proves the FF). Gate-to-gate chaining and DAgger/RL are downstream specs.

## 1. Motivation

Cornering at speed is mandatory for drone racing and is the last un-cracked piece of the analytic stack:
- **Straight-line speed:** SOLVED — drag/tilt-saturation, ~8–10 m/s, tunable via the tilt budget (REFIT-02). `traj_track` flies a straight at 8.5 m/s live (TRACK-01).
- **Cornering:** a turn of curvature κ at speed v needs centripetal accel `v²·κ`, which competes with drag for the tilt budget (`v=√(g·tan φ/(c+κ))` caps it). The *hard* part is rotating the nose through the turn: this session proved **nose-follows-tangent triggers the weathervane** (the translated/curve tumble, t<3 s). A **coordinated turn** — nose tracks the velocity vector so sideslip β≈0 — is the right structure, and `coord_turn.py` already implements it (3 FF terms: coordinated bank `φ=atan(v²/Rg)`, yaw-rate `ψ̇=v/R`, pitch-coupling-comp `sin φ·ψ̇`; nose-tracks-velocity; ZVD yaw shaping; speed governed to `V_MAX=√(R·g·tan φ_safe)`).

**Why prior coordinated turns still failed:** they rely on β→0 *alone*. β never stays exactly 0 (entry/exit transients, tracking error), and the weathervane *amplifies* any β — so a small β error diverges. `coord_turn` is entry-fragile (CRUISE-02 pinned at v=0); `indi_course` (FF+INDI) got 2/5 laps, tumbling on sharp gate-to-gate where velocity-aim snaps → sideslip → tumble. None compensated the weathervane.

**The unlock — explicit weathervane feedforward.** WV-DYNAMIC (this session) *validated* the weathervane model in the dynamic regime. So we can now **cancel `wv·v_body_y` directly in the rate command** — the missing piece. Coordinated turn (β→0) + weathervane-FF (kills the residual-β divergence) + tilt-budget speed cap = corner at speed, stably. The interface is unchanged (ACRO rate `[thr,wx,wy,wz]`), so this is consistent with the RL sim for later transfer.

## 2. Goal & success criteria

Execute ONE coordinated corner at racing speed on the live VQ sim:
- R≈10 m, speed ramped to ~6 m/s (well above coord_turn's validated 2.5).
- β (sideslip) bounded < ~20°; tilt bounded (no tumble, never the abort).
- Completes ≥90° of heading change (a real corner, not a straight).
- **A/B test:** with the weathervane-FF vs without — the FF keeps β bounded / prevents the divergence that bare coord_turn shows at speed. This proves the FF is the unlock.

Done = a corner flown at ~6 m/s with bounded β, and evidence the weathervane-FF is what made it possible.

## 3. The physics

- **Tilt budget:** total horizontal accel `g·tan φ`. A corner spends `c·v²` (drag) + `v²·κ` (centripetal). Cap `v_corner=√(g·tan(φ_budget)/(c+κ))`, c≈0.057.
- **Coordinated turn:** bank `φ=atan(v²/Rg)` provides centripetal; nose tracks the velocity so β≈0; yaw-rate FF `ψ̇=v/R` rotates the heading with the velocity; pitch-comp `sin φ·ψ̇` cancels the bank×yaw-rate kinematic pitch coupling (the speed pump). All from coord_turn.
- **Weathervane (validated model):** `om_roll += (roll_wv0+roll_wv1·v_body_x)·v_body_y·dt`, `om_yaw += yaw_wv·v_body_y·dt`. A nonzero β=`atan2(v_body_y, v_body_x)` drives a destabilizing moment. **FF cancellation** (command goes through `ω=G·wcmd`, so divide by G):
  - `wcmd_roll_ff = −(roll_wv0+roll_wv1·v_body_x)·v_body_y / G_roll`
  - `wcmd_yaw_ff  = −yaw_wv·v_body_y / G_yaw`
  Added to the existing rate command, this subtracts the weathervane moment → the attitude/heading loop is no longer fought by it → β stays bounded through the turn.

## 4. Components

### C1 — Coordinated-turn FF (reuse `coord_turn.py`)
Keep the 3 FF terms + nose-tracks-velocity + ZVD yaw shaping + ramped onset, as-is. The new build wraps/extends it; do not re-derive the turn geometry.

### C2 — Weathervane-FF (the new piece)
Add `wcmd_roll_ff`, `wcmd_yaw_ff` (§3) to the rate command, using the validated coefficients (`roll_wv0=−0.105, roll_wv1=−0.019, yaw_wv=−0.149` from `vq_model.json`/`vq_matched.py`) and **live `v_body_x, v_body_y`**.
- **`v_body_y` computation is the frame-critical part** (this session's β bug came from mixing body/world velocity). Compute `v_body = R(quat)ᵀ · vel_world` (or the project's established body-velocity convention) and VERIFY the sign by the effect: with the FF on, β must SHRINK, not grow. A `--wvff_sign` flag + a `--no-wvff` A/B switch make verification mechanical. Never assume the sign.
- Gate the FF magnitude (clip) so a bad β estimate can't command a huge rate.

### C3 — Tilt-budget speed cap
`v_corner=√(g·tan(φ_budget)/(c+κ))` with κ=1/R. Use coord_turn's `V_MAX` style governor, but with the drag term included (so it's the same law as `gate_traj`). Ramp speed up to `v_corner` (no step).

### C4 — Validation (single corner)
A coordinated corner (R≈10), speed ramped to ~6, on live VQ. Record + dashboard + tilt gate. Runs: (a) `--no-wvff` (bare coord_turn at speed) → expect β growth/divergence; (b) weathervane-FF on → β bounded, completes the turn. Judge by **tilt + β on the dashboard**; sign-verify the FF.

## 5. Parameters & defaults
| param | default | meaning |
|---|---|---|
| R | 10 m | corner radius |
| φ_budget | 30° | tilt budget for the corner |
| c_drag | 0.057 | drag (REFIT-02) |
| roll_wv0/1, yaw_wv | −0.105/−0.019, −0.149 | validated weathervane coeffs |
| `--wvff_sign` | +1 | FF sign, verified live |
| `--no-wvff` | off | A/B switch |

## 6. Acceptance test
1. `--no-wvff` corner at v→6: record β/tilt (expect the bare-coordinated-turn failure mode — β grows or it tumbles).
2. weathervane-FF on: β bounded <~20°, tilt bounded, ≥90° turn completed.
3. The FF sign is verified (β shrinks with FF on); the A/B shows the FF is decisive.

## 7. Risks & mitigations
- **`v_body_y` frame/sign wrong → FF amplifies β (diverges faster).** Mitigation: `--wvff_sign` + `--no-wvff` A/B; verify on the dashboard before trusting; clip the FF magnitude; start at low speed.
- **Model error in the dynamic regime** (the weathervane fit is noisy, R²0.2–0.4). Mitigation: the FF only needs to be approximately right to keep β bounded; if explicit-FF is too fragile, escalate to Approach 2 (INDI, model-free rejection).
- **Entry/exit transient** (coord_turn's known fragility). Mitigation: ramp all FF onsets (coord_turn already does, 4 s); start the corner from established forward flight.
- **Speed overshoot into the wall** (TRACK-01). Mitigation: ramp speed to `v_corner`, don't step; the tilt-budget cap bounds it.

## 8. Out of scope (YAGNI / downstream specs)
- Gate-to-gate chaining (integrate the coordinated turn into the `gate_traj` tracker over a multi-gate course).
- DAgger/RL on the validated sim (the analytic corner becomes the teacher).
- The straight-line speed controller (TRACK-01, done) and the lateral-acquisition tracker fix.
- INDI inner loop (Approach 2) — only if the explicit weathervane-FF proves too fragile.
