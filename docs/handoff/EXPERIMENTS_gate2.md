# Gate-2 terminal experiments — SOTA/adjacent-field trials (2026-07-15 evening)

Context: gate-1 tick ~90% reliable (vision punch). Gate-2 leg 1/11 — eleven
patch-layers on a mode-switched vision/DR handoff. Research pass (racing SOTA +
adjacent fields) picked three replacements to trial:

- **E1. PN terminal (missile guidance):** collision course == LOS rate nulled.
  Punch when |bearing-rate| small (not when bearing centered). Blind-range
  robust by construction: once LOS rate ~ 0 at 3-5 m, vision loss is harmless.
- **E2. Go-around doctrine (aviation):** approach not stabilized by the
  decision range -> abort to staging and retry. Never salvage. Converts
  crashes into retries (missiles get one shot; we don't).
- **E3. Cyan-line following (autonomous driving):** the sim paints the racing
  line through every gate; a lane-keeping controller on CPU color segmentation
  replaces map/DR/staging for transit. (Fresh-session scale.)
- **VO note (Alex):** expect VO in the loop eventually — slot = transit
  odometry (EKF prediction source), replacing kinematic DR. Blocked on the
  lietorch Windows build (clean 14.38 rebuild still AVs; WSL2 build is the
  fallback). The classical layer (E1-E3) is deliberately VO-independent so VO
  becomes an accuracy upgrade, not a dependency.

Protocol per run: fresh sim (menu RESTART, screenshot-verified), nose-down
-17.8 verified, gidx2 scoring, frames-before-theories on failures.

| Run | Change under test | Gate1 | Gate2 | Outcome / evidence |
|-----|-------------------|-------|-------|--------------------|
| fg74 | E1 PN punch trigger + E2 go-around, on the fg73 stack | TICK | timeout, no crash | E1/E2 behaved correctly-negatively: PN refused to punch (no collision course ever established), go-around never armed (hole never reached w>=80 in view). Root cause per frames: CEILING CLIMB AGAIN despite the relative est-z hold -- est-z under-reads climbs ~40% (documented), so the hold is satisfied while truth ascends. **Conclusion: vertical needs a real observation, not an est-relative one.** Two candidates: VO (blocked on lietorch), or E3 -- the cyan line row-position in the image is a direct height-over-course observation, CPU-only. Proceeding to E3 minimal: cyan mask -> lateral offset + row -> roll/thr corrections on the corridor. |
