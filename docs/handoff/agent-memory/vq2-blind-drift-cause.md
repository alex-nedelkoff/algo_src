---
name: vq2-blind-drift-cause
description: "The blind gate-1 DR divergence is the roll/pitch TRIM misfiring — not a force, not IMU gaps"
metadata: 
  node_type: memory
  type: project
  originSessionId: ac528723-aa01-44f3-ad7b-c8f642be8516
---

**The blind gate-1 failure is the roll/pitch TRIM injecting false attitude, NOT an
"unmeasured force" and NOT IMU gaps — measured 2026-07-14 (vq2_blind1).**

Evidence chain:
- Frames (blind_approach.jpg): drone flew a NEAR-PERFECT approach — gate dead-centered,
  closing to ~3 m (frames 0-2) — then banked away and flew into the ceiling (3-8).
  So the drone was ON TARGET; the estimate/controller threw it away. **No force moved
  it off** (overturns the 07-10 "inner assist" claim Alex was right to doubt).
- The KF/DR y-estimate swung 0 -> -3.6 -> +4.6 -> +6.3 m; the controller (steers on
  ref - DR) "corrected" the drone off a gate it was about to tick, then ran away.
- IMU-gap hypothesis FALSIFIED (imu_gap_analysis.py): gaps <=22 ms (median 7), Euler vs
  trapezoidal integration diverge only 0.43 deg total / 2 deg max, net yaw ~0. Gyro
  integration + yaw are fine. So NOT the gaps, NOT yaw frame-rotation.
- ROOT CAUSE = the roll/pitch trim (rx_loop ~L411): 54 trim events, repeatedly
  SATURATING at the +-2 deg clamp with oscillating sign. With a noiseless IMU roll
  can't drift, so a trim slamming +-2 deg is misreading the approach's REAL lateral
  accel as roll error and INJECTING false roll -> gravity leaks to lateral -> phantom
  sideways force -> the +-4 m y-drift. The trim's "quiet gates" (gm<0.4, |an-9.81|<1,
  |gz|<0.03) let maneuvering accel through.
- Why blind << servo: blind (DIRTEST=center) stays in 'route' phase where the trim
  runs, so it trims through the whole approach. The servo enters 'approach' which
  CLEARS trim_ok -> trim off. Servo dodges the injection AND locks on the visible gate.

CONFIRMED (vq2_blind_nt, NOTRIM=1, one clean variable): approach y-swing dropped from
**10.06 m -> 0.46 m** (20x), 0 trim events, no runaway/crash (vs trim run crashed). The
trim manufactured the ENTIRE divergence. DR is now accurate enough to thread the gate.
FIX: NOTRIM=1 env flag (added, rx_loop ~L411). Real fix = tighten the trim quiet-gates
or disable it in the terminal approach; robust path = servo + NOTRIM (clean DR + gate
lock). NOTE the AIMBIAS_Y=-2 "aperture offset" was likely COMPENSATING for the trim
drift -- re-check/drop it now that DR is clean. Next: servo + NOTRIM for the tick,
faster VMAX (0.5 timed out).
Related: [[vq2-perception-latency]], [[vq2-pad-acquisition]], [[dpvo-tracking]].
