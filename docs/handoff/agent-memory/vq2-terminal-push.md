---
name: vq2-terminal-push
description: "SOLVED 07-15: the 'IMU-invisible rightward push' is EST-YAW error x forward speed — our vy_b loop manufactures a real crab (~vx*sin(yaw_err)); no environmental force exists"
metadata:
  node_type: memory
  type: project
  originSessionId: ac528723-aa01-44f3-ad7b-c8f642be8516
---

**OVERTURNED 2026-07-15 (three-probe measurement, session fg15-30 + hp1/st1).
There is NO environmental push. The rightward drift is self-inflicted:**

1. **Hover probe** (hp1: RELEVEL then 12 s zero-cmd hover, fastgate 30 Hz hole
   bearing as truth): lateral drift 0.03-0.08 m/s — noise level. No push at hover.
2. **Straight probe** (st1: RELEVEL then 5 s straight at 0.8 m/s, yaw PINNED at
   0.0001 rad, gz~0): TRUTH (frames) shows ~1-1.5 m rightward translation over the
   4 m leg (~0.2-0.3 m/s) while est-y moves 0.03 m. Est-blind, velocity-coupled.
3. Terminal misses (fg23/26/27/28/29, 5x identical): 0.5-0.9 m right over the last
   ~3 m at 1.8 m/s — ratio ~0.2-0.3 of forward speed, same as the straight probe.

**Mechanism:** the takeoff level-off banks attitude error across the bursty IMU
gaps (07-12 mechanism). RELEVEL repairs roll/pitch from gravity but gravity gives
NO yaw information — the banked yaw error (~10-17 deg) survives. level_cmd nulls
vy_b = est-world velocity rotated by EST yaw; with yaw off by theta, real forward
flight reads as fictitious lateral velocity vx*sin(theta), the roll loop cancels
it, and that CREATES a real crab ~vx*sin(theta), after which est vy reads 0.
Explains: always-rightward (identical takeoff maneuver), IMU/est-invisible (by
construction), historical AIMBIAS_Y=-2, why the blind chute needed heading-lock
(yr=0 stops yaw error from banking), and why FGPURSUIT (yaw unlocked for
turn-to-target) resurrected the drift.

**UPDATE (07-15 late, fg31-33): the yaw-crab mechanism was FALSIFIED by a
sign test** — flying the calibration leg BACKWARD, the drift stayed RIGHTWARD
(+0.13 m/s); a yaw-error crab must flip with vx. Measured facts: rightward,
motion-gated (~0 at hover), direction-independent, 0.08-0.3 m/s, varies
run-to-run. Mechanism OPEN (sim-side translation effect, or attitude error
banked per-maneuver).

**YAWCAL implemented in vq2wp** (default-on with FGPURSUIT): after RELEVEL,
yaw-center the tracked hole, fly a short BACKWARD pinned-yaw leg (forward legs
thread the start-light pole corridor — fg31 hit one), fit hole-bearing walk x
range -> per-flight drift measurement -> constant opposite vy bias in all
pursuit branches + punch. fg33: lateral channel held (bearing pinned, no slide).

**THE DEEPER ILLNESS (fg33 frames):** level_cmd closes vx/vy/vz on EST velocity,
which accrues ~1+ m/s offsets after maneuvers — fg33 flew backward-upward in
truth while est read "vx +0.9 satisfied". Each fixed channel surfaces the next.
NEXT: pure-attitude pursuit mode (no est-velocity feedback: roll bias from
bearing P+I, pitch bias for speed, throttle from visual vertical + hover trim;
est only for guards).
Related: [[vq2-blind-drift-cause]], [[vq2-fastgate]], [[vq2-perception-latency]].
