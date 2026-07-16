---
name: vq2-pad-acquisition
description: "Why VQ2 scored flights abort at \"NO PAD ACQUISITION\" — spawn-pitch, not GateNet"
metadata: 
  node_type: memory
  type: project
  originSessionId: ac528723-aa01-44f3-ad7b-c8f642be8516
---

The VQ2 pad-acquisition abort (`NO PAD ACQUISITION -- aborting before takeoff`,
vq2wp.py ~line 944) is a **spawn-pitch problem, not a GateNet problem** — measured
& proven 2026-07-13.

- GateNet works: `detector ready` prints every run; on a nose-down spawn it locks
  gate 1 at ~6.2 m instantly (`pad lock: g_lvl [6.22 0.03 -0.36]`).
- Pad-acq waits for `state['obs']` (a GateNet detection) or `fix_count>=3` within
  15 s. The nose camera is fixed **20° up** (CAM_TILT); gate 1 sits ~1.35 m *below*
  the pad (≈12° below horizontal at 6 m).
  - **Nose-down spawn (−17.8°):** cam net axis ≈ +2° → gate centered → locks. ✅
  - **Level spawn (0.0°):** cam axis +20° → gate at −12° falls just below the FOV
    → nothing to detect → abort. ❌ (The sim's on-screen render still shows the
    gate — that's NOT the nose-cam FOV.)
- The hard reset `param1=1` **preserves** the current pad attitude (watched pitch
  hold −17.8° for 30 s post-reset); it does NOT set or settle it.
- A **genuinely fresh relaunch+login spawns nose-down −17.8°** (good). The three
  07-13 scored aborts at 0.0° were a *poisoned restart* state (menu-restart churn /
  leftover), not an inherent fresh-session property. **Verify pitch −17.8° with a
  passive HIGHRES_IMU read (spawn_att_probe.py) BEFORE launching a scored flight.**

Passive-probe recipe: `atan2(ax, sqrt(ay²+az²))` on HIGHRES_IMU accel = resting
pitch; at rest ax≈−3.00, az≈−9.34 → −17.8°. Related: [[dpvo-live-scale]].
