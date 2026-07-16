---
name: dpvo-tracking
description: "The gate-1 killer is DPVO tracking quality (freeze/divergence), NOT scale"
metadata: 
  node_type: memory
  type: project
  originSessionId: ac528723-aa01-44f3-ad7b-c8f642be8516
---

**Definitive test 2026-07-14 (reconstruct_corpus.py -> recon_vq2_scored8.npz: offline
DPVO on the flight's frames, KF-INDEPENDENT; recon_vs_dr.py aligns via the
deterministic climb-scale recipe and overlays on the DR path).**

RESULT — the live-DPVO-for-gate-1 killer is TRACKING QUALITY, not scale:
- Scale does NOT drift: climb-scaled offline DPVO matches DR displacement early
  (ratio 0.88->1.04->1.02->0.98->0.96), session scale ~2.2 m/unit. So DPVO is roughly
  metric; the flight's locked scale 21 was the gate-2 CALIBRATION bug, fixable.
- BUT DPVO loses tracking and FREEZES: poses identical (4.17,-14.27,1.96) for the last
  third of the flight, keyframes stalled at ~32. And even before the freeze its
  trajectory SHAPE diverged from DR (y -2.5->-14 vs DR y +1.8->+5.9).
- The freeze coincides with the hover-HOLD (Part B, GNSCALE_HOLD): DPVO needs
  continuous parallax; a low-motion hold starves it -> drift then freeze. **Part B is
  counterproductive for DPVO.**
- Offline used 96 patches (more robust than live's 48) and STILL froze -> live DPVO is
  even weaker.

CONCLUSION: no scale-calibration scheme fixes a tracker that freezes. Live DPVO cannot
reliably fly the short/low-parallax gate-1 approach. Reserve DPVO for continuous-motion
legs (gates 2+, cruise); fly gate 1 by another means (nose-down GateNet servo = every
historical tick). Answers Alex's 'is the gatenet lock late / scale drifting' thread:
lock latency real (~1s, [[vq2-perception-latency]]) but NOT in the gate-1 steering loop;
scale NOT drifting; the wall is DPVO tracking. Related: [[vq2-dpvo-live-scale]],
[[vq2-perception-latency]], [[vq2-pad-acquisition]].
