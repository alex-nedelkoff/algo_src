---
name: vq2-perception-latency
description: GateNet inference is ~0.45-1.0s late -- a prime suspect for gate-1 crashes
metadata: 
  node_type: memory
  type: project
  originSessionId: ac528723-aa01-44f3-ad7b-c8f642be8516
---

**GateNet perception latency is huge and likely a PRIMARY cause of the gate-1
blind-chute crashes** -- measured 2026-07-14 (det_latency.py, matches each detection's
jlog wall `t` to its source frame `rx_wall` by sim_ns).

- frame_rx -> detection-computed latency: **median 454 ms GateNet-alone (scored5);
  median 1029 ms (max 1255) when GateNet + WSL2-DPVO share the GPU under GNSCALE
  (scored7).** The overlap DOUBLES it.
- Effective detection rate: 2.1 Hz alone, **1.2 Hz** under overlap.
- Total staleness (latency + half a cadence gap): ~690 ms alone, ~1450 ms overlap.
  At VMAX=0.6 m/s the drone travels **41 cm / 87 cm** during that staleness.
- The frame STREAM is fine (29 fps) -- the bottleneck is GateNet inference on the
  VRAM-starved 4 GB RTX 3050 (a SegFormer-b2 pass should be ~20-50 ms; it's ~450 ms).
- det_loop reads state['frame'] (latest, overwritten by cam_loop) so latency ~=
  inference time, not a growing queue.

WHY IT MATTERS: the staleness travel (41-87 cm) is BIGGER than the ~0.53 m crash miss.
The drone steers toward where the gate sat ~0.5-1 s ago and is already past it when it
acts. This likely explains why DPVO position fixes never helped -- the gate-1 error is
substantially WHEN the info is from (latency), not WHERE the drone thinks it is. Note
vq2wp already latency-compensates ATTITUDE (ATT_BUF, det_loop ~L589) but NOT the
detection's translation/bearing.

FIXES to try: (a) latency-compensate the detection -- propagate g_lvl forward by the
drone's motion (v * latency) before the servo uses it; (b) speed up inference (lower
input res / half-precision / fewer decode candidates); (c) do NOT overlap GateNet+DPVO
(GNSCALE doubles latency -- argues against the live-DPVO-for-gate-1 path). Related:
[[vq2-dpvo-live-scale]], [[vq2-pad-acquisition]].
