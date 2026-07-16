# GateNet-DPVO Scale Observe Attempt 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run one valid observe-only scale-calibration creep with DPVO poses available during the GateNet anchor window.

**Architecture:** Preserve the reviewed atomic pairing and robust estimator. Start the already-prewarmed DPVO tracking loop immediately instead of waiting for GO, while the launcher enables the existing recenter creep, disables visualization networking, retains the measured 12-second GateNet overlap, and records to a fresh `obs2` corpus.

**Tech Stack:** Python 3, pytest, Windows batch, WSL2 DPVO bridge, AI-GP simulator.

## Global Constraints

- DPVO remains observe-only and MF_DR remains the sole controller.
- Keep `GNSCALE_TMAX=12.0`; do not expand the measured GPU window.
- Enable `RECENTER=1` so the existing 18-second safe creep actually executes.
- Set `VIZ=0` so an empty RRD path cannot fall back to the Mac gRPC viewer.
- Use `C:/Users/alexj/vq2_servo_dpvo_gnscale_obs2`; fail preflight if it exists.
- Run exactly one fresh attempt, score only with `gidx2.py`, and menu-restart after a collision.

---

### Task 1: Lock Corrected Timing and Launcher Safety

**Files:**
- Modify: `vq2/tests/test_dpvo_gate_scale_bridge.py`
- Modify: `vq2/tests/test_dpvo_gnscale_observe_batch.py`
- Modify: `vq2/live/dpvo_odom_bridge.py`
- Modify: `vq2/live/fly_servo_dpvo_gnscale_observe.bat`

**Interfaces:**
- Consumes: existing `track_during_gatenet()` observe-only guard.
- Produces: immediate post-prewarm tracking in GateNet scale mode and a fresh attempt-2 launcher.

- [ ] **Step 1: Write failing tests**

Require the bridge source to enter `_track()` without a `go_passed` wait in scale mode. Require the launcher to contain `RECENTER=1`, `VIZ=0`, `GNSCALE_TMAX=12.0`, `DPVO_OBSERVE=1`, and the fresh `obs2` record path.

- [ ] **Step 2: Verify red**

Run: `python -m pytest vq2/tests/test_dpvo_gate_scale_bridge.py vq2/tests/test_dpvo_gnscale_observe_batch.py -q`

Expected: failures for the GO wait, `RECENTER=0`, missing `VIZ=0`, and the occupied obs1 path.

- [ ] **Step 3: Apply the minimal timing/configuration correction**

Remove only the scale-mode GO wait from `DpvoOdom._run()`, retain the observe-only guard, log `pre_go=True`, and update the four launcher settings specified above.

- [ ] **Step 4: Verify green and regression suite**

Run: `python -m pytest vq2/tests -q`

Expected: all non-skipped VQ2 tests pass.

### Task 2: Deploy and Run Once

**Files:**
- Deploy: `C:\Users\alexj\dpvo_odom_bridge.py`
- Deploy: `C:\Users\alexj\fly_servo_dpvo_gnscale_observe.bat`
- Produce: `C:\Users\alexj\vq2_servo_dpvo_gnscale_obs2`

**Interfaces:**
- Consumes: reviewed attempt-2 source and the existing DPVO WSL service launcher.
- Produces: one corpus with exact-time GateNet/DPVO pairs or explicit timing evidence explaining why none were possible.

- [ ] **Step 1: Deploy with hash verification**

Back up overwritten live files, copy the two changed files, and require source/deployed SHA-256 equality.

- [ ] **Step 2: Fresh simulator preflight**

Screenshot-verify the restart menu and pad scene, run the spawn attitude probe, verify the corpus path is absent, and start the WSL bridge hidden with dedicated attempt-2 logs.

- [ ] **Step 3: Launch exactly one attempt**

Start the batch hidden with dedicated stdout/stderr logs and monitor through exit. Do not launch a second run automatically.

- [ ] **Step 4: Analyze and clean up**

Score with `gidx2.py`; report anchor/pair counts, first-pose timing, candidate scale/MAD/baseline/readiness, GPU minimum free memory, failures, corpus size, and control source. Stop the task-owned WSL service, restart the simulator after collision, and append the result to the Mac vault or outbox.
