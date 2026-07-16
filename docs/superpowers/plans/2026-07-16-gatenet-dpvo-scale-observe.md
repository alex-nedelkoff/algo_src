# GateNet-DPVO Scale Observe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether exact-time GateNet metric observations can produce a stable monocular DPVO scale without giving DPVO any control authority.

**Architecture:** A focused estimator stores timestamped DPVO poses and GateNet-derived metric drone positions, linearly interpolates DPVO at each GateNet source-frame timestamp, and robustly fits scale from same-gate relative displacements. The Windows bridge runs tracking during a bounded GateNet overlap only when `GNSCALE=1` and `DPVO_OBSERVE=1`, publishes diagnostic state and JSONL events, but never modifies the route scale or sets the legacy `dpvo_scale_locked` control flag.

**Tech Stack:** Python 3, NumPy, pytest, existing Windows-to-WSL2 DPVO bridge, GateNet telemetry, Windows batch flight launcher.

## Global Constraints

- Calibration is telemetry-only: it must not alter the KF, `TickRebasedRoute.scale`, flight commands, or `state['dpvo_scale_locked']`.
- Pair GateNet observations using their camera source timestamp `gatenet_ns`, not detector completion wall time.
- Never mix observations from different gate identities in one displacement fit.
- Require at least 8 paired observations, at least 1.0 m metric baseline, and robust relative dispersion no greater than 10% before reporting a candidate ready.
- Run only one observe-only simulator attempt and preserve the existing MF_DR control path.
- Record to a new corpus directory and keep Rerun recording disabled to limit disk use.
- Do not overwrite or revert the existing dirty changes in `vq2/live/vq2wp.py` or `vq2/live/dpvo_odom.py`.

---

### Task 1: Exact-Time Robust Scale Estimator

**Files:**
- Create: `vq2/live/dpvo_gate_scale.py`
- Create: `vq2/tests/test_dpvo_gate_scale.py`

**Interfaces:**
- Produces: `GateScaleCalibrator.add_dpvo(frame_ns: int, raw_p: Iterable[float]) -> list[GateScaleEstimate]`.
- Produces: `GateScaleCalibrator.add_gatenet(frame_ns: int, gate_id: int, metric_p: Iterable[float]) -> list[GateScaleEstimate]`.
- Produces: immutable `GateScaleEstimate` fields `scale`, `relative_mad`, `baseline_m`, `pair_count`, `sample_count`, `ready`, `gate_id`, and `frame_ns`.

- [ ] **Step 1: Write failing interpolation and robust-fit tests**

```python
def test_pairs_gatenet_at_source_time_by_interpolating_dpvo():
    cal = GateScaleCalibrator(min_samples=3, min_baseline_m=1.0)
    cal.add_dpvo(0, (0, 0, 0))
    cal.add_dpvo(2_000_000_000, (1, 0, 0))
    estimates = cal.add_gatenet(1_000_000_000, 0, (2, 0, 0))
    assert cal.pairs[-1].raw_p == (0.5, 0.0, 0.0)

def test_reports_stable_scale_only_after_sample_baseline_and_dispersion_gates():
    cal = GateScaleCalibrator(min_samples=8, min_baseline_m=1.0,
                              max_relative_mad=0.10)
    for index in range(10):
        ns = index * 1_000_000_000
        cal.add_dpvo(ns, (index * 0.1, 0, 0))
        estimates = cal.add_gatenet(ns, 0, (index * 0.2, 0, 0))
    assert estimates[-1].ready
    assert estimates[-1].scale == pytest.approx(2.0)

def test_outlier_and_other_gate_do_not_create_bad_lock():
    # One corrupt metric point and a different gate identity cannot pull the
    # same-gate median scale away from 2.0 or contribute cross-gate baselines.
```

- [ ] **Step 2: Run tests and verify import failure**

Run: `python -m pytest vq2/tests/test_dpvo_gate_scale.py -q`

Expected: FAIL because `vq2.live.dpvo_gate_scale` does not exist.

- [ ] **Step 3: Implement timestamp interpolation and robust pairwise fitting**

Create a dependency-light module using bounded deques, finite 3-vectors, bracketing timestamp interpolation with a maximum 250 ms bracket span per side, same-gate pairwise distance ratios, median/MAD aggregation, and the global readiness thresholds. Deduplicate both DPVO and GateNet timestamps.

- [ ] **Step 4: Run estimator tests**

Run: `python -m pytest vq2/tests/test_dpvo_gate_scale.py -q`

Expected: all estimator tests PASS.

### Task 2: Observe-Only Bridge Integration

**Files:**
- Modify: `vq2/live/dpvo_odom_bridge.py`
- Create: `vq2/tests/test_dpvo_gate_scale_bridge.py`

**Interfaces:**
- Consumes: `GateScaleCalibrator` from Task 1.
- Produces: `publish_gate_scale(state: dict, estimate: GateScaleEstimate, wall: float) -> None` diagnostic fields prefixed `dpvo_gate_scale_`.
- Produces: JSONL events `dpvo_gate_scale_pair` and `dpvo_gate_scale_fit` with source timestamp, pairing residual/bracket information, scale, dispersion, baseline, sample count, and readiness.

- [ ] **Step 1: Write failing bridge safety tests**

```python
def test_publish_gate_scale_never_sets_control_lock_or_route_scale():
    state = {}
    publish_gate_scale(state, estimate, wall=10.0)
    assert state['dpvo_gate_scale_ready'] is True
    assert 'dpvo_scale_locked' not in state

def test_gnscale_overlap_requires_observe_only():
    assert track_during_gatenet({'GNSCALE': '1', 'DPVO_OBSERVE': '1'})
    assert not track_during_gatenet({'GNSCALE': '1', 'DPVO_OBSERVE': '0'})
```

- [ ] **Step 2: Run bridge tests and verify failure**

Run: `python -m pytest vq2/tests/test_dpvo_gate_scale_bridge.py -q`

Expected: FAIL because the bridge helpers do not exist.

- [ ] **Step 3: Integrate estimator into the bridge**

Instantiate the calibrator only in `GNSCALE=1` observe mode. During each successful pose reply, add the pose, ingest each new `gatenet_ns` exactly once, publish diagnostic state, and log every successful pair/fit. In `_run`, bypass `wait_for_gpu_handoff` only for this explicitly observe-only overlap mode; retain the existing handoff behavior for all control-capable configurations.

- [ ] **Step 4: Run focused and existing bridge tests**

Run: `python -m pytest vq2/tests/test_dpvo_gate_scale_bridge.py vq2/tests/test_dpvo_bridge_client.py vq2/tests/test_dpvo_prewarm.py vq2/tests/test_dpvo_tick_alignment.py -q`

Expected: all tests PASS.

### Task 3: Flight Configuration and Validation

**Files:**
- Create: `vq2/live/fly_servo_dpvo_gnscale_observe.bat`
- Modify only if a test proves necessary: `vq2/live/vq2wp.py`

**Interfaces:**
- Consumes: bridge diagnostics from Task 2 and existing `GNSCALE` GateNet telemetry in `vq2wp.py`.
- Produces: new corpus `C:/Users/alexj/vq2_servo_dpvo_gnscale_obs1` and log `C:/Users/alexj/fly_servo_dpvo_gnscale_obs1.log`.

- [ ] **Step 1: Write a static batch-safety test**

Assert the batch contains `GNSCALE=1`, `DPVO_OBSERVE=1`, `DPVO_ROUTE=1`, `NOFIX=1`, `RRD=`, a unique `RECORD`, and does not enable DPVO control.

- [ ] **Step 2: Run the static test and verify failure**

Run: `python -m pytest vq2/tests/test_dpvo_gnscale_observe_batch.py -q`

Expected: FAIL because the batch does not exist.

- [ ] **Step 3: Create the one-run batch**

Copy the known obs2 control configuration, enable `GNSCALE=1`, use a bounded 12-second overlap, retain 32 patches and the matching observe-only calibration identity, disable RRD, and write to the new corpus.

- [ ] **Step 4: Run the complete focused test set**

Run: `python -m pytest vq2/tests/test_dpvo_gate_scale.py vq2/tests/test_dpvo_gate_scale_bridge.py vq2/tests/test_dpvo_gnscale_observe_batch.py vq2/tests/test_dpvo_bridge_client.py vq2/tests/test_dpvo_prewarm.py vq2/tests/test_dpvo_tick_alignment.py vq2/tests/test_dpvo_observe_batch.py -q`

Expected: all tests PASS.

- [ ] **Step 5: Deploy only the changed live files**

Copy `dpvo_gate_scale.py`, `dpvo_odom_bridge.py`, and `fly_servo_dpvo_gnscale_observe.bat` to `C:\Users\alexj\`, preserving the source files and recording deployed hashes.

- [ ] **Step 6: Run one fresh observe-only flight**

Verify the simulator pause/menu state visually, restart the race according to the repository procedure, launch the batch detached with its dedicated log, and monitor until process exit or a safety abort. Do not launch a second attempt automatically.

- [ ] **Step 7: Analyze the run**

Check process exit, GPU survival, GateNet unload timing, DPVO first-pose timing, exact-time pair count, scale median/MAD/baseline/readiness, collisions, and corpus size. Score gate ticks only with `gidx2.py` and report that DPVO remained observe-only.
