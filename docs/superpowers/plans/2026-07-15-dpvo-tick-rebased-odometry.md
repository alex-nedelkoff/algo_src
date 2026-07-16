# Tick-Rebased DPVO Gate-to-Gate Odometry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a low-memory WSL2 DPVO path that rebases at judge tick 1, publishes an independent metric route pose, and guides only the gate-1-to-gate-2 transit before CPU fastgate resumes terminal control.

**Architecture:** A pure-Python `dpvo_route` module owns immutable configuration identity, offline scale calibration, tick rebasing, and temporal health checks. The Windows bridge client and WSL service exchange an explicit JSON session configuration before JPEG frames. `vq2wp.py` consumes a separate `state['dpvo_route_p']`; it never calibrates or rejects DPVO against the acceleration-derived KF.

**Tech Stack:** Python 3, numpy, pytest, existing DPVO/torch/WSL2 environments, TCP length-prefixed JSON/JPEG protocol.

## Global Constraints

- GateNet performs only the stationary pad lock and is permanently unloaded after GO.
- CPU `fastgate` owns terminal gate guidance.
- Native Windows `lietorch` is not rebuilt or used.
- No model, corpus, frame, RRD, dependency, or build-cache copies.
- Offline replay reads `C:\Users\alexj\vq2_servo_fg62` and later corpora in place.
- The service uses the existing DPVO checkout and `C:\Users\alexj\DPVO\dpvo.pth`.
- Default live candidate is 32 patches; 24 patches is the low-memory fallback.
- Full-resolution intrinsics are `(226.0, 226.0, 319.5, 179.5)` and scale with image resize.
- All judge scoring uses `gidx2.py`; the first live flight is observe-only.

---

### Task 1: Pure DPVO route state

**Files:**
- Create: `vq2/live/dpvo_route.py`
- Test: `vq2/tests/test_dpvo_route.py`

**Interfaces:**
- Produces: `DpvoSessionConfig`, `scale_intrinsics`, `calibration_identity`, `fit_tick_scale`, and `TickRebasedRoute`.
- `TickRebasedRoute.observe(raw_p, frame_ns, gate_idx, yaw) -> RouteObservation` returns `p`, `healthy`, and `reason` without consulting the flight KF.

- [ ] **Step 1: Write failing configuration and scale tests**

```python
from vq2.live.dpvo_route import (
    DpvoSessionConfig, calibration_identity, fit_tick_scale, scale_intrinsics,
)

def test_half_resolution_scales_calibrated_intrinsics():
    assert scale_intrinsics((226, 226, 319.5, 179.5), (640, 360), (320, 180)) == (
        113.0, 113.0, 159.75, 89.75)

def test_identity_changes_with_tracking_configuration():
    a = DpvoSessionConfig(patches=32, width=640, height=360, stride=2)
    b = DpvoSessionConfig(patches=24, width=640, height=360, stride=2)
    assert calibration_identity(a, "weights") != calibration_identity(b, "weights")

def test_tick_scale_uses_known_gate_displacement():
    scale = fit_tick_scale([0, 0, 0], [2, 0, 0], [6, 0, 0], [10, 0, 0])
    assert scale == 2.0
```

- [ ] **Step 2: Run tests and verify import failures**

Run: `python -B -m pytest vq2/tests/test_dpvo_route.py -v`

Expected: collection fails because `vq2.live.dpvo_route` does not exist.

- [ ] **Step 3: Implement configuration and scale functions**

```python
@dataclass(frozen=True)
class DpvoSessionConfig:
    patches: int = 32
    width: int = 640
    height: int = 360
    stride: int = 2
    intrinsics: tuple[float, float, float, float] = (226.0, 226.0, 319.5, 179.5)
    cuda_fraction: float = 0.48

def scale_intrinsics(intrinsics, source_size, target_size): ...
def calibration_identity(config, model_sha256): ...
def fit_tick_scale(dpvo_g1, dpvo_g2, world_g1, world_g2): ...
```

Validation rejects patches outside `8..96`, non-positive dimensions/stride,
and CUDA fractions outside `(0, 1]`. Identity is canonical SHA-256 JSON.

- [ ] **Step 4: Run configuration tests and verify pass**

Run: `python -B -m pytest vq2/tests/test_dpvo_route.py -v`

Expected: 3 passed.

- [ ] **Step 5: Write failing rebasing and health tests**

```python
def test_route_rebases_on_first_tick_without_position_jump():
    route = TickRebasedRoute(scale=2.0, gate1_world=[6, 0, -1], max_speed=8.0)
    assert route.observe([1, 0, 0], 1_000_000_000, 0, 0).p is None
    at_tick = route.observe([2, 0, 0], 2_000_000_000, 1, 0)
    assert np.allclose(at_tick.p, [6, 0, -1])
    moved = route.observe([2.5, 0, 0], 2_500_000_000, 1, 0)
    assert np.allclose(moved.p, [7, 0, -1])

def test_route_rejects_impossible_temporal_jump():
    route = TickRebasedRoute(scale=1.0, gate1_world=[0, 0, 0], max_speed=3.0)
    route.observe([0, 0, 0], 1_000_000_000, 1, 0)
    bad = route.observe([10, 0, 0], 2_000_000_000, 1, 0)
    assert not bad.healthy
    assert bad.reason == "speed"
```

- [ ] **Step 6: Run rebasing tests and verify expected failures**

Run: `python -B -m pytest vq2/tests/test_dpvo_route.py -v`

Expected: failures because `TickRebasedRoute` is missing.

- [ ] **Step 7: Implement route rebasing and health output**

Use the tick frame's DPVO translation as `raw_origin`, gate 1 as
`world_origin`, and `Rz(yaw_at_tick) @ R_cam_alignment` for relative deltas.
Reject stale/non-monotonic timestamps, speed violations, and non-finite poses.
Do not expose or accept a KF argument.

- [ ] **Step 8: Run focused tests**

Run: `python -B -m pytest vq2/tests/test_dpvo_route.py -v`

Expected: all pass.

### Task 2: Configurable bridge protocol

**Files:**
- Create: `vq2/live/bridge_dpvo.py`
- Create: `vq2/live/dpvo_odom_bridge.py`
- Test: `vq2/tests/test_dpvo_bridge_protocol.py`

**Interfaces:**
- Produces: `encode_packet`, `recv_packet`, `session_message`, and `parse_session_message`.
- Protocol: client sends one length-prefixed `{"type":"session",...}` JSON packet; service responds with `{"type":"ready","identity":...}` before any JPEG packet.

- [ ] **Step 1: Write failing protocol round-trip tests**

```python
def test_session_message_round_trip_uses_32_patches_and_226_fx():
    cfg = DpvoSessionConfig(patches=32)
    wire = session_message(cfg, "abc")
    parsed = parse_session_message(wire)
    assert parsed.config.patches == 32
    assert parsed.config.intrinsics[0] == 226.0
    assert parsed.model_sha256 == "abc"

def test_protocol_rejects_jpeg_before_session():
    with pytest.raises(ValueError, match="session"):
        parse_session_message(b"not json")
```

- [ ] **Step 2: Verify tests fail because protocol module is absent**

Run: `python -B -m pytest vq2/tests/test_dpvo_bridge_protocol.py -v`

- [ ] **Step 3: Implement pure protocol helpers and run tests**

Keep socket I/O wrappers separate from DPVO imports so tests never initialize
CUDA. Validate service configuration before acknowledging `ready`.

Run: `python -B -m pytest vq2/tests/test_dpvo_bridge_protocol.py -v`

Expected: all pass.

- [ ] **Step 4: Port the existing WSL service into the repo**

Replace hardcoded `PATCHES_PER_FRAME=48`, intrinsics 320, and CUDA fraction
0.90 with the validated session configuration. Resize only when requested and
use `scale_intrinsics` values sent by the client. Return `frame_ns`, `n`, pose,
and `dt_ms`; do not write images or model files.

- [ ] **Step 5: Port and modify the Windows bridge client**

Send session configuration on connection, require matching ready identity,
feed replies into `TickRebasedRoute`, and publish:

```python
state['dpvo_route_p'] = observation.p
state['dpvo_route_wall'] = time.time()
state['dpvo_route_healthy'] = observation.healthy
state['dpvo_route_reason'] = observation.reason
```

In observe-only mode, never mutate the KF. In control mode, still never call
`KF.update_position`; map-follow consumes `dpvo_route_p` directly.

- [ ] **Step 6: Add synthetic client-state tests and run both suites**

Run: `python -B -m pytest vq2/tests/test_dpvo_route.py vq2/tests/test_dpvo_bridge_protocol.py -v`

Expected: all pass without CUDA imports.

### Task 3: Read-only tick calibration CLI

**Files:**
- Create: `vq2/dpvo_tick_calibrate.py`
- Test: `vq2/tests/test_dpvo_tick_calibrate.py`

**Interfaces:**
- Consumes: a pose JSONL produced by an offline bridge replay and a corpus `mavlink.jsonl`.
- Produces: one compact calibration JSON via `build_calibration(...)`; never modifies the corpus.

- [ ] **Step 1: Write failing decoder-alignment tests**

Use synthetic race-status rows with a reset followed by `0->1` and `1->2`, and
synthetic DPVO poses timestamped around both ticks. Assert the output includes
the scale, configuration identity, tick timestamps, sample separation, and
model hash.

- [ ] **Step 2: Run and verify failure**

Run: `python -B -m pytest vq2/tests/test_dpvo_tick_calibrate.py -v`

- [ ] **Step 3: Implement calibration extraction**

Reuse the exact `<BQqqIq` decoder law from `vq2/gidx2.py`. Select the first
fresh post-reset `0->1` and subsequent `1->2` segment. Pair each tick to the
nearest DPVO pose within 150 ms. Refuse stale-only, one-tick, identity-mismatch,
or insufficient-pose inputs.

- [ ] **Step 4: Run calibration tests**

Run: `python -B -m pytest vq2/tests/test_dpvo_tick_calibrate.py -v`

Expected: all pass.

### Task 4: Flight integration

**Files:**
- Modify: `vq2/live/vq2wp.py` near DPVO startup and MAPFOLLOW position source
- Test: `vq2/tests/test_dpvo_route.py`

**Interfaces:**
- Consumes: `DPVO_ROUTE=1`, `DPVO_CAL=<json>`, `DPVO_OBSERVE=1`, bridge state fields.
- Produces: map-follow `_pp` sourced from healthy `dpvo_route_p` only after tick 1.

- [ ] **Step 1: Add a failing pure source-selection test**

Add `select_route_position(state, ticks, now, observe_only)` to
`dpvo_route.py`. Assert it returns DPVO only for tick >=1, healthy, fresh
observations in control mode; otherwise returns `None` and a reason.

- [ ] **Step 2: Run and verify failure**

Run: `python -B -m pytest vq2/tests/test_dpvo_route.py -v`

- [ ] **Step 3: Implement source selection and integrate it**

Start the WSL bridge with `DpvoSessionConfig` and loaded calibration identity.
In MAPFOLLOW, prefer the selected DPVO route pose over `_mf_p` and KF position.
During observe-only flight, log comparisons but leave MF_DR control unchanged.
If control mode loses DPVO health before staging, set an explicit abort reason;
do not fall back silently to MF_DR.

- [ ] **Step 4: Parse and test integration**

Run:

```powershell
python -B -c "import ast; ast.parse(open('vq2/live/vq2wp.py', encoding='utf-8').read())"
python -B -m pytest vq2/tests/test_dpvo_route.py vq2/tests/test_dpvo_bridge_protocol.py vq2/tests/test_dpvo_tick_calibrate.py -v
```

Expected: parse exit 0 and all focused tests pass.

### Task 5: Disk-safe deployment and offline verification

**Files:**
- Update in place: `C:\Users\alexj\bridge_dpvo.py`
- Update in place: `C:\Users\alexj\dpvo_odom_bridge.py`
- Update in place: `C:\Users\alexj\vq2wp.py`
- Create small config/calibration only: `C:\Users\alexj\dpvo_fg62_32.json`

**Interfaces:**
- Deployment copies source text only; no model or corpus copies.

- [ ] **Step 1: Recheck disk space and file hashes**

Run `wsl -d Ubuntu df -h / /mnt/c` and hash deployed/repo files before copying.
Abort optional cache-producing work if either filesystem has less than 20 GB.

- [ ] **Step 2: Copy verified repo files in place**

Use native PowerShell `Copy-Item -LiteralPath`; do not delete or move corpora.

- [ ] **Step 3: Start the existing WSL service and run a handshake smoke test**

Expected ready reply: identity matches 32 patches and focal length 226.

- [ ] **Step 4: Replay fg62 read-only at 32 patches**

Write pose output and the compact calibration JSON to `C:\tmp`, then retain
only the final JSON if validation passes. Record DPVO initialization peak,
steady VRAM, latency, and track continuity.

- [ ] **Step 5: Retry at 24 patches only if 32 lacks headroom**

Change one variable. Do not create another environment or model copy.

### Task 6: Final verification and handoff

**Files:**
- Review all changed repo and deployment files.

- [ ] **Step 1: Run the complete VQ2 test suite**

Run: `python -B -m pytest vq2/tests -v`

- [ ] **Step 2: Run syntax checks for every deployed Python file**

Run `ast.parse` against repo and deployed bridge/flight files.

- [ ] **Step 3: Inspect diffs and calibration identity**

Confirm no corpora, weights, build outputs, or unrelated user files changed.

- [ ] **Step 4: Commit only verified repo files**

Use Conventional Commits and exact path staging. Deployment files remain
laptop-local unless separately mirrored into the repo files above.

- [ ] **Step 5: Report the live gate**

Do not claim flight success. Report whether offline replay and memory checks
pass, then provide the exact observe-only batch configuration for the next
fresh simulator attempt.

