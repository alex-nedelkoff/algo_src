# DPVO Prewarm Tick Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prewarm the low-memory WSL2 DPVO service before GO, start frame submission only after GateNet releases the GPU, and produce a pose within 150 ms of the first judge tick without changing flight control.

**Architecture:** `DpvoOdom` becomes a staged lifecycle: configure, prewarm, wait for an explicit GateNet handoff, then track continuously. `vq2wp.py` starts route DPVO immediately after the hard reset, using the reset-settle and pad-acquisition windows for prewarming, and publishes the handoff only after GateNet tensors are deleted and its CUDA cache is emptied. Existing tick rebasing, 32-patch configuration, calibration, MF_DR control, and CPU fastgate behavior remain unchanged.

**Tech Stack:** Python 3, pytest, pymavlink flight script, PyTorch/CUDA GateNet, existing WSL2 DPVO TCP bridge, PowerShell deployment.

## Global Constraints

- The next flight remains `DPVO_OBSERVE=1`; DPVO must not influence control.
- Change one live variable only: DPVO lifecycle timing.
- Keep 32 patches, 640x360 input, stride 2, CUDA fraction 0.48, 150 ms tick tolerance, and 8 m/s health limit.
- GateNet remains active through pad lock and is never reloaded after GO.
- DPVO remains active after gate 1; multi-gate rebasing is a later change.
- Do not copy weights, corpora, frames, RRD files, environments, or caches.
- Preserve all pre-existing dirty worktree changes. Do not stage or commit overlapping campaign files without a separate exact-path review.
- Score only a fresh simulator session and only with `gidx2.py`.

---

## File structure

- `vq2/live/dpvo_odom_bridge.py`: staged DPVO lifecycle, GO deadline, GPU-handoff wait, and timing telemetry.
- `vq2/live/vq2wp.py`: route-DPVO startup immediately after the hard reset and authoritative GateNet-unloaded signal.
- `vq2/tests/test_dpvo_prewarm.py`: pure lifecycle and staged-order tests with no CUDA imports.
- `vq2/tests/test_vq2wp_dpvo_route_wiring.py`: source-level flight wiring assertions.
- `vq2/live/fly_servo_dpvo_observe.bat`: fresh observe-only record target; no obsolete fixed GPU-settle delay.
- `vq2/tests/test_dpvo_observe_batch.py`: batch safety assertions.

### Task 1: Pure lifecycle gates

**Files:**
- Modify: `vq2/live/dpvo_odom_bridge.py`
- Create: `vq2/tests/test_dpvo_prewarm.py`

**Interfaces:**
- Produces: `prewarm_abort_reason(state: dict) -> str | None`
- Produces: `wait_for_gpu_handoff(state: dict, sleep=time.sleep, poll_s: float = 0.02) -> str`
- Returns only `"handoff"`, `"stop"`, or `"prewarm_late"` at lifecycle boundaries.

- [ ] **Step 1: Write the failing lifecycle tests**

```python
from __future__ import annotations

from vq2.live.dpvo_odom_bridge import (
    prewarm_abort_reason,
    wait_for_gpu_handoff,
)


def test_prewarm_deadline_is_go():
    assert prewarm_abort_reason({}) is None
    assert prewarm_abort_reason({"stop": True}) == "stop"
    assert prewarm_abort_reason({"go_passed": True}) == "prewarm_late"


def test_gpu_handoff_waits_for_explicit_gatenet_release():
    state = {"gatenet_unloaded": False}
    polls = []

    def release_after_one_poll(_seconds):
        polls.append(1)
        state["gatenet_unloaded"] = True

    assert wait_for_gpu_handoff(state, sleep=release_after_one_poll) == "handoff"
    assert polls == [1]


def test_gpu_handoff_exits_when_flight_stops():
    assert wait_for_gpu_handoff({"stop": True}, sleep=lambda _: None) == "stop"
```

- [ ] **Step 2: Run the tests and verify the missing-interface failure**

Run:

```powershell
python -B -m pytest vq2/tests/test_dpvo_prewarm.py -v
```

Expected: collection fails because `prewarm_abort_reason` and
`wait_for_gpu_handoff` do not exist.

- [ ] **Step 3: Implement the pure lifecycle gates**

Add below the configuration constants in `dpvo_odom_bridge.py`:

```python
def prewarm_abort_reason(state: dict) -> str | None:
    if state.get("stop"):
        return "stop"
    if state.get("go_passed"):
        return "prewarm_late"
    return None


def wait_for_gpu_handoff(
    state: dict,
    sleep=time.sleep,
    poll_s: float = 0.02,
) -> str:
    while True:
        if state.get("stop"):
            return "stop"
        if state.get("gatenet_unloaded"):
            return "handoff"
        sleep(float(poll_s))
```

Export both names in `__all__`.

- [ ] **Step 4: Run the focused tests**

Run:

```powershell
python -B -m pytest vq2/tests/test_dpvo_prewarm.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Record a dirty-tree checkpoint**

Run:

```powershell
git status --short -- vq2/live/dpvo_odom_bridge.py vq2/tests/test_dpvo_prewarm.py
git diff -- vq2/live/dpvo_odom_bridge.py vq2/tests/test_dpvo_prewarm.py
```

Expected: only the lifecycle helpers and their tests are new in this task.
Do not stage yet because `dpvo_odom_bridge.py` predates this refinement as an
uncommitted campaign file.

### Task 2: Stage `DpvoOdom` prewarm before tracking

**Files:**
- Modify: `vq2/live/dpvo_odom_bridge.py`
- Modify: `vq2/tests/test_dpvo_prewarm.py`

**Interfaces:**
- Consumes: `prewarm_abort_reason`, `wait_for_gpu_handoff`
- Produces: `DpvoOdom._prepare()`, `DpvoOdom._prewarm() -> bool`, and `DpvoOdom._track()`
- Publishes: `state['dpvo_prewarm_ready']`, `state['dpvo_prewarm_ready_wall']`
- Logs: `dpvo_prewarm_start`, `dpvo_ready`, `dpvo_tracking_start`, `dpvo_first_pose`, `dpvo_tick_align`

- [ ] **Step 1: Write failing staged-order and deadline tests**

Append to `test_dpvo_prewarm.py`:

```python
import numpy as np

from vq2.live.dpvo_odom_bridge import DpvoOdom


def bare_odom(state=None):
    return DpvoOdom(state or {}, object(), object(), np.eye(3))


def test_run_orders_prepare_prewarm_handoff_track(monkeypatch):
    odom = bare_odom()
    calls = []
    monkeypatch.setattr(odom, "_prepare", lambda: calls.append("prepare"))
    monkeypatch.setattr(odom, "_prewarm", lambda: calls.append("prewarm") or True)
    monkeypatch.setattr(
        "vq2.live.dpvo_odom_bridge.wait_for_gpu_handoff",
        lambda state: calls.append("handoff") or "handoff",
    )
    monkeypatch.setattr(odom, "_track", lambda: calls.append("track"))
    odom._run()
    assert calls == ["prepare", "prewarm", "handoff", "track"]


def test_prewarm_late_latches_health_without_connect(monkeypatch):
    state = {"go_passed": True}
    odom = bare_odom(state)
    monkeypatch.setattr(odom, "_connect", lambda **kwargs: (_ for _ in ()).throw(
        AssertionError("late prewarm must not connect")
    ))
    assert not odom._prewarm()
    assert state["dpvo_route_healthy"] is False
    assert state["dpvo_route_reason"] == "prewarm_late"


def test_run_never_tracks_when_handoff_stops(monkeypatch):
    odom = bare_odom({"stop": True})
    monkeypatch.setattr(odom, "_prepare", lambda: None)
    monkeypatch.setattr(odom, "_prewarm", lambda: True)
    monkeypatch.setattr(
        "vq2.live.dpvo_odom_bridge.wait_for_gpu_handoff",
        lambda state: "stop",
    )
    monkeypatch.setattr(odom, "_track", lambda: (_ for _ in ()).throw(
        AssertionError("tracking must not start")
    ))
    odom._run()
```

- [ ] **Step 2: Run the staged tests and verify failure**

Run:

```powershell
python -B -m pytest vq2/tests/test_dpvo_prewarm.py -v
```

Expected: failures because `_prepare`, `_prewarm`, and `_track` do not exist
and `_run` still waits for GO before connecting.

- [ ] **Step 3: Add failure latching and split configuration from tracking**

Add these methods to `DpvoOdom`:

```python
    def _latch_failure(self, reason: str) -> None:
        self.state["dpvo_route_healthy"] = False
        self.state["dpvo_route_reason"] = str(reason)

    def _prepare(self) -> None:
        self.config = config_from_env()
        self.model_sha256 = hash_file(MODEL)
        calibration_path = os.environ.get("DPVO_CAL")
        if not calibration_path:
            raise ValueError("DPVO_CAL is required for tick-rebased route odometry")
        calibration = load_calibration(
            calibration_path, self.config, self.model_sha256
        )
        ensure_control_approved(
            calibration, os.environ.get("DPVO_OBSERVE") == "1"
        )
        self.route = TickRebasedRoute(
            scale=calibration["scale"],
            gate1_world=calibration["gate1_world"],
            max_speed=float(os.environ.get("DPVO_MAX_SPEED", "8.0")),
            camera_alignment=self.m_body_cam,
        )

    def _prewarm(self) -> bool:
        started = time.time()
        self.prewarm_start_wall = started
        self.jlog(
            "dpvo_prewarm_start",
            race_ms=int(self.state.get("race_ms", 0)),
            go_passed=bool(self.state.get("go_passed")),
        )
        reason = prewarm_abort_reason(self.state)
        if reason is not None:
            self._latch_failure(reason)
            self.jlog("dpvo_prewarm_failed", reason=reason)
            return False
        if not self._connect(stop_at_go=True):
            return False
        self.jlog(
            "dpvo_prewarm_ready",
            ready_wall=self.ready_wall,
            init_ms=round((self.ready_wall - started) * 1000.0, 1),
        )
        return True
```

Initialize `self.ready_wall = None` and `self.prewarm_start_wall = None` in `__init__`.

- [ ] **Step 4: Make connection attempts obey the GO deadline**

Change `_connect` to return `bool` and accept `stop_at_go=False`. Before every
attempt and after the ready reply, check `prewarm_abort_reason`. Use a 1-second
socket timeout during prewarm and restore the 15-second tracking timeout after
the handshake:

```python
    def _connect(self, stop_at_go: bool = False) -> bool:
        host = _wsl_ip()
        error = None
        for _ in range(30):
            if stop_at_go:
                reason = prewarm_abort_reason(self.state)
                if reason is not None:
                    self._latch_failure(reason)
                    self.jlog("dpvo_prewarm_failed", reason=reason)
                    return False
            sock = None
            try:
                sock = socket.socket()
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                sock.settimeout(1.0)
                sock.connect((host, PORT))
                sock.sendall(
                    encode_packet(session_message(self.config, self.model_sha256))
                )
                sock.settimeout(0.25 if stop_at_go else 15.0)
                while True:
                    try:
                        payload = recv_packet(sock)
                        break
                    except socket.timeout:
                        reason = prewarm_abort_reason(self.state)
                        if stop_at_go and reason is not None:
                            sock.close()
                            self._latch_failure(reason)
                            self.jlog("dpvo_prewarm_failed", reason=reason)
                            return False
                if payload is None:
                    raise ConnectionError("DPVO service closed during handshake")
                reply = parse_json_message(payload)
                validate_ready(reply, self.config, self.model_sha256)
                reason = prewarm_abort_reason(self.state)
                if stop_at_go and reason is not None:
                    sock.close()
                    self._latch_failure(reason)
                    self.jlog("dpvo_prewarm_failed", reason=reason)
                    return False
                ready_wall = time.time()
                self.ready_wall = ready_wall
                self.state["dpvo_prewarm_ready"] = True
                self.state["dpvo_prewarm_ready_wall"] = ready_wall
                sock.settimeout(15.0)
                self.sock = sock
                self.jlog(
                    "dpvo_ready",
                    identity=reply["identity"],
                    patches=self.config.patches,
                    intrinsics=list(self.config.intrinsics),
                    ready_wall=ready_wall,
                    init_ms=None if self.prewarm_start_wall is None else round(
                        (ready_wall - self.prewarm_start_wall) * 1000.0, 1
                    ),
                    vram_alloc_mb=reply.get("vram_alloc_mb"),
                    vram_reserved_mb=reply.get("vram_reserved_mb"),
                    cuda_free_mb=reply.get("cuda_free_mb"),
                )
                print(
                    f"DPVO bridge ready {host}:{PORT}, "
                    f"patches={self.config.patches}",
                    flush=True,
                )
                return True
            except (OSError, ValueError) as caught:
                error = caught
                if sock is not None:
                    try:
                        sock.close()
                    except OSError:
                        pass
                time.sleep(0.1 if stop_at_go else 1.0)
        raise ConnectionError(f"DPVO bridge unavailable: {error!r}")
```

- [ ] **Step 5: Split the existing frame loop into `_track` and add telemetry**

Move the current post-connect loop into `_track` without changing frame
selection, stride, route math, or health limits. Add one-shot first-pose and
tick-alignment logging:

```python
    def _track(self) -> None:
        state = self.state
        last_ns = 0
        submitted = 0
        first_pose_logged = False
        alignment_logged = False
        pose_history = deque(maxlen=300)
        while not state.get("stop"):
            jpeg = state.get("jpeg")
            frame_ns = int(state.get("frame_ns", 0))
            if jpeg is None or frame_ns == last_ns:
                time.sleep(0.004)
                continue
            last_ns = frame_ns
            submitted += 1
            if submitted % self.config.stride:
                continue
            reply = self._step(frame_ns, jpeg)
            if not reply.get("ok"):
                continue
            raw_p = tuple(float(value) for value in reply["p"])
            yaw = float(state.get("yaw", 0.0))
            pose_history.append((frame_ns, raw_p, yaw))
            if not first_pose_logged:
                first_pose_logged = True
                wall = time.time()
                self.jlog(
                    "dpvo_first_pose",
                    frame_ns=frame_ns,
                    ready_to_pose_ms=None if self.ready_wall is None else round(
                        (wall - self.ready_wall) * 1000.0, 1
                    ),
                )

            gate_idx = int(state.get("gate_idx", 0))
            tick_ns = int(state.get("gate_tick_ns", 0))
            if gate_idx >= 1 and self.route.raw_origin is None and not alignment_logged:
                alignment_logged = True
                if tick_ns > 0:
                    sample = min(pose_history, key=lambda entry: abs(entry[0] - tick_ns))
                    offset_ms = (sample[0] - tick_ns) / 1e6
                    decision = "aligned" if abs(offset_ms) <= 150.0 else "tick_alignment"
                    pose_ns = sample[0]
                else:
                    offset_ms = None
                    decision = "tick_timestamp"
                    pose_ns = None
                self.jlog(
                    "dpvo_tick_align",
                    tick_ns=tick_ns,
                    pose_ns=pose_ns,
                    offset_ms=offset_ms,
                    decision=decision,
                )

            observation = observe_route_pose(
                self.route,
                pose_history,
                raw_p,
                frame_ns,
                gate_idx,
                yaw,
                tick_ns,
            )
            publish_route_observation(state, observation, time.time())
            self.jlog(
                "dpvo_route",
                ok=observation.healthy,
                reason=observation.reason,
                frame_ns=frame_ns,
                p=None if observation.p is None else list(observation.p),
                raw=list(raw_p),
                dt_ms=reply.get("dt_ms"),
                vram_alloc_mb=reply.get("vram_alloc_mb"),
                vram_reserved_mb=reply.get("vram_reserved_mb"),
                cuda_free_mb=reply.get("cuda_free_mb"),
            )
```

- [ ] **Step 6: Replace `_run` with the staged sequence**

```python
    def _run(self) -> None:
        self._prepare()
        if not self._prewarm():
            return
        print("DPVO route thread waiting for GateNet GPU handoff", flush=True)
        outcome = wait_for_gpu_handoff(self.state)
        if outcome != "handoff":
            return
        tracking_wall = time.time()
        self.jlog(
            "dpvo_tracking_start",
            handoff_delay_ms=round(
                (tracking_wall - float(self.state["gatenet_unloaded_wall"]))
                * 1000.0,
                1,
            ),
        )
        self._track()
```

In `run()`, reuse `_latch_failure("service")` in the exception path.

- [ ] **Step 7: Run client and lifecycle tests**

Run:

```powershell
python -B -m pytest vq2/tests/test_dpvo_prewarm.py vq2/tests/test_dpvo_bridge_client.py vq2/tests/test_dpvo_tick_alignment.py vq2/tests/test_dpvo_bridge_thread_safety.py -v
```

Expected: all tests pass without importing CUDA.

- [ ] **Step 8: Record a task checkpoint without staging**

Run:

```powershell
git diff --check -- vq2/live/dpvo_odom_bridge.py vq2/tests/test_dpvo_prewarm.py
git status --short -- vq2/live/dpvo_odom_bridge.py vq2/tests/test_dpvo_prewarm.py
```

Expected: no whitespace errors. Preserve the uncommitted campaign boundary.

### Task 3: Wire early startup and authoritative GateNet handoff

**Files:**
- Modify: `vq2/live/vq2wp.py`
- Modify: `vq2/tests/test_vq2wp_dpvo_route_wiring.py`

**Interfaces:**
- Produces: `state['gatenet_unloaded']`, `state['gatenet_unloaded_wall']`
- Starts route `DpvoOdom` immediately after the hard reset and before its six-second settle
- Keeps legacy/non-route DPVO startup after arming

- [ ] **Step 1: Write failing source-wiring tests**

Append:

```python
def test_route_dpvo_prewarm_starts_during_hard_reset_settle():
    source = (ROOT / "vq2/live/vq2wp.py").read_text(encoding="utf-8")
    reset = source.index("31000, 0, 1")
    started = source.index("_dpvo_route_started = True")
    reset_settle = source.index("time.sleep(6.0)", reset)
    assert reset < started < reset_settle


def test_gatenet_handoff_is_explicit_and_timestamped():
    source = (ROOT / "vq2/live/vq2wp.py").read_text(encoding="utf-8")
    assert "state['gatenet_unloaded'] = True" in source
    assert "state['gatenet_unloaded_wall'] = _handoff_wall" in source
    assert "jlog('gatenet_unloaded'" in source


def test_go_wall_is_published_before_go_flag():
    source = (ROOT / "vq2/live/vq2wp.py").read_text(encoding="utf-8")
    assert source.index("state['go_wall'] = time.time()") < source.index(
        "state['go_passed'] = True"
    )
```

- [ ] **Step 2: Run and verify wiring failures**

Run:

```powershell
python -B -m pytest vq2/tests/test_vq2wp_dpvo_route_wiring.py -v
```

Expected: the three new assertions fail.

- [ ] **Step 3: Publish GateNet handoff after memory release**

Immediately after `torch.cuda.empty_cache()` in the existing unload block:

```python
                    _handoff_wall = time.time()
                    state['gatenet_unloaded'] = True
                    state['gatenet_unloaded_wall'] = _handoff_wall
                    jlog(
                        'gatenet_unloaded',
                        go_delay_ms=round(
                            (_handoff_wall - state.get('go_wall', _handoff_wall))
                            * 1000.0,
                            1,
                        ),
                    )
```

Initialize `gatenet_unloaded` to `False` in the main `state` dictionary.

- [ ] **Step 4: Start only route DPVO during the hard-reset settle window**

Insert immediately after the `31000 param1=1` hard-reset command and before
the existing six-second reset sleep. This provides the reset settle and pad
acquisition windows for model loading before GO:

```python
select_dpvo_route_position = None
_dpvo_route_started = False
if (os.environ.get('DPVO') == '1'
        and os.environ.get('DPVO_ROUTE') == '1'):
    if os.environ.get('DPVO_BRIDGE') != '1':
        raise RuntimeError('DPVO_ROUTE requires the low-memory WSL2 bridge')
    from dpvo_odom_bridge import DpvoOdom
    from dpvo_route import select_route_position
    select_dpvo_route_position = select_route_position
    print('DPVO: prewarming independent WSL2 route bridge', flush=True)
    DpvoOdom(state, KF, KF_LOCK, M_BODY_CAM, jlog=jlog).start()
    _dpvo_route_started = True
    print('DPVO route prewarm thread started', flush=True)
```

- [ ] **Step 5: Preserve legacy DPVO startup after arming**

Replace the existing post-arm DPVO block with:

```python
state['go_wall'] = time.time()
state['go_passed'] = True
m.mav.command_long_send(m.target_system, m.target_component,
    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0)
time.sleep(0.5)
print('armed', flush=True)

if os.environ.get('DPVO') == '1' and not _dpvo_route_started:
    if os.environ.get('DPVO_BRIDGE') == '1':
        from dpvo_odom_bridge import DpvoOdom
        print('DPVO: using WSL2 bridge', flush=True)
    else:
        from dpvo_odom import DpvoOdom
    DpvoOdom(state, KF, KF_LOCK, M_BODY_CAM, jlog=jlog).start()
    print('DPVO odometry thread started', flush=True)
```

- [ ] **Step 6: Run wiring tests and parse the flight script**

Run:

```powershell
python -B -m pytest vq2/tests/test_vq2wp_dpvo_route_wiring.py -v
python -B -c "import ast; ast.parse(open('vq2/live/vq2wp.py', encoding='utf-8').read())"
```

Expected: tests pass and AST parse exits 0.

- [ ] **Step 7: Review only the new integration hunks**

Run:

```powershell
git diff --check -- vq2/live/vq2wp.py vq2/tests/test_vq2wp_dpvo_route_wiring.py
git diff -U5 -- vq2/live/vq2wp.py vq2/tests/test_vq2wp_dpvo_route_wiring.py
```

Expected: new hunks are limited to state initialization, GateNet handoff,
route prewarm startup, GO ordering, and tests. Do not stage `vq2wp.py`; it
contains pre-existing campaign changes.

### Task 4: Prepare a fresh observe-only batch

**Files:**
- Modify: `vq2/live/fly_servo_dpvo_observe.bat`
- Modify: `vq2/tests/test_dpvo_observe_batch.py`

**Interfaces:**
- Produces: fresh recording `C:/Users/alexj/vq2_servo_dpvo_obs2`
- Preserves: `DPVO_OBSERVE=1`, 32 patches, stride 2, full resolution, MF_DR

- [ ] **Step 1: Write failing batch assertions**

Add:

```python
    assert "set RECORD=C:/Users/alexj/vq2_servo_dpvo_obs2" in source
    assert "DPVO_GPU_SETTLE" not in source
```

- [ ] **Step 2: Run and verify failure**

Run:

```powershell
python -B -m pytest vq2/tests/test_dpvo_observe_batch.py -v
```

Expected: failure because the batch still targets `obs1` and sets the obsolete
fixed settle delay.

- [ ] **Step 3: Update only the lifecycle-related batch lines**

Remove:

```bat
set DPVO_GPU_SETTLE=1.0
```

Replace the record line with:

```bat
set RECORD=C:/Users/alexj/vq2_servo_dpvo_obs2
```

- [ ] **Step 4: Run the batch test**

Run:

```powershell
python -B -m pytest vq2/tests/test_dpvo_observe_batch.py -v
```

Expected: pass.

- [ ] **Step 5: Confirm the record target is absent**

Run:

```powershell
if (Test-Path 'C:\Users\alexj\vq2_servo_dpvo_obs2') {
    throw 'record target already exists'
} else {
    'fresh target absent'
}
```

Expected: `fresh target absent`. Never delete or reuse a corpus.

### Task 5: Full verification and in-place deployment

**Files:**
- Deploy in place: `C:\Users\alexj\dpvo_odom_bridge.py`
- Deploy in place: `C:\Users\alexj\vq2wp.py`
- Deploy in place: `C:\Users\alexj\fly_servo_dpvo_observe.bat`

**Interfaces:**
- Consumes: verified repo source only
- Produces: hash-identical deployed modules and fresh observe-only batch

- [ ] **Step 1: Run all DPVO tests**

Run:

```powershell
python -B -m pytest vq2/tests -k "dpvo or vq2wp_dpvo_route_wiring" -v
```

Expected: all DPVO tests pass.

- [ ] **Step 2: Run the complete VQ2 suite**

Run:

```powershell
python -B -m pytest vq2/tests -q
```

Expected: zero failures.

- [ ] **Step 3: Parse every repo Python file**

Run:

```powershell
$bad = @()
Get-ChildItem vq2 -Recurse -Filter '*.py' | ForEach-Object {
    python -B -c "import ast; ast.parse(open(r'$($_.FullName)', encoding='utf-8').read())"
    if ($LASTEXITCODE) { $bad += $_.FullName }
}
if ($bad) { throw "parse failures: $bad" }
```

Expected: no parse failures.

- [ ] **Step 4: Check disk space before deployment**

Run:

```powershell
$d = [System.IO.DriveInfo]::new('C:\')
"C free GB: $([Math]::Round($d.AvailableFreeSpace / 1GB, 1))"
wsl -d Ubuntu df -h / /mnt/c
```

Expected: both filesystems have more than 20 GB free. This step creates no
cache or artifact.

- [ ] **Step 5: Copy verified files in place**

Run with the required local-filesystem approval:

```powershell
Copy-Item -LiteralPath 'vq2\live\dpvo_odom_bridge.py' -Destination 'C:\Users\alexj\dpvo_odom_bridge.py' -Force
Copy-Item -LiteralPath 'vq2\live\vq2wp.py' -Destination 'C:\Users\alexj\vq2wp.py' -Force
Copy-Item -LiteralPath 'vq2\live\fly_servo_dpvo_observe.bat' -Destination 'C:\Users\alexj\fly_servo_dpvo_observe.bat' -Force
```

- [ ] **Step 6: Verify deployed syntax and hashes**

Run:

```powershell
$pairs = @(
    @('vq2\live\dpvo_odom_bridge.py', 'C:\Users\alexj\dpvo_odom_bridge.py'),
    @('vq2\live\vq2wp.py', 'C:\Users\alexj\vq2wp.py'),
    @('vq2\live\fly_servo_dpvo_observe.bat', 'C:\Users\alexj\fly_servo_dpvo_observe.bat')
)
foreach ($pair in $pairs) {
    $a = (Get-FileHash -Algorithm SHA256 $pair[0]).Hash
    $b = (Get-FileHash -Algorithm SHA256 $pair[1]).Hash
    if ($a -ne $b) { throw "hash mismatch: $($pair[0])" }
}
python -B -c "import ast; ast.parse(open(r'C:\Users\alexj\dpvo_odom_bridge.py', encoding='utf-8').read()); ast.parse(open(r'C:\Users\alexj\vq2wp.py', encoding='utf-8').read())"
```

Expected: matching hashes and parse exit 0.

- [ ] **Step 7: Preserve the dirty-tree boundary**

Run:

```powershell
git status --short
git diff --check
git diff --stat
```

Expected: no whitespace errors. Do not commit or stage the implementation
because its source files overlap pre-existing uncommitted campaign work.

### Task 6: Fresh live acceptance flight

**Files:**
- Record: `C:\Users\alexj\vq2_servo_dpvo_obs2`
- Log: `C:\Users\alexj\dpvo_observe_flight.log`
- Service log: `C:\Users\alexj\dpvo_observe_service.log`

**Interfaces:**
- Consumes: fresh judge session, deployed observe-only batch, local WSL service
- Produces: one scored corpus and explicit alignment verdict

- [ ] **Step 1: Verify the simulator and fresh target**

Run:

```powershell
Get-Process FlightSim,DCGame-Win64-Shipping -ErrorAction Stop
if (Test-Path 'C:\Users\alexj\vq2_servo_dpvo_obs2') {
    throw 'record target already exists'
}
```

Expected: simulator processes exist and target is absent. If the simulator is
not running, stop here and report implementation ready rather than launching
it without the user.

- [ ] **Step 2: Perform the screenshot-verified menu restart**

Use `simctl.ps1` to focus `AI-GP`, capture the current state, press `ESC`,
verify `RESUME` is selected with `RESTART` below it, then send `DOWN`, `ENTER`.
Wait five seconds and verify the green-light pad scene.

- [ ] **Step 3: Verify race-ready spawn attitude**

Run:

```powershell
& 'C:\Users\alexj\miniconda3\envs\monorace\python.exe' 'C:\Users\alexj\spawn_att_probe.py'
```

Expected: `NOSE-DOWN (race-ready, gate visible)` near -17.8 degrees.

- [ ] **Step 4: Start and verify the WSL bridge**

Start hidden with stdout/stderr redirected to the existing service logs:

```powershell
Start-Process -FilePath 'wsl.exe' `
    -ArgumentList @('-d','Ubuntu','bash','/mnt/c/Users/alexj/start_bridge_service.sh') `
    -WindowStyle Hidden `
    -RedirectStandardOutput 'C:\Users\alexj\dpvo_observe_service.log' `
    -RedirectStandardError 'C:\Users\alexj\dpvo_observe_service.err.log'
```

Expected log: `DPVO bridge listening on 0.0.0.0:9099`.

- [ ] **Step 5: Launch the detached observe-only batch**

```powershell
Start-Process -FilePath 'cmd.exe' `
    -ArgumentList @('/c','C:\Users\alexj\fly_servo_dpvo_observe.bat') `
    -WorkingDirectory 'C:\Users\alexj' `
    -WindowStyle Hidden `
    -RedirectStandardOutput 'C:\Users\alexj\dpvo_observe_flight.log' `
    -RedirectStandardError 'C:\Users\alexj\dpvo_observe_flight.err.log'
```

Monitor logs and simulator frames until the flight exits. Do not send commands
during the countdown.

- [ ] **Step 6: Score only with `gidx2.py`**

```powershell
& 'C:\Users\alexj\miniconda3\envs\monorace\python.exe' 'C:\Users\alexj\gidx2.py' 'vq2_servo_dpvo_obs2'
```

Expected for a valid alignment trial: one fresh post-reset `0 -> 1`
transition. No transition makes the run invalid, not an alignment failure.

- [ ] **Step 7: Evaluate the hard alignment gate**

Parse `livelog.jsonl` and require all of:

- `dpvo_ready` timestamp before GO.
- `dpvo_tick_align.decision == "aligned"` and `abs(offset_ms) <= 150`.
- One healthy `dpvo_route` with `reason == "rebase"` within one wall second of
  `gate_tick`.
- At least ten later healthy `dpvo_route` rows with `reason == "ok"`.
- Minimum `cuda_free_mb >= 750`.
- No `dpvo_died`, `service`, `speed`, `timestamp`, or `nonfinite` latch.
- Flight log and batch confirm observe-only MF_DR control.

Classify the run exactly as PASS, FAIL, or INVALID using the approved spec.

- [ ] **Step 8: Cleanup and log**

Stop only the DPVO service started by this task, verify port 9099 is free, and
menu-restart the simulator after any collision. Append the experiment result
to the Mac Obsidian vault when reachable; otherwise append it to
`C:\Users\alexj\obsidian_outbox.md`. Retain the single corpus and compact logs;
do not duplicate them.

---

## Final handoff

Report separately:

1. Automated verification result and deployed hashes.
2. Judge result from `gidx2.py`.
3. Tick offset, rebase wall delay, consecutive healthy poses, latency, and GPU
   headroom.
4. PASS, FAIL, or INVALID alignment classification.
5. Confirmation that DPVO control remains disabled.
