# VQ2 Master Execution Order Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide one unambiguous execution order for the VQ2 survey-to-map-to-race program and freeze the last shared map-gate and approval-chain contracts.

**Architecture:** The detailed plans remain the implementation source, while this master plan defines precedence, phase boundaries, and the artifact identities handed from one phase to the next. Map gates expose one canonical gate-to-map pose; approvals form a verified hash chain from A corpora through B1 and observe-only shadow evidence to control.

**Tech Stack:** The Python, NumPy, OpenCV, SciPy, pymavlink, GateNet, DPVO bridge, SHA-256, and pytest stack declared by the detailed plans; no new dependency.

## Plan Precedence

When instructions conflict, use this order from highest to lowest:

1. `2026-07-16-vq2-master-execution-order.md`
2. `2026-07-16-vq2-survey-runtime-corpus-addendum.md`
3. `2026-07-16-vq2-final-safety-mapping-corrections.md`
4. `2026-07-16-vq2-pose-bootstrap-identity-addendum.md`
5. `2026-07-16-vq2-cross-plan-interface-contract.md`
6. `2026-07-16-vq2-end-to-end-plan-self-review-fixes.md`
7. The four base survey, mapping, localization, and race plans.

The addenda replace conflicting snippets; they are not optional follow-up
work. Each implementation task starts with a failing focused test, makes the
minimum change, runs its focused and regression suites, and commits only its
listed files.

---

### Task 1: Freeze gate pose, route, and A-input manifest contracts

**Files:**
- Modify: `vq2/mapping/schema.py`
- Modify: `vq2/mapping/artifact.py`
- Modify: `vq2/live/map_route.py`
- Test: `vq2/tests/test_cross_plan_gate_contract.py`
- Modify: `vq2/tests/test_mapping_artifact.py`
- Modify: `vq2/tests/test_map_route_live.py`

**Interfaces:**
- `MapGate.T_map_gate` is gate-local-to-map SE3 with translation at the band
  midplane center and quaternion `(x,y,z,w)` rotating gate-local axes
  `right,up,normal` into map axes.
- `center_m`, `normal_map`, and world corners are derived from that pose; no
  consumer assumes an undocumented pose layout.
- Map manifest `input_manifests` is keyed by exactly `A1`, `A2`, and `A3`,
  each containing both `manifest_sha256` and `content_sha256`.

- [ ] **Step 1: Write pose-derived route tests**

```python
def test_map_gate_center_normal_and_corners_share_one_pose():
    gate = MapGate(
        stable_gate_id="G3", route_order=2,
        T_map_gate=PoseSE3(0, (1., 2., 3.),
                           tuple(Rotation.from_euler("y", 90, degrees=True)
                                 .as_quat())),
        covariance_6x6=np.eye(6))
    assert np.allclose(gate.center_m, [1., 2., 3.])
    expected_normal = Rotation.from_quat(gate.T_map_gate.q_xyzw).apply([0, 0, 1])
    assert np.allclose(gate.normal_map, expected_normal)
    assert gate.corners_map.shape == (8, 3)
```

Add a route-loader test proving it consumes these properties, preserves
`route_order`, and never sorts by center x. Reject duplicate route orders,
nonunit normals, nonfinite covariance, or a geometry-digest mismatch.

- [ ] **Step 2: Implement derived gate properties**

```python
@property
def center_m(self):
    return np.asarray(self.T_map_gate.p, float)

@property
def normal_map(self):
    return Rotation.from_quat(self.T_map_gate.q_xyzw).apply([0., 0., 1.])

@property
def corners_map(self):
    rotation = Rotation.from_quat(self.T_map_gate.q_xyzw)
    return rotation.apply(GATE_CORNERS_8) + self.center_m
```

If center/normal are serialized for fast loading, verification must recompute
and compare them to `T_map_gate`; the pose remains authoritative.

- [ ] **Step 3: Key all A corpus identities**

The map writer emits:

```json
{
  "input_manifests": {
    "A1": {"manifest_sha256": "...", "content_sha256": "..."},
    "A2": {"manifest_sha256": "...", "content_sha256": "..."},
    "A3": {"manifest_sha256": "...", "content_sha256": "..."}
  }
}
```

Reject missing, duplicate, extra, reordered-as-list, or B1 entries. Recompute
both hashes before a map build and when verifying the artifact.

- [ ] **Step 4: Run gate, route, and artifact tests**

Run: `python -m pytest vq2/tests/test_cross_plan_gate_contract.py vq2/tests/test_mapping_artifact.py vq2/tests/test_map_route_live.py -v`

Expected: all tests PASS.

- [ ] **Step 5: Commit the final map schema contract**

```bash
git add vq2/mapping/schema.py vq2/mapping/artifact.py vq2/live/map_route.py vq2/tests/test_cross_plan_gate_contract.py vq2/tests/test_mapping_artifact.py vq2/tests/test_map_route_live.py
git commit -m "fix(vq2): freeze gate and input identities"
```

### Task 2: Freeze the full validation and control approval chain

**Files:**
- Modify: `vq2/mapping/benchmark.py`
- Modify: `vq2/mapping/validate.py`
- Modify: `vq2/map_control_approve.py`
- Modify: `vq2/live/map_control.py`
- Test: `vq2/tests/test_cross_plan_approval_chain.py`
- Modify: `vq2/tests/test_map_benchmark.py`
- Modify: `vq2/tests/test_map_control_approval.py`
- Modify: `vq2/tests/test_map_control_live.py`

**Interfaces:**
- Localization approval binds verified map package, policy, B1 manifest,
  B1 content, camera, geometry, features, and stable DPVO config.
- Control approval binds the localization-approval file hash and the exact
  immutable observe-only shadow report/log hash.
- Live startup verifies every referenced file and digest before arming; source
  selection rechecks the resulting runtime identity against every state.

- [ ] **Step 1: Write one-mutation-per-link tests**

Construct a valid chain, then mutate each link independently:

```text
A1/A2/A3 content -> map package -> policy -> B1 content/seal
-> localization approval -> shadow log/report -> control approval
-> runtime camera/geometry/features/DPVO config -> DPVO session state
```

Each mutation must fail before B1 image read, approval generation, map query,
or control-source selection as appropriate. Include a control approval whose
embedded shadow digest is valid text but does not match the supplied immutable
shadow file.

- [ ] **Step 2: Emit complete localization approval**

Include `map_sha256`, `policy_sha256`, `b1_manifest_sha256`,
`b1_content_sha256`, `camera_id`, `geometry_id`, `feature_id`,
`dpvo_config_id`, frozen thresholds, raw report hash, and `approved`. The B1
open receipt and failed report remain immutable whether approval passes or
fails.

- [ ] **Step 3: Emit and verify complete control approval**

`map_control_approve` verifies the localization approval, reads the immutable
shadow log/report, hashes it, evaluates frozen noninterference/health metrics,
and writes `localization_approval_sha256`, `shadow_log_sha256`, thresholds,
and `approved`. Runtime receives explicit paths to the localization approval,
control approval, and shadow evidence; it recomputes all three hashes. A copied
approval without its matching evidence fails closed.

- [ ] **Step 4: Separate stable config from session epoch**

Approvals bind `dpvo_config_id` made from model, patches, resolution, bridge,
camera, and relevant settings. `dpvo_session_id` is live epoch state: it must
match each `MapPoseState`, and a change resets Sim3 and disables control, but a
restart with the same verified config does not require a new B1 approval.

- [ ] **Step 5: Run approval-chain tests**

Run: `python -m pytest vq2/tests/test_cross_plan_approval_chain.py vq2/tests/test_map_benchmark.py vq2/tests/test_map_control_approval.py vq2/tests/test_map_control_live.py -v`

Expected: all tests PASS with every mutation rejected at the earliest gate.

- [ ] **Step 6: Commit the approval chain**

```bash
git add vq2/mapping/benchmark.py vq2/mapping/validate.py vq2/map_control_approve.py vq2/live/map_control.py vq2/tests/test_cross_plan_approval_chain.py vq2/tests/test_map_benchmark.py vq2/tests/test_map_control_approval.py vq2/tests/test_map_control_live.py
git commit -m "fix(vq2): bind validation through control"
```

### Task 3: Execute the program in dependency order

**Files:**
- Update checkboxes in the applicable dated plans as work completes.
- Append experiment evidence to the configured Obsidian vault or
  `C:\Users\alexj\obsidian_outbox.md` when the Mac is unavailable.

- [ ] **Phase 0: Protect the branch and baseline**

Record branch, Git SHA, dirty paths, disk free, sim version, camera/calibration
IDs, GateNet checkpoint, DPVO config/model, and current passing tests. Preserve
the other agent's `vq2wp.py` and `dpvo_odom.py` changes. Do not mix refactors.

- [ ] **Phase 1: Build and verify the survey package offline**

Execute survey base Tasks 1-8 together with cross-plan manifest Task 1, final
safety Tasks 2-3, and survey-runtime addendum Tasks 1-3 in their stated
insertion order. Run synthetic bypass and fg62/fg75 replays. Then run the
no-arm soak, arm-hover/manual abort, and takeoff-stare-land ladder.

- [ ] **Phase 2: Collect A1 and A2**

Use fresh sim sessions, unique directories, DPVO off, left then right bypass,
full corpus recording, and one variable per attempt. Verify content digests,
clearance, recorder shutdown, race/reset consistency, and `gidx2` evidence.
Invalid, collided, unexpectedly ticked, or incomplete runs never enter mapping.

- [ ] **Phase 3: Derive coverage and collect A3**

Freeze exactly one valid A1 and A2 into the coverage artifact. Execute only
the declared missing views in A3, bind its coverage digest, and verify all
three A corpus identities. Do not collect B1 yet.

- [ ] **Phase 4: Build and freeze map A**

Execute gate/gauge Task 1, offline map base Tasks 1-8, final mapping Task 4,
master Task 1, and the full synthetic end-to-end test. Run GateNet and DPVO
sequentially with simulator closed. Freeze the external map digest and A-only
approval policy. Do not open or collect B1 during map construction.

- [ ] **Phase 5: Collect and seal B1**

Fly the same autonomous survey stack without map control, but require the
frozen map/policy digests in collection context. Verify and seal the complete
corpus content. This is the first B1 flight for that map/policy pair.

- [ ] **Phase 6: Build localization and open B1 once**

Execute prior-map base Tasks 1-7, pose-bootstrap Tasks 1-2, final localization
Task 5, and master Task 2. Freeze thresholds before creating the globally
exclusive B1 open receipt. Issue localization approval only if all frozen B1
criteria pass. A failure keeps its evidence and requires a new map/policy and
new B1 flight for another attempt.

- [ ] **Phase 7: Integrate race observe-only**

Execute race base Tasks 1-5 plus final safety Task 6 without enabling map
control. Verify legacy commands are byte/behavior equivalent, identities match
per frame, map health survives tick windows, no GPU overcommit occurs, and the
immutable shadow evidence passes the frozen control-approval tool.

- [ ] **Phase 8: Promote control in the fresh-sim ladder**

Load the exact approval chain and start at one gate, then two gates, then full
course. DPVO remains resident normally; CPU PnP and FastGate provide map and
terminal guidance; GateNet recovery is stopped-hover only and includes the
verified rebootstrap dither. Score only with `gidx2`; land on any identity,
freshness, uncertainty, route, collision, GPU, or failsafe violation.

- [ ] **Step 9: Declare completion only with full-course evidence**

Completion requires reproducible A corpora, frozen map and policies, one-shot
B1 approval, passing shadow/control approvals, a fresh judge-scored full-course
traversal, complete logs, and no unresolved safety or identity exception.

---

## Master completion gate

The program advances only through the ordered phases above. No later artifact
or live mode may compensate for a failed earlier gate, and no B1 or control
evidence may be reused after a bound map, policy, corpus, model/config, or
shadow artifact changes.
