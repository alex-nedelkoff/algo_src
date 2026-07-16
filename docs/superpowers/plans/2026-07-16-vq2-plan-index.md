# VQ2 Authoritative Plan Index Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give every implementation worker one authoritative entry point, precedence order, and phase schedule for the complete autonomous survey, metric map, prior-map localization, and race-control program.

**Architecture:** Four base plans describe subsystem construction. Contract and review plans override specific unsafe or inconsistent details. This index is the sole authority for precedence and directs workers to apply all relevant overrides at each insertion point.

**Tech Stack:** The stack declared by the referenced plans; no new dependency.

## Sole Precedence Order

This index supersedes every earlier statement about which plan is “highest” or
“final.” For conflicting instructions, use the first applicable document in
this list:

1. `2026-07-16-vq2-plan-index.md`
2. `2026-07-16-vq2-survey-state-machine-closure.md`
3. `2026-07-16-vq2-final-review-closure.md`
4. `2026-07-16-vq2-master-execution-order.md`
5. `2026-07-16-vq2-survey-runtime-corpus-addendum.md`
6. `2026-07-16-vq2-final-safety-mapping-corrections.md`
7. `2026-07-16-vq2-pose-bootstrap-identity-addendum.md`
8. `2026-07-16-vq2-cross-plan-interface-contract.md`
9. `2026-07-16-vq2-end-to-end-plan-self-review-fixes.md`
10. The applicable base plan:
    - `2026-07-16-vq2-autonomous-survey-acquisition.md`
    - `2026-07-16-vq2-offline-metric-map.md`
    - `2026-07-16-vq2-prior-map-localization.md`
    - `2026-07-16-vq2-race-map-integration.md`

An override replaces only the conflicting interface/example. Unrelated base
tasks remain required. CLI angle-bracket tokens denote command metavariables,
not unfinished design; live attempt-001 commands use the concrete paths in the
self-review-fixes companion.

---

### Task 1: Execute the complete plan set in gated order

**Files:**
- Read and check off the referenced plan documents.
- Modify only the source/test files named by the current task.
- Log experiments to the configured Obsidian vault or local outbox.

- [ ] **Step 1: Survey software and offline proof**

Implement the survey base plan while inserting, in precedence order, the
manifest contract, safety/mapping corrections Tasks 2-3, runtime/corpus
addendum Tasks 1-3, final-review closure Tasks 1-5, and survey-state-machine
closure Tasks 1-3. Run unit, synthetic bypass, complete-state replay,
recorder-backpressure, no-arm, abort, and fake-MAVLink lifecycle tests.

- [ ] **Step 2: Survey live ladder and A collection**

Run no-arm soak, arm-hover/abort, takeoff-stare-land, one-segment, and
two-segment checks. Then collect verified A1 and A2, derive their immutable
coverage artifact, and collect measured coverage-directed A3. Keep DPVO off
and use unique fresh-sim attempts.

- [ ] **Step 3: Metric map construction**

Implement the offline-map base plan with corrected gate/gauge Task 1, connected
mapping Task 4, pose/artifact contracts, and master gate/input Task 1. Run
GateNet then DPVO sequentially with the sim closed, pass the synthetic
three-run map test, and freeze map A plus A-only policy.

- [ ] **Step 4: B1 collection and localization approval**

Only after map/policy freeze, collect B1 using the survey stack without map
control and seal its entire content against that identity. Implement the
prior-map base plan with the pose-bootstrap, covariance/propagation, artifact,
and approval-chain overrides. Open B1 exactly once and issue approval only on
the frozen criteria.

- [ ] **Step 5: Observe-only race integration**

Implement race Tasks 1-5 with identity-per-cycle checks, bounded correction,
owned failsafe, and stable-config/session-epoch separation. Run full replay and
fresh observe-only flight; prove legacy control noninterference and generate
immutable shadow evidence.

- [ ] **Step 6: Approved control and full-course promotion**

Verify the full approval chain, then enable map control at one gate, two gates,
and full course. Keep DPVO resident normally; use CPU map localization and
FastGate; allow GateNet only in stopped-hover recovery followed by a measured
DPVO rebootstrap. Score only with `gidx2` and land on any failed gate.

- [ ] **Step 7: Completion evidence**

Require all automated suites, corpus/map/approval digests, disk/GPU telemetry,
fresh judge evidence, experiment logs, and a reproducible full-course traversal
with no unresolved safety, identity, association, or recovery exception.

---

## Index completion gate

No implementation worker may skip an applicable higher-precedence correction
or advance past a failed phase. The work is complete only at Step 7, not when a
subsystem builds or an observe-only run looks plausible.
