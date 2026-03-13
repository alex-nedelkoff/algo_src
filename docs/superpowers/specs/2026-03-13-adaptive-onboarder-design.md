# Adaptive Experiment Onboarder with Framework Evolution

## Goal

Enhance the experiment onboarder skill to proactively detect when new components don't fit the current framework, propose minimal changes, and drive or escalate framework evolution — keeping the onboarder itself as a living document that evolves with the framework.

## Context

The current onboarder (`/.claude/skills/experiment-onboarder.md`) is a static checklist. It assumes every new environment fits the existing `EpisodeMetrics` schema and `TrajectoryProvider` protocol from `metrics/contract.py`. When a new component breaks these assumptions (e.g., a unified perception+control model, a non-quadrotor vehicle, a variable-motor-count platform), the onboarder has no path forward.

## Architecture

### 1. Framework Assumptions Section

Add a maintained list of framework assumptions directly in the onboarder skill. These are the invariants that current code depends on. Each assumption references the code that enforces it.

Example assumptions:
- Perception and control are separate modules (`perception/`, `control/`)
- All environments are quadrotor-based with 4 motors
- State includes quaternion orientation (w,x,y,z) — see `TrajectoryProvider.get_state()`
- `EpisodeMetrics` fields are mandatory for all envs; racing fields default to zero for non-racing
- Envs are internally vectorized (n_envs parameter, batched arrays)
- Actions are motor-level (RPM or rate commands, 4-dimensional)
- Hydra config composition: `configs/sim/`, `configs/experiment/`
- `TrajectoryProvider.get_state()` returns exactly: position (3,), quaternion (4,), velocity (3,), body_rates (3,), motor_rpms (4,)

### 2. Gap Detection Flow

Before jumping to the existing checklist, the onboarder asks probing questions about the new component:

1. **Component profile** — "Describe what this component does and what makes it different from existing ones."
2. **Assumption comparison** — Walk through each framework assumption and check alignment. Surface mismatches as named gaps.
3. **Gap classification** — For each gap:
   - **Back-compatible**: Can be solved by adding optional fields, new protocol methods, or config options without changing existing code or breaking existing envs.
   - **Breaking**: Requires changes to existing contracts, protocols, or consumer code that would affect existing envs.

### 3. Evolution Protocol

#### Back-compatible gaps (onboarder drives end-to-end):

1. Name the gap and the assumption it violates
2. Propose the minimal additive change (e.g., "add optional `acceleration` key to `get_state()` return dict")
3. Invoke `/superpowers:brainstorming` to validate the proposal and design the change
4. Invoke `/superpowers:skill-creator` to update the onboarder skill's assumptions list and checklists
5. Implement the framework change via the normal plan → execute cycle
6. Continue onboarding with the updated framework

#### Breaking gaps (escalate to team):

1. Document the gap clearly: what assumption is violated, why it can't be solved additively
2. Propose 2-3 options with trade-offs (e.g., "generalize the protocol" vs. "create a parallel protocol" vs. "refactor the existing contract")
3. Flag for team discussion — output a summary suitable for a Linear issue or async discussion
4. Pause onboarding until the breaking change is resolved and landed
5. After resolution, invoke `/superpowers:skill-creator` to update the onboarder

### 4. Self-Update Mechanism

After any framework evolution (whether driven by the onboarder or by external changes), the onboarder updates itself via `skill-creator`:

- Add or modify framework assumptions
- Update checklists to reflect new contract fields, protocol methods, or component categories
- Add new component types if the evolution introduced a new category
- Remove assumptions that are no longer true

This keeps the onboarder as a living document that accurately reflects the current framework state.

## Example: Unified Perception+Control Model

A team member wants to onboard a single feed-forward model that handles both perception (camera input) and control (action output).

1. **Component profile**: "End-to-end visuomotor policy — takes camera frames, outputs motor RPMs directly."
2. **Gap detection**: Violates assumption "perception and control are separate modules."
3. **Classification**: Breaking — the training entrypoint, Hydra config structure, and observation pipeline all assume separate perception and control.
4. **Escalation**: Onboarder documents the gap, proposes options:
   - Option A: Add a new `end_to_end/` module that bypasses the perception/control split, keeping existing modules untouched (partially back-compatible)
   - Option B: Refactor to a unified `policy/` module that subsumes both (breaking)
   - Option C: Treat it as a control algorithm that takes raw observations, with perception being "identity" (back-compatible shim)
5. **Team discussion**: Summary posted to Linear issue for async discussion.
6. **After resolution**: Onboarder updates its assumptions and checklists via skill-creator.

## Example: Non-Quadrotor Vehicle

A team member wants to onboard a fixed-wing env with 2 control surfaces instead of 4 motors.

1. **Gap detection**: Violates "4 motors", "motor_rpms in state", "actions are 4-dimensional."
2. **Classification**: Breaking for TrajectoryProvider (motor_rpms assumes 4), back-compatible for EpisodeMetrics (episode fields are vehicle-agnostic).
3. **Proposal**: Generalize `motor_rpms` to `actuator_state` with variable dimensionality; update `get_state()` protocol; keep EpisodeMetrics unchanged.
4. **Escalation**: Breaking change to TrajectoryProvider → team discussion.

## Non-Goals

- Automatic code generation for framework changes (human designs and reviews)
- Replacing brainstorming or writing-plans skills (onboarder delegates to them)
- Changing the metrics contract for COR-57 (this spec is about future evolution, not current implementation)
