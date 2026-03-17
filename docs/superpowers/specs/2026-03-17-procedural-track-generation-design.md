# Procedural Track Generation for Generalization to Unseen Courses

**Date:** 2026-03-17
**Status:** Approved
**Goal:** Enable RL policy to generalize to arbitrary unseen race courses by training on procedurally generated track layouts.

## Problem

The current training pipeline uses a single hardcoded figure-8 track (8 gates). The policy learns to race this specific geometry well but may fail on courses with different gate counts, spacing, turn angles, or elevation profiles. For competition readiness, the policy must handle any track layout it has never seen.

## Approach: Sequential Gate Placement

Generate closed-loop tracks by iteratively placing gates with randomized parameters. Each gate is placed relative to the previous one by sampling a turn angle, distance, and elevation delta. The final gate must connect back to gate 0 within closure constraints.

### Why this approach

- Direct control over all parameters that matter (spacing, turn sharpness, elevation)
- Lightweight — just geometry and trig, no spline fitting
- Produces the widest variety of track shapes (hairpins, straights, S-curves, loops)
- Closure validation ensures all tracks are flyable closed loops

## Design

### 1. Track Generator (`sim/procedural_tracks.py`)

New class `ProceduralTrackGenerator` with the following configurable parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_gates_min` | int | 4 | Minimum gates per track |
| `n_gates_max` | int | 12 | Maximum gates per track |
| `gate_spacing_min` | float | 2.0 | Min distance between consecutive gates (m) |
| `gate_spacing_max` | float | 8.0 | Max distance between consecutive gates (m) |
| `turn_angle_min` | float | -120.0 | Min direction change at each gate (degrees) |
| `turn_angle_max` | float | 120.0 | Max direction change at each gate (degrees) |
| `elevation_min` | float | 1.0 | Min gate height (m) |
| `elevation_max` | float | 4.0 | Max gate height (m) |
| `elevation_delta_max` | float | 0.8 | Max height change between consecutive gates (m) |
| `arena_half_width` | float | 20.0 | Gates must fit within this bound (m) |
| `closure_max_angle` | float | 90.0 | Max turn angle for closing segment (degrees) |
| `closure_max_retries` | int | 10 | Retries before relaxing constraints |

**Algorithm:**

1. Sample `n_gates` uniformly from `[n_gates_min, n_gates_max]`.
2. Place gate 0 at a random position near center with random yaw.
3. For each subsequent gate `i`:
   - Sample turn angle from clipped normal distribution (gentle turns more common).
   - Sample distance from `[gate_spacing_min, gate_spacing_max]`.
   - Sample elevation: clamp to `[elevation_min, elevation_max]` and `|delta_z| <= elevation_delta_max`.
   - Compute new position. If outside arena bounds, clamp and adjust.
4. Validate closure: check that the segment from last gate back to gate 0 has turn angle <= `closure_max_angle` and distance <= `gate_spacing_max * 2`. If not, reject and regenerate (up to `closure_max_retries`). If all retries exhausted, fall back to `build_figure8_track()` and log a warning.
5. Validate minimum separation: check all non-consecutive gate pairs are >= `gate_spacing_min` apart. Reject if violated (counts toward retry budget).
6. Set each gate's orientation quaternion to face the next gate (matching existing `build_figure8_track` convention).
7. Return a `Track` object (list of `GateState`).

**Interface:**

```python
class ProceduralTrackGenerator:
    def __init__(self, *, n_gates_min=4, n_gates_max=12, ...): ...
    def generate(self, rng: np.random.Generator) -> Track: ...
```

### 2. GateRaceEnv Integration (`sim/envs/gate_race_env.py`)

**New constructor parameter:**

```python
track_generator: ProceduralTrackGenerator | None = None
```

**Per-env track storage:**

Replace `self.track: Track` with `self._tracks: list[Track]` (one per env). All internal code that accesses `self.track` changes to `self._tracks[i]` where `i` is the env index. Affected areas:

- Gate passage detection (plane-crossing loop)
- Observation computation (gate-relative position/yaw)
- Distance-to-gate calculation
- Lap counting (wraps modulo `self._tracks[i].num_gates`)
- `_randomize_start` (spawns drone relative to a gate)
- `_update_gate_tracking` (gate index management)
- `get_gate_geometry()` (protocol method, now per-env)

**Reset behavior:**

When `track_generator` is set, each env gets a fresh track on reset:

```python
def _reset_env(self, i: int) -> None:
    if self.track_generator is not None:
        self._tracks[i] = self.track_generator.generate(self.np_random)
    # ... existing reset logic (drone state, gate index, etc.)
```

**Progress reward normalization:** `GateRaceEnv` uses delta-distance progress rewards via `monorace_reward()` (`prev_dist - curr_dist`), which scales naturally with gate spacing — closer gates produce smaller deltas. No IGD normalization is needed (that pattern exists only in `RateCtrlEnv`). The delta-distance approach works correctly with variable gate counts without modification.

**Backwards compatibility:**

- No generator, no track: defaults to figure-8 (all envs share it) — current behavior
- Track provided, no generator: all envs share that track — current behavior
- Generator provided: per-env procedural tracks — new behavior

### 3. TrajectoryProvider / Metrics Contract

The `TrajectoryProvider` protocol (`metrics/contract.py`) defines `get_gate_geometry()` returning a single `(n_gates, 3)` array, assuming all envs share one track. With per-env tracks, this breaks.

**Solution:** `get_gate_geometry()` takes an optional `env_idx` parameter (default 0). This is a breaking change to a `@runtime_checkable` Protocol — all implementors and call sites must update:

- `GateRaceEnv.get_gate_geometry()` — returns `self._tracks[env_idx]` geometry
- `RateCtrlEnv.get_gate_geometry()` — updated signature (behavior unchanged, single track)
- `TrajectoryRecorderCallback` (`training/trajectory_recorder.py` line 145) — passes env index, re-extracts geometry on each recording interval (not cached at init, since tracks change on reset)

Aggregate metrics (gates/ep, laps/ep) remain env-agnostic since they count events, not positions.

### 4. Factory and Config (`numpy_quad_factory.py`, `configs/track_gen/`)

**New Hydra config group** `configs/track_gen/procedural.yaml`:

```yaml
n_gates_min: 4
n_gates_max: 12
gate_spacing_min: 2.0
gate_spacing_max: 8.0
turn_angle_min: -120
turn_angle_max: 120
elevation_min: 1.0
elevation_max: 4.0
elevation_delta_max: 0.8
closure_max_angle: 90
```

**Factory changes:**

When `track_gen` config is present, the factory constructs a `ProceduralTrackGenerator` and passes it to `GateRaceEnv`.

**Usage:**

```bash
# Existing behavior (figure-8)
python -m training +experiment=monorace_baseline

# Procedural tracks
python -m training +experiment=monorace_baseline track_gen=procedural
```

**Eval tracks:**

For evaluation, the factory generates a fixed set of 10 tracks at env construction time using a dedicated eval seed (derived from the global `seed` config). These 10 tracks are assigned round-robin to eval envs (env 0 gets track 0, env 1 gets track 1, ..., env 10 gets track 0 again). On eval reset, each env keeps its assigned track (no regeneration). This provides diverse eval while keeping metrics reproducible across runs.

**Arena bounds sync:**

The generator's `arena_half_width` is not set independently — the factory passes the env's `arena_bounds` value to the generator, ensuring gates are always placed within the env's crash boundary.

### 5. Logging

**Per-episode info dict additions:**

- `n_gates`: number of gates in the episode's track
- `track_id`: hash of gate positions (for correlating performance with specific tracks in wandb)

**At viz_freq:** Full track geometry logged alongside rerun recordings for visual inspection.

### 6. Performance Impact

Track generation cost per reset: microseconds (random sampling + trig for ~4-12 gates). Memory: 100 envs × 12 gates × 7 floats × 8 bytes = ~67 KB. Expected FPS impact: <1%. The physics step dominates, not track generation.

## Files

| File | Change |
|------|--------|
| `sim/procedural_tracks.py` | New — generator class |
| `sim/envs/gate_race_env.py` | Per-env tracks, generator integration |
| `sim/envs/numpy_quad_factory.py` | Pass track_gen config to env, arena bounds sync |
| `metrics/contract.py` | Add `env_idx` param to `get_gate_geometry()` |
| `training/trajectory_recorder.py` | Pass env_idx, refresh geometry per recording interval |
| `configs/track_gen/procedural.yaml` | New — generation parameters |
| `tests/test_sim/test_procedural_tracks.py` | New — generator unit tests |
| `tests/test_sim/test_procedural_training.py` | New — integration tests |

## Testing

**Unit tests (`test_procedural_tracks.py`):**

- Generated tracks have correct gate count within `[n_gates_min, n_gates_max]`
- All gate positions within arena bounds
- All orientations are unit quaternions
- Consecutive gate distances within `[spacing_min, spacing_max]`
- Gate heights within `[elevation_min, elevation_max]`
- Consecutive elevation deltas within `elevation_delta_max`
- Closure segment angle within `closure_max_angle`
- Each gate faces the next gate
- Seeded RNG produces identical tracks
- Invalid params (e.g., `n_gates_min=0`) raise ValueError

**Integration tests (`test_procedural_training.py`):**

- `GateRaceEnv` with generator builds and steps without errors
- Different envs hold different tracks after reset
- Gate passage detection works on generated tracks
- Factory with `track_gen=procedural` config produces working env

## Not In Scope

- Point-to-point (non-looping) tracks — can be added later
- Track scale randomization — fixed arena size for now
- Curriculum learning (progressive difficulty) — separate feature
- Multi-track eval with DR on — separate investigation
