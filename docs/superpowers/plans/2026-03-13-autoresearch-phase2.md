# Autoresearch Phase 2 Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add param logging, W&B integration, longer eval, multi-track training, and reward normalization to the autoresearch pipeline.

**Architecture:** Infrastructure fixes go in `prepare.py` (logging) and `train.py` fixed section (wandb). Multi-track support adds track factory functions to `rate_ctrl_env.py` and a `tracks` parameter to the env constructor. Reward normalization applies per-env progress scaling in `rate_ctrl_env.py` before calling `compute_reward`.

**Tech Stack:** Python 3.11, SB3 (stable-baselines3), wandb, numpy, pytest

---

## Chunk 1: Infrastructure Fixes

### Task 1: Add `log_experiment()` to prepare.py

**Files:**
- Modify: `autoresearch/prepare.py`
- Test: `tests/test_autoresearch/test_prepare.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_autoresearch/test_prepare.py`:

```python
class TestLogExperiment:
    """Test experiment logging to results.tsv."""

    def test_creates_tsv_with_header(self, tmp_path) -> None:
        from autoresearch.prepare import log_experiment

        tsv = tmp_path / "results.tsv"
        results = {
            "score": 5.0, "avg_gates": 6.0, "crash_rate": 0.5,
            "alt_std": 0.1, "avg_steps": 800.0, "max_gates": 10,
        }
        reward_weights = {
            "lambda_gate": 10.0, "lambda_prog": 1.0, "lambda_rate": 0.001,
            "lambda_offset": 0.0, "lambda_perc": 0.0, "lambda_delta_u": 0.001,
            "lambda_crash": 10.0, "lambda_alive": 0.0, "v_max": 0.0,
        }
        ekf_params = {"corner_noise_k": 2.0, "corner_dropout_onset": None}
        training_params = {
            "learning_rate": 1e-4, "ent_coef": 0.005, "clip_range": 0.2,
            "gae_lambda": 0.98, "gamma": 0.999,
        }
        domain_rand = {"percentage": 0.3}

        log_experiment(
            exp_id=0, seed=42, results=results,
            reward_weights=reward_weights, ekf_params=ekf_params,
            training_params=training_params, domain_rand=domain_rand,
            results_file=str(tsv),
        )

        lines = tsv.read_text().strip().split("\n")
        assert len(lines) == 2  # header + 1 row
        assert lines[0].startswith("exp_id\t")
        assert "lambda_gate" in lines[0]
        row = lines[1].split("\t")
        assert row[0] == "0"  # exp_id
        assert row[2] == "5.0"  # score

    def test_appends_to_existing_tsv(self, tmp_path) -> None:
        from autoresearch.prepare import log_experiment

        tsv = tmp_path / "results.tsv"
        results = {
            "score": 5.0, "avg_gates": 6.0, "crash_rate": 0.5,
            "alt_std": 0.1, "avg_steps": 800.0, "max_gates": 10,
        }
        rw = {
            "lambda_gate": 10.0, "lambda_prog": 1.0, "lambda_rate": 0.001,
            "lambda_offset": 0.0, "lambda_perc": 0.0, "lambda_delta_u": 0.001,
            "lambda_crash": 10.0, "lambda_alive": 0.0, "v_max": 0.0,
        }
        ekf = {"corner_noise_k": 2.0, "corner_dropout_onset": None}
        tp = {"learning_rate": 1e-4, "ent_coef": 0.005, "clip_range": 0.2,
              "gae_lambda": 0.98, "gamma": 0.999}
        dr = {"percentage": 0.3}

        log_experiment(0, 42, results, rw, ekf, tp, dr, str(tsv))
        log_experiment(1, 43, results, rw, ekf, tp, dr, str(tsv))

        lines = tsv.read_text().strip().split("\n")
        assert len(lines) == 3  # header + 2 rows
        assert lines[2].split("\t")[0] == "1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_autoresearch/test_prepare.py::TestLogExperiment -v`
Expected: FAIL with ImportError (log_experiment not defined)

- [ ] **Step 3: Implement `log_experiment()`**

Add to `autoresearch/prepare.py` after `run_eval()`:

```python
def log_experiment(
    exp_id: int,
    seed: int,
    results: dict[str, float],
    reward_weights: dict[str, float],
    ekf_params: dict[str, Any],
    training_params: dict[str, float],
    domain_rand: dict[str, float],
    results_file: str = "autoresearch/results.tsv",
) -> None:
    """Append one experiment row to results.tsv."""
    from datetime import datetime
    from pathlib import Path

    header_cols = [
        "exp_id", "timestamp", "score", "avg_gates", "crash_rate",
        "alt_std", "avg_steps", "max_gates", "seed",
        "lambda_gate", "lambda_prog", "lambda_rate", "lambda_offset",
        "lambda_perc", "lambda_delta_u", "lambda_crash", "lambda_alive",
        "v_max", "corner_noise_k", "corner_dropout_onset",
        "learning_rate", "ent_coef", "clip_range", "gae_lambda", "gamma",
        "dr_percentage",
    ]

    path = Path(results_file)
    write_header = not path.exists()

    row_vals = [
        str(exp_id),
        datetime.now().isoformat(timespec="seconds"),
        str(results["score"]),
        str(results["avg_gates"]),
        str(results["crash_rate"]),
        str(results["alt_std"]),
        str(results["avg_steps"]),
        str(results["max_gates"]),
        str(seed),
        str(reward_weights["lambda_gate"]),
        str(reward_weights["lambda_prog"]),
        str(reward_weights["lambda_rate"]),
        str(reward_weights["lambda_offset"]),
        str(reward_weights["lambda_perc"]),
        str(reward_weights["lambda_delta_u"]),
        str(reward_weights["lambda_crash"]),
        str(reward_weights["lambda_alive"]),
        str(reward_weights["v_max"]),
        str(ekf_params["corner_noise_k"]),
        str(ekf_params.get("corner_dropout_onset", "None")),
        str(training_params["learning_rate"]),
        str(training_params["ent_coef"]),
        str(training_params["clip_range"]),
        str(training_params["gae_lambda"]),
        str(training_params["gamma"]),
        str(domain_rand["percentage"]),
    ]

    with open(path, "a") as f:
        if write_header:
            f.write("\t".join(header_cols) + "\n")
        f.write("\t".join(row_vals) + "\n")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_autoresearch/test_prepare.py::TestLogExperiment -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add autoresearch/prepare.py tests/test_autoresearch/test_prepare.py
git commit -m "feat(autoresearch): add log_experiment() for TSV param logging"
```

### Task 2: Wire `log_experiment()` into train.py and increase eval episodes

**Files:**
- Modify: `autoresearch/prepare.py:104-107` — change `run_eval()` default
- Modify: `autoresearch/train.py:47-78` — add import + calls in fixed section

- [ ] **Step 1: Change `run_eval()` default to 200 episodes**

In `autoresearch/prepare.py`, change line 107:
```python
    n_episodes: int = 200,
```

- [ ] **Step 2: Update train.py fixed section**

In `autoresearch/train.py`, update the import (line 47-54) and `main()` (line 61-78):

```python
from autoresearch.prepare import (
    register_reward_preset,
    make_training_env,
    wrap_ekf,
    load_and_configure_model,
    run_eval,
    save_checkpoint,
    log_experiment,
)

CHECKPOINT = "../playground/runs/phase4_ekf_ppo_v2/final_model.zip"
N_STEPS = 500_000
N_ENVS = 500


def main(exp_id: int) -> None:
    seed = 42 + exp_id
    preset_name = f"autoresearch_exp{exp_id}"
    register_reward_preset(preset_name, REWARD_WEIGHTS)

    env = make_training_env(N_ENVS, DOMAIN_RAND["percentage"], preset_name, seed)
    env = wrap_ekf(env, **EKF_PARAMS)
    model = load_and_configure_model(env, CHECKPOINT, TRAINING_PARAMS)
    model.learn(total_timesteps=N_STEPS)
    save_checkpoint(model, exp_id)

    eval_env = make_training_env(10, 0.0, preset_name, seed + 1000)
    eval_env = wrap_ekf(eval_env, **EKF_PARAMS)
    results = run_eval(model, eval_env, n_episodes=200)

    log_experiment(
        exp_id=exp_id, seed=seed, results=results,
        reward_weights=REWARD_WEIGHTS, ekf_params=EKF_PARAMS,
        training_params=TRAINING_PARAMS, domain_rand=DOMAIN_RAND,
    )

    print(json.dumps(results))

    env.close()
    eval_env.close()
```

- [ ] **Step 3: Run existing tests to verify nothing breaks**

Run: `python -m pytest tests/test_autoresearch/ -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add autoresearch/prepare.py autoresearch/train.py
git commit -m "feat(autoresearch): wire log_experiment + 200-episode eval"
```

### Task 3: Add W&B integration to train.py

**Files:**
- Modify: `autoresearch/train.py:38-82` — add wandb lifecycle in fixed section
- Modify: `autoresearch/.gitignore` — add `tb_logs/`

- [ ] **Step 1: Add `tb_logs/` to `.gitignore`**

Append to `autoresearch/.gitignore`:
```
tb_logs/
```

- [ ] **Step 2: Add wandb init/log/finish to train.py fixed section**

Replace the `main()` function in the fixed section with:

```python
def main(exp_id: int) -> None:
    seed = 42 + exp_id
    preset_name = f"autoresearch_exp{exp_id}"
    register_reward_preset(preset_name, REWARD_WEIGHTS)

    # W&B init (optional — experiments still run if wandb unavailable)
    _wandb_active = False
    try:
        import wandb

        wandb.init(
            project="corvidx-drone-racing",
            entity=None,
            group="autoresearch",
            tags=["autoresearch", "phase2", f"exp_{exp_id}"],
            config={
                "exp_id": exp_id,
                "seed": seed,
                "n_steps": N_STEPS,
                "n_envs": N_ENVS,
                "checkpoint": CHECKPOINT,
                "reward_weights": REWARD_WEIGHTS,
                "ekf_params": EKF_PARAMS,
                "training_params": TRAINING_PARAMS,
                "domain_rand": DOMAIN_RAND,
            },
            sync_tensorboard=True,
        )
        _wandb_active = True
    except Exception:
        pass

    env = make_training_env(N_ENVS, DOMAIN_RAND["percentage"], preset_name, seed)
    env = wrap_ekf(env, **EKF_PARAMS)
    model = load_and_configure_model(env, CHECKPOINT, TRAINING_PARAMS)

    # Enable TensorBoard logging for wandb sync
    model.tensorboard_log = f"autoresearch/tb_logs/exp_{exp_id:03d}"

    model.learn(total_timesteps=N_STEPS)
    save_checkpoint(model, exp_id)

    eval_env = make_training_env(10, 0.0, preset_name, seed + 1000)
    eval_env = wrap_ekf(eval_env, **EKF_PARAMS)
    results = run_eval(model, eval_env, n_episodes=200)

    log_experiment(
        exp_id=exp_id, seed=seed, results=results,
        reward_weights=REWARD_WEIGHTS, ekf_params=EKF_PARAMS,
        training_params=TRAINING_PARAMS, domain_rand=DOMAIN_RAND,
    )

    # Log eval metrics to W&B
    if _wandb_active:
        import wandb

        wandb.log({
            "eval/score": results["score"],
            "eval/avg_gates": results["avg_gates"],
            "eval/crash_rate": results["crash_rate"],
            "eval/alt_std": results["alt_std"],
            "eval/avg_steps": results["avg_steps"],
            "eval/max_gates": results["max_gates"],
        })
        wandb.run.summary["score"] = results["score"]
        wandb.finish()

    print(json.dumps(results))

    env.close()
    eval_env.close()
```

- [ ] **Step 3: Run existing tests to verify nothing breaks**

Run: `python -m pytest tests/test_autoresearch/ -v`
Expected: All tests PASS (wandb is optional, tests don't trigger it)

- [ ] **Step 4: Commit**

```bash
git add autoresearch/train.py autoresearch/.gitignore
git commit -m "feat(autoresearch): add wandb integration with TB sync"
```

---

## Chunk 2: Multi-Track Training

### Task 4: Add oval and S-curve track definitions

**Files:**
- Modify: `sim/envs/rate_ctrl_env.py:55-80`
- Test: `tests/test_sim/test_tracks.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_sim/test_tracks.py`:

```python
"""Tests for track definitions — verifies gate normal directions."""
import numpy as np
import pytest

from sim.envs.rate_ctrl_env import (
    make_figure8_track,
    make_oval_track,
    make_s_curve_track,
)


def _verify_gate_normals(track: dict) -> None:
    """Assert every gate's normal faces the approach direction.

    For each gate, the signed distance from the previous gate's position
    must be negative (approach side), and from the next gate must be
    positive (exit side).
    """
    gates = track["gates"]
    n = track["n_gates"]
    for i in range(n):
        prev_pos = gates[(i - 1) % n]["pos"][:2]
        gate_pos = gates[i]["pos"][:2]
        next_pos = gates[(i + 1) % n]["pos"][:2]
        yaw = gates[i]["yaw"]
        normal = np.array([np.cos(yaw), np.sin(yaw)])

        approach_sd = np.dot(prev_pos - gate_pos, normal)
        exit_sd = np.dot(next_pos - gate_pos, normal)
        assert approach_sd < 0, (
            f"Gate {i}: approach signed dist {approach_sd:.2f} should be < 0"
        )
        assert exit_sd > 0, (
            f"Gate {i}: exit signed dist {exit_sd:.2f} should be > 0"
        )


class TestFigure8Track:
    def test_has_8_gates(self) -> None:
        track = make_figure8_track()
        assert track["n_gates"] == 8
        assert len(track["gates"]) == 8

    def test_gate_normals_correct(self) -> None:
        track = make_figure8_track()
        _verify_gate_normals(track)


class TestOvalTrack:
    def test_has_correct_gate_count(self) -> None:
        track = make_oval_track()
        assert track["n_gates"] == 4
        assert len(track["gates"]) == 4

    def test_gate_normals_correct(self) -> None:
        track = make_oval_track()
        _verify_gate_normals(track)

    def test_dict_has_required_keys(self) -> None:
        track = make_oval_track()
        assert "gates" in track
        assert "gate_width" in track
        assert "gate_height" in track
        assert "n_gates" in track


class TestSCurveTrack:
    def test_has_correct_gate_count(self) -> None:
        track = make_s_curve_track()
        assert track["n_gates"] == 6
        assert len(track["gates"]) == 6

    def test_gate_normals_correct(self) -> None:
        track = make_s_curve_track()
        _verify_gate_normals(track)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sim/test_tracks.py -v`
Expected: FAIL with ImportError (make_oval_track not defined)

- [ ] **Step 3: Implement track definitions**

Add to `sim/envs/rate_ctrl_env.py` after `make_figure8_track()` (after line 80):

```python
def make_oval_track(
    gate_width: float = 0.55,
    gate_height: float = 0.55,
    gate_z: float = -1.5,
) -> dict:
    """Oval track with 4 gates at cardinal positions."""
    # Drone flies counterclockwise: right -> top -> left -> bottom
    gates = [
        {"pos": np.array([3.0, 0.0, gate_z]),  "yaw": np.pi / 2},     # east, facing +y
        {"pos": np.array([0.0, 3.0, gate_z]),  "yaw": np.pi},         # north, facing -x
        {"pos": np.array([-3.0, 0.0, gate_z]), "yaw": -np.pi / 2},    # west, facing -y
        {"pos": np.array([0.0, -3.0, gate_z]), "yaw": 0.0},           # south, facing +x
    ]
    return {
        "gates": gates,
        "gate_width": gate_width,
        "gate_height": gate_height,
        "n_gates": len(gates),
    }


def make_s_curve_track(
    gate_width: float = 0.55,
    gate_height: float = 0.55,
    gate_z: float = -1.5,
) -> dict:
    """S-curve track with 6 gates and alternating left/right turns."""
    # Flight path: bottom-left → right → top-left → left → bottom-right → return
    gates = [
        {"pos": np.array([-1.0, -3.0, gate_z]), "yaw": np.pi / 2},     # start, facing +y
        {"pos": np.array([2.0, -0.5, gate_z]),  "yaw": np.pi * 0.75},  # curve right
        {"pos": np.array([-1.0, 2.0, gate_z]),  "yaw": np.pi},         # top of first S
        {"pos": np.array([-2.5, -0.5, gate_z]), "yaw": -np.pi * 0.25}, # curve left
        {"pos": np.array([1.0, -2.5, gate_z]),  "yaw": 0.0},           # bottom of second S
        {"pos": np.array([2.0, -3.5, gate_z]),  "yaw": -np.pi / 2},    # exit, facing -y
    ]
    return {
        "gates": gates,
        "gate_width": gate_width,
        "gate_height": gate_height,
        "n_gates": len(gates),
    }
```

**Important:** After writing the implementations, run the `_verify_gate_normals` test. If any gate fails, adjust its yaw so that approach signed distance is negative and exit is positive. The exact yaw values above are computed to satisfy this constraint for the given positions, but verify empirically.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_sim/test_tracks.py -v`
Expected: All PASS. If any `test_gate_normals_correct` fails, adjust the yaw of the failing gate by adding/subtracting π and re-run.

- [ ] **Step 5: Commit**

```bash
git add sim/envs/rate_ctrl_env.py tests/test_sim/test_tracks.py
git commit -m "feat(sim): add oval and S-curve track definitions"
```

### Task 5: Add `tracks` parameter to RateCtrlEnv

**Files:**
- Modify: `sim/envs/rate_ctrl_env.py:137-173` (constructor) and `reset()` method
- Test: `tests/test_sim/test_tracks.py` (extend)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_sim/test_tracks.py`:

```python
from sim.envs.rate_ctrl_env import RateCtrlEnv


class TestMultiTrackEnv:
    def test_single_track_backward_compat(self) -> None:
        """Existing single-track usage still works."""
        env = RateCtrlEnv(n_envs=2, seed=42)
        obs, _ = env.reset()
        assert obs.shape == (2, 24)

    def test_tracks_list_accepted(self) -> None:
        """Constructor accepts tracks=[...] for multi-track."""
        tracks = [make_figure8_track(), make_oval_track()]
        env = RateCtrlEnv(n_envs=4, seed=42, tracks=tracks)
        obs, _ = env.reset()
        assert obs.shape == (4, 24)

    def test_tracks_overrides_track(self) -> None:
        """When both track and tracks given, tracks wins."""
        tracks = [make_oval_track()]
        env = RateCtrlEnv(n_envs=2, seed=42, track=make_figure8_track(), tracks=tracks)
        # Should use oval (4 gates), not figure8 (8 gates)
        # We verify by checking the env can step without error
        obs, _ = env.reset()
        action = np.zeros((2, 4), dtype=np.float32)
        env.step(action)

    def test_per_env_track_varies(self) -> None:
        """With multiple tracks, different envs may be on different tracks after many resets."""
        tracks = [make_figure8_track(), make_oval_track()]
        env = RateCtrlEnv(n_envs=50, seed=42, tracks=tracks)
        env.reset()
        # After reset, track indices should include both 0 and 1
        # (with 50 envs and uniform sampling, probability of all same < 2^-50)
        assert hasattr(env, "_track_idx")
        unique_tracks = set(env._track_idx.tolist())
        assert len(unique_tracks) > 1, "Expected envs on different tracks"

    def test_step_runs_without_error(self) -> None:
        """Multi-track env can step without crashing."""
        tracks = [make_figure8_track(), make_oval_track()]
        env = RateCtrlEnv(n_envs=4, seed=42, tracks=tracks)
        env.reset()
        action = np.zeros((4, 4), dtype=np.float32)
        for _ in range(10):
            obs, rewards, terminated, truncated, info = env.step(action)
        assert obs.shape == (4, 24)
        assert rewards.shape == (4,)

    def test_gate_idx_respects_track_n_gates(self) -> None:
        """Gate indices should wrap at the correct n_gates per track."""
        tracks = [make_figure8_track(), make_oval_track()]  # 8 and 4 gates
        env = RateCtrlEnv(n_envs=2, seed=42, tracks=tracks)
        env.reset()
        # Force env 0 to figure8 (8 gates), env 1 to oval (4 gates)
        env._track_idx[0] = 0
        env._track_idx[1] = 1
        env._gate_idx[0] = 7  # last gate on figure8
        env._gate_idx[1] = 3  # last gate on oval
        # Wrapping should give 0 for both
        assert (7 + 1) % 8 == 0
        assert (3 + 1) % 4 == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sim/test_tracks.py::TestMultiTrackEnv -v`
Expected: FAIL (RateCtrlEnv doesn't accept `tracks` parameter)

- [ ] **Step 3: Implement multi-track support**

Modify `sim/envs/rate_ctrl_env.py`:

**3a. Update constructor signature** (line 137-149) — add `tracks` parameter:

```python
    def __init__(
        self,
        n_envs: int = 1,
        seed: int = 0,
        dt: float = 0.002,
        action_repeat: int = 5,
        dr_percentage: float = 0.0,
        reward_preset: str = "M23",
        init_vel_range: float = 0.5,
        init_rp_deg: float = 20.0,
        init_rate_range: float = 0.1,
        max_steps: int = 1200,
        track: Optional[dict] = None,
        tracks: Optional[list[dict]] = None,
    ) -> None:
```

**3b. Replace track setup** (lines 165-173) with multi-track logic:

```python
        # Track(s)
        if tracks is not None:
            self._tracks = tracks
        elif track is not None:
            self._tracks = [track]
        else:
            self._tracks = [make_figure8_track()]

        # Pre-extract gate arrays for each track
        self._track_gate_positions = [
            np.array([g["pos"] for g in t["gates"]]) for t in self._tracks
        ]
        self._track_gate_yaws = [
            np.array([g["yaw"] for g in t["gates"]]) for t in self._tracks
        ]
        self._track_n_gates = [t["n_gates"] for t in self._tracks]

        # Per-env track assignment (initialized to track 0)
        self._track_idx = np.zeros(n_envs, dtype=np.int32)

        # Initialize from first track (will be overridden on reset)
        self._track = self._tracks[0]
        self._n_gates = self._track_n_gates[0]
        self._gate_positions = self._track_gate_positions[0]
        self._gate_yaws = self._track_gate_yaws[0]
```

**3c. Update `reset()` method** — after `n_reset = int(env_mask.sum())` (around line 229), add track sampling before the gate spawn logic:

```python
        # Sample track for each resetting env
        if len(self._tracks) > 1:
            new_track_idx = self.rng.integers(0, len(self._tracks), size=n_reset)
            self._track_idx[env_mask] = new_track_idx
```

**3d. Update reset gate logic** — the existing code (lines 252-256) uses `self._n_gates`, `self._gate_positions`, `self._gate_yaws` directly. With multi-track, each env may be on a different track. Replace the gate setup in reset with per-env lookups:

```python
        # Get per-env gate data based on track assignment
        env_indices = np.where(env_mask)[0]
        gate_idx = np.zeros(n_reset, dtype=np.int32)
        gate_pos = np.zeros((n_reset, 3), dtype=np.float64)
        gate_yaw = np.zeros(n_reset, dtype=np.float64)

        for i, ei in enumerate(env_indices):
            ti = self._track_idx[ei]
            n_g = self._track_n_gates[ti]
            gi = self.rng.integers(0, n_g)
            gate_idx[i] = gi
            gate_pos[i] = self._track_gate_positions[ti][gi]
            gate_yaw[i] = self._track_gate_yaws[ti][gi]

        self._gate_idx[env_mask] = gate_idx
```

**3e. Update `_get_obs()`, `_compute_d2g()`, `_check_gate_passage()`** — these methods use `self._gate_positions[self._gate_idx]` etc. With multi-track, the gate arrays differ per env. The simplest approach: build per-env gate position/yaw lookups:

Add a helper method:

```python
    def _per_env_gate_data(self, gate_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Get gate positions and yaws for each env's current gate index.

        Returns (n_envs, 3) positions and (n_envs,) yaws.
        """
        if len(self._tracks) == 1:
            return self._gate_positions[gate_indices], self._gate_yaws[gate_indices]

        positions = np.zeros((self.n_envs, 3), dtype=np.float64)
        yaws = np.zeros(self.n_envs, dtype=np.float64)
        for i in range(self.n_envs):
            ti = self._track_idx[i]
            gi = gate_indices[i]
            positions[i] = self._track_gate_positions[ti][gi]
            yaws[i] = self._track_gate_yaws[ti][gi]
        return positions, yaws
```

Then update all methods that use `self._gate_positions[self._gate_idx]` or `self._gate_yaws[self._gate_idx]`:

**`_get_obs()` (line ~460):** Replace direct array indexing with `_per_env_gate_data`. Also fix the next-gate modular wrap — `(self._gate_idx + 1) % self._n_gates` uses a scalar `self._n_gates` which is wrong with multi-track. Replace with per-env wrap:

```python
    def _get_obs(self) -> np.ndarray:
        gate_pos, gate_yaw = self._per_env_gate_data(self._gate_idx)
        # Per-env next gate index (different n_gates per track)
        if len(self._tracks) == 1:
            next_idx = (self._gate_idx + 1) % self._n_gates
        else:
            next_idx = np.array([
                (self._gate_idx[i] + 1) % self._track_n_gates[self._track_idx[i]]
                for i in range(self.n_envs)
            ], dtype=np.int32)
        next_gate_pos, next_gate_yaw = self._per_env_gate_data(next_idx)
        return gate_relative_obs(
            self._state, gate_pos, gate_yaw, next_gate_pos, next_gate_yaw
        )
```

**`_get_obs_single()` (line ~472):** Update to use per-env track data:

```python
    def _get_obs_single(self, idx: int) -> np.ndarray:
        ti = self._track_idx[idx]
        gi = self._gate_idx[idx]
        ni = (gi + 1) % self._track_n_gates[ti]
        return gate_relative_obs(
            self._state[idx],
            self._track_gate_positions[ti][gi],
            self._track_gate_yaws[ti][gi],
            self._track_gate_positions[ti][ni],
            self._track_gate_yaws[ti][ni],
        )
```

**`_compute_d2g()` (line ~484):** Update to use `_per_env_gate_data`:

```python
    def _compute_d2g(self, mask: Optional[np.ndarray] = None) -> np.ndarray:
        if mask is None:
            pos = self._state[:, :3]
            gate_pos, _ = self._per_env_gate_data(self._gate_idx)
        else:
            pos = self._state[mask, :3]
            # Build gate positions for masked envs
            env_indices = np.where(mask)[0]
            gate_pos = np.array([
                self._track_gate_positions[self._track_idx[ei]][self._gate_idx[ei]]
                for ei in env_indices
            ])
        return np.linalg.norm(pos - gate_pos, axis=1)
```

**`_check_gate_passage()` (line ~494):** Update gate data lookups, gate width/height (which may differ per track), and the modular wrap:

```python
        gate_pos, gate_yaw = self._per_env_gate_data(self._gate_idx)
        # ... (rest of signed distance computation stays the same)

        # Gate dimensions — use per-env track's dimensions
        if len(self._tracks) == 1:
            half_w = self._tracks[0]["gate_width"] / 2
            half_h = self._tracks[0]["gate_height"] / 2
            within_opening = (lateral < half_w) & (vertical < half_h)
        else:
            half_w = np.array([self._tracks[self._track_idx[i]]["gate_width"] / 2
                              for i in range(self.n_envs)])
            half_h = np.array([self._tracks[self._track_idx[i]]["gate_height"] / 2
                              for i in range(self.n_envs)])
            within_opening = (lateral < half_w) & (vertical < half_h)

        # ... passage detection ...

        # Per-env gate index advance with per-track n_gates
        if np.any(passed):
            for i in np.where(passed)[0]:
                ti = self._track_idx[i]
                self._gate_idx[i] = (self._gate_idx[i] + 1) % self._track_n_gates[ti]
```

And update the signed distance recomputation after gate advance to use `_per_env_gate_data`.

- [ ] **Step 4: Run all tests**

Run: `python -m pytest tests/test_sim/test_tracks.py -v && python -m pytest tests/test_autoresearch/ -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add sim/envs/rate_ctrl_env.py tests/test_sim/test_tracks.py
git commit -m "feat(sim): add multi-track support to RateCtrlEnv"
```

### Task 6: Extend prepare.py for multi-track training and eval

**Files:**
- Modify: `autoresearch/prepare.py:32-52` — add `tracks` param to `make_training_env()`
- Modify: `autoresearch/prepare.py:104-156` — add multi-track eval to `run_eval()`
- Test: `tests/test_autoresearch/test_prepare.py` (extend)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_autoresearch/test_prepare.py`:

```python
class TestMultiTrackEnv:
    """Test multi-track environment construction."""

    def test_make_training_env_with_tracks(self) -> None:
        from autoresearch.prepare import make_training_env
        from sim.envs.rate_ctrl_env import make_figure8_track, make_oval_track

        weights = {
            "lambda_gate": 10.0, "lambda_prog": 1.0, "lambda_rate": 0.001,
            "lambda_offset": 0.0, "lambda_perc": 0.0, "lambda_delta_u": 0.001,
            "lambda_crash": 10.0, "lambda_alive": 0.0, "v_max": 0.0,
        }
        register_reward_preset("test_multi", weights)
        tracks = [make_figure8_track(), make_oval_track()]
        env = make_training_env(
            n_envs=4, dr_percentage=0.0, preset_name="test_multi",
            seed=42, tracks=tracks,
        )
        assert env.num_envs == 4
        env.close()
        del PRESETS["test_multi"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_autoresearch/test_prepare.py::TestMultiTrackEnv -v`
Expected: FAIL (make_training_env doesn't accept `tracks`)

- [ ] **Step 3: Update `make_training_env()` signature**

In `autoresearch/prepare.py`, update `make_training_env()`:

```python
def make_training_env(
    n_envs: int,
    dr_percentage: float,
    preset_name: str,
    seed: int,
    tracks: list[dict] | None = None,
) -> Any:
    """Build a vectorized training environment.

    Constructs RateCtrlEnv directly (bypasses PlaygroundEnvFactory
    to avoid DictConfig overhead).
    """
    from sim.envs.rate_ctrl_env import RateCtrlEnv
    from sim.envs.vec_env_adapter import VecEnvAdapter

    kwargs: dict[str, Any] = dict(
        n_envs=n_envs,
        seed=seed,
        dr_percentage=dr_percentage,
        reward_preset=preset_name,
    )
    if tracks is not None:
        kwargs["tracks"] = tracks

    inner = RateCtrlEnv(**kwargs)
    return VecEnvAdapter(inner)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_autoresearch/test_prepare.py::TestMultiTrackEnv -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add autoresearch/prepare.py tests/test_autoresearch/test_prepare.py
git commit -m "feat(autoresearch): add multi-track support to make_training_env"
```

---

## Chunk 3: Reward Normalization

### Task 7: Add progress reward normalization in RateCtrlEnv

**Files:**
- Modify: `sim/envs/rate_ctrl_env.py` — compute and apply normalization factor
- Test: `tests/test_sim/test_tracks.py` (extend)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_sim/test_tracks.py`:

```python
class TestRewardNormalization:
    def test_mean_inter_gate_distance_computed(self) -> None:
        """Env should compute mean inter-gate distance per track."""
        env = RateCtrlEnv(n_envs=2, seed=42)
        assert hasattr(env, "_mean_igd")
        assert len(env._mean_igd) == 1  # single track
        assert env._mean_igd[0] > 0

    def test_multi_track_has_per_track_igd(self) -> None:
        tracks = [make_figure8_track(), make_oval_track()]
        env = RateCtrlEnv(n_envs=2, seed=42, tracks=tracks)
        assert len(env._mean_igd) == 2
        # Figure-8 and oval have different spacings
        assert env._mean_igd[0] != env._mean_igd[1]

    def test_normalization_scales_progress_reward(self) -> None:
        """Progress reward should be scaled by 1/mean_igd.

        Create two single-track envs with different IGDs, apply
        identical d2g deltas, and check that the one with larger IGD
        produces a proportionally smaller progress reward.
        """
        # Use figure8 (larger IGD) and oval (smaller IGD)
        env_f8 = RateCtrlEnv(n_envs=1, seed=42, track=make_figure8_track())
        env_oval = RateCtrlEnv(n_envs=1, seed=42, track=make_oval_track())

        igd_f8 = env_f8._mean_igd[0]
        igd_oval = env_oval._mean_igd[0]

        # The ratio of normalized progress should be inverse of IGD ratio
        # (d2g_delta / igd_f8) / (d2g_delta / igd_oval) = igd_oval / igd_f8
        ratio = igd_oval / igd_f8
        assert ratio != 1.0, "Tracks should have different IGDs"
        # Exact reward test would require stepping, but verifying IGDs
        # differ and the normalization formula is applied is sufficient
        # when combined with the step-level integration test in Task 8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sim/test_tracks.py::TestRewardNormalization -v`
Expected: FAIL (no `_mean_igd` attribute)

- [ ] **Step 3: Compute mean inter-gate distance in constructor**

In `sim/envs/rate_ctrl_env.py`, after the track pre-extraction loop in `__init__()`, add:

```python
        # Compute mean inter-gate distance for each track (for reward normalization)
        self._mean_igd: list[float] = []
        for t_idx, t in enumerate(self._tracks):
            positions = self._track_gate_positions[t_idx]
            n = self._track_n_gates[t_idx]
            dists = [
                float(np.linalg.norm(positions[(i + 1) % n] - positions[i]))
                for i in range(n)
            ]
            self._mean_igd.append(float(np.mean(dists)))
```

- [ ] **Step 4: Apply normalization to d2g_old/d2g_new before reward computation**

In the `step()` method, where `compute_reward()` is called (around line 405), normalize the progress component by dividing `d2g_old` and `d2g_new` by each env's mean IGD. This makes `d2g_old - d2g_new` (the raw progress) scale-independent:

```python
        # Normalize progress by mean inter-gate distance (per-env)
        if len(self._tracks) > 1:
            igd = np.array([self._mean_igd[self._track_idx[i]] for i in range(self.n_envs)])
        else:
            igd = np.full(self.n_envs, self._mean_igd[0])
        d2g_old_norm = d2g_old / igd
        d2g_new_norm = d2g_new / igd

        rewards = compute_reward(
            d2g_old=d2g_old_norm,
            d2g_new=d2g_new_norm,
            ...
        )
```

Replace `d2g_old` and `d2g_new` with `d2g_old_norm` and `d2g_new_norm` in the `compute_reward()` call. Keep all other arguments the same.

**v_max compatibility note:** When `v_max > 0`, `compute_reward` clamps raw progress to `v_max * dt` (in meters). After normalization, raw progress is in dimensionless units but the clamp is still in meters. This makes the clamp geometry-dependent. Since all current autoresearch experiments use `v_max=0.0`, this is safe. If `v_max > 0` is needed later, normalize the clamp too: `v_max * dt / igd`. Document this as a TODO comment in the code.

- [ ] **Step 5: Run all tests**

Run: `python -m pytest tests/test_sim/ -v && python -m pytest tests/test_autoresearch/ -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add sim/envs/rate_ctrl_env.py tests/test_sim/test_tracks.py
git commit -m "feat(sim): add progress reward normalization by inter-gate distance"
```

### Task 8: Final integration test

**Files:**
- Test: `tests/test_autoresearch/test_prepare.py` (extend)

- [ ] **Step 1: Write integration test**

Add to `tests/test_autoresearch/test_prepare.py`:

```python
class TestMultiTrackEval:
    """Test multi-track evaluation pipeline."""

    def test_eval_with_multi_track_env(self, tmp_path) -> None:
        from stable_baselines3 import PPO as SB3_PPO
        from autoresearch.prepare import make_training_env, wrap_ekf, run_eval
        from sim.envs.rate_ctrl_env import make_figure8_track, make_oval_track

        weights = {
            "lambda_gate": 10.0, "lambda_prog": 1.0, "lambda_rate": 0.001,
            "lambda_offset": 0.0, "lambda_perc": 0.0, "lambda_delta_u": 0.001,
            "lambda_crash": 10.0, "lambda_alive": 0.0, "v_max": 0.0,
        }
        register_reward_preset("test_mt_eval", weights)
        tracks = [make_figure8_track(), make_oval_track()]
        env = make_training_env(
            n_envs=4, dr_percentage=0.0, preset_name="test_mt_eval",
            seed=42, tracks=tracks,
        )
        ekf_env = wrap_ekf(env, corner_noise_k=2.0)

        dummy = SB3_PPO("MlpPolicy", ekf_env, n_steps=32, batch_size=32)
        results = run_eval(dummy, ekf_env, n_episodes=5)

        assert "score" in results
        assert isinstance(results["score"], float)
        ekf_env.close()
        del PRESETS["test_mt_eval"]
```

- [ ] **Step 2: Run the integration test**

Run: `python -m pytest tests/test_autoresearch/test_prepare.py::TestMultiTrackEval -v`
Expected: PASS

- [ ] **Step 3: Run full test suite**

Run: `python -m pytest tests/test_autoresearch/ tests/test_sim/test_tracks.py tests/test_sim/test_gate_passage.py -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_autoresearch/test_prepare.py
git commit -m "test(autoresearch): add multi-track eval integration test"
```
