"""Trajectory generators and dataset I/O for sysID experiments.

A single "trajectory" is a fixed-length forward rollout of NumpyQuadDynamics
from a perturbed-hover initial state, driven by a motor-speed sequence drawn
from one of several styles (random walk, sinusoidal sweep, step). The styles
together provide spectral coverage of the actuator → state response.

Saved format (.npz):
    states  : float64, shape (N_traj, T + 1, 17)
    actions : float64, shape (N_traj, T,     4)
    dt      : float scalar
    styles  : object array of strings, length N_traj
    params  : 1-D json blob describing the ground-truth VehicleParams used
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from sim.dynamics.numpy_quad import NumpyQuadDynamics
from sim.dynamics.params import VehicleParams

STYLES = ("random_walk", "sinusoidal", "step")


@dataclass
class TrajectoryDataset:
    states: np.ndarray   # (N, T+1, 17)
    actions: np.ndarray  # (N, T,   4)
    dt: float
    styles: list[str]
    params: VehicleParams

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            states=self.states.astype(np.float64),
            actions=self.actions.astype(np.float64),
            dt=np.float64(self.dt),
            styles=np.array(self.styles, dtype=object),
            params_json=_params_to_json(self.params),
        )

    @classmethod
    def load(cls, path: Path | str) -> "TrajectoryDataset":
        data = np.load(path, allow_pickle=True)
        return cls(
            states=data["states"],
            actions=data["actions"],
            dt=float(data["dt"]),
            styles=list(data["styles"]),
            params=_params_from_json(str(data["params_json"])),
        )


def _params_to_json(p: VehicleParams) -> str:
    return json.dumps({
        "mass": p.mass,
        "inertia_diag": np.diag(p.inertia).tolist(),
        "k_thrust": p.k_thrust,
        "k_torque": p.k_torque,
        "arm_length": p.arm_length,
        "tau_motor": p.tau_motor,
        "max_rpm": p.max_rpm,
        "drag_coeff": p.drag_coeff.tolist(),
    })


def _params_from_json(s: str) -> VehicleParams:
    d = json.loads(s)
    return VehicleParams(
        mass=d["mass"],
        inertia=np.diag(d["inertia_diag"]),
        k_thrust=d["k_thrust"],
        k_torque=d["k_torque"],
        arm_length=d["arm_length"],
        tau_motor=d["tau_motor"],
        max_rpm=d["max_rpm"],
        drag_coeff=np.asarray(d["drag_coeff"]),
    )


def _initial_state(p: VehicleParams, rng: np.random.Generator) -> np.ndarray:
    """Hover at z=1m with small random perturbation in vel / omega / tilt."""
    s = np.zeros(17, dtype=np.float64)
    s[0:3] = [rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5), 1.0]
    s[3:6] = rng.uniform(-0.3, 0.3, size=3)
    # Small random quaternion: ±10° tilt around random axis
    angle = rng.uniform(-0.17, 0.17)  # ~10 deg
    axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
    s[6] = np.cos(angle / 2)
    s[7:10] = np.sin(angle / 2) * axis
    s[10:13] = rng.uniform(-1.0, 1.0, size=3)
    s[13:17] = float(np.sqrt(p.mass * 9.81 / (4.0 * p.k_thrust)))
    return s


def _action_sequence(
    style: str,
    n_steps: int,
    dt: float,
    hover_omega: float,
    max_omega: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a (n_steps, 4) motor-speed command sequence in the chosen style."""
    # Action perturbations are sized to excite dynamics without spinning the
    # drone into divergence. Tighter than typical RL training because we want
    # most trajectories to survive 2 sec of integration.
    if style == "random_walk":
        # Decompose into common-mode (random walk on total thrust) + small
        # per-motor differential (torque excitation). Pure-independent random
        # walks on 4 motors create runaway torques; this stays well-behaved.
        sigma_common = 0.04 * hover_omega
        sigma_diff = 0.015 * hover_omega
        common = np.zeros(n_steps)
        common[0] = rng.normal(0, sigma_common)
        for t in range(1, n_steps):
            common[t] = 0.98 * common[t - 1] + rng.normal(0, sigma_common)
        diff = rng.normal(0, sigma_diff, size=(n_steps, 4))
        # remove diff mean per-step so it doesn't bias thrust
        diff -= diff.mean(axis=1, keepdims=True)
        actions = hover_omega + common[:, None] + diff
    elif style == "sinusoidal":
        # Each motor follows hover + A_i * sin(2π f_i t + φ_i), independent.
        amp = rng.uniform(0.03, 0.10, size=4) * hover_omega
        freq = rng.uniform(0.5, 4.0, size=4)
        phase = rng.uniform(0, 2 * np.pi, size=4)
        t = np.arange(n_steps) * dt
        actions = hover_omega + amp[None, :] * np.sin(
            2 * np.pi * freq[None, :] * t[:, None] + phase[None, :]
        )
    elif style == "step":
        # Piecewise-constant: jump every ~0.2s to a new random level around hover.
        actions = np.empty((n_steps, 4))
        steps_per_jump = max(1, int(0.2 / dt))
        for start in range(0, n_steps, steps_per_jump):
            end = min(start + steps_per_jump, n_steps)
            level = hover_omega + rng.normal(0, 0.07 * hover_omega, size=4)
            actions[start:end] = level[None, :]
    else:
        raise ValueError(f"Unknown style: {style!r}. Supported: {STYLES}")

    return np.clip(actions, 0.0, max_omega)


def _is_clean_trajectory(states: np.ndarray, omega_max: float = 50.0) -> bool:
    """A trajectory is clean iff it has no NaN/Inf and stays within physical bounds."""
    if not np.isfinite(states).all():
        return False
    omega = states[:, 10:13]
    if np.abs(omega).max() > omega_max:
        return False
    return True


def generate_dataset(
    n_traj: int,
    n_steps: int,
    dt: float,
    params: VehicleParams,
    seed: int = 0,
    styles: Iterable[str] = STYLES,
    max_retries: int = 5,
) -> TrajectoryDataset:
    """Roll N clean trajectories with a balanced mix of action styles.

    Diverged trajectories (NaN / Inf / runaway omega) are re-rolled with a
    fresh random draw up to max_retries times before giving up.
    """
    rng = np.random.default_rng(seed)
    styles_list = list(styles)
    chosen = [styles_list[i % len(styles_list)] for i in range(n_traj)]
    rng.shuffle(chosen)

    dyn = NumpyQuadDynamics(params, dt=dt)
    dyn.reset(1)
    hover_omega = dyn.hover_omega()
    max_omega = float(params.max_omega)

    states_out = np.empty((n_traj, n_steps + 1, 17), dtype=np.float64)
    actions_out = np.empty((n_traj, n_steps, 4), dtype=np.float64)
    retries_used = 0

    for i, style in enumerate(chosen):
        for attempt in range(max_retries + 1):
            s = _initial_state(params, rng)
            actions = _action_sequence(style, n_steps, dt, hover_omega, max_omega, rng)
            traj = np.empty((n_steps + 1, 17), dtype=np.float64)
            traj[0] = s
            for t in range(n_steps):
                s = dyn.step(s[None, :], actions[t][None, :], dt=dt)[0]
                traj[t + 1] = s
            if _is_clean_trajectory(traj):
                states_out[i] = traj
                actions_out[i] = actions
                retries_used += attempt
                break
        else:
            raise RuntimeError(
                f"Failed to generate a clean trajectory for style={style!r} "
                f"after {max_retries + 1} attempts. Tighten action perturbations."
            )

    if retries_used:
        print(f"  (regenerated {retries_used} diverged trajectories)")

    return TrajectoryDataset(
        states=states_out,
        actions=actions_out,
        dt=dt,
        styles=chosen,
        params=params,
    )
