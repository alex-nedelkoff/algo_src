"""NumpyQuad environment factory.

Concrete ``EnvFactory`` implementation that builds ``GateRaceEnv`` +
``VecEnvAdapter`` using our pure-numpy quadrotor dynamics.  Selected via
Hydra ``_target_: sim.envs.numpy_quad_factory.NumpyQuadEnvFactory`` in
the ``sim/numpy_quad.yaml`` config group.
"""

from __future__ import annotations

import logging
from typing import Any

from omegaconf import DictConfig, OmegaConf

from sim.domain_randomization import DomainRandomizer
from sim.dynamics.params import VehicleParams
from sim.envs.gate_race_env import GateRaceEnv
from sim.envs.vec_env_adapter import VecEnvAdapter

log = logging.getLogger(__name__)


class NumpyQuadEnvFactory:
    """Factory that builds GateRaceEnv-backed VecEnv instances.

    All sim-level parameters are captured at construction time (typically
    via Hydra recursive instantiation of the ``sim`` config group).  The
    ``make_*`` methods only receive the training-level knobs that vary
    between training and evaluation (domain randomization, reward config,
    and env count).

    Args:
        n_envs: Number of parallel environments for training.
        dt: Simulation timestep in seconds.
        max_steps: Maximum steps per episode before timeout truncation.
        gate_passage_radius: Lateral distance threshold for gate passage.
        params: Vehicle parameters — accepts either a ``VehicleParams``
            instance (when Hydra recursively instantiates the nested
            ``_target_``) or a ``DictConfig`` that will be converted.
        random_gate_start: Randomize starting gate each episode.
        start_behind_dist: Distance behind gate for start placement (m).
        start_vel_std: Std-dev of initial velocity perturbation (m/s).
        start_att_std: Std-dev of initial attitude perturbation (rad).
        start_omega_std: Std-dev of initial body-rate perturbation (rad/s).
        gate_collision: Terminate on gate-plane crossing outside opening.
        esc_nonlinearity: ESC curve parameter k in [0, 1].
        max_body_rate: Max body angular rate before crash (rad/s).
        max_velocity: Velocity clamp for float32 overflow protection (m/s).
        arena_bounds: Half-width of lateral arena (m).
        track_gen: Procedural track generation config (DictConfig or None).
        seed: Random seed for reproducible eval track generation.
    """

    def __init__(
        self,
        n_envs: int = 100,
        dt: float = 0.01,
        max_steps: int = 1000,
        gate_passage_radius: float = 1.0,
        params: VehicleParams | DictConfig | None = None,
        random_gate_start: bool = False,
        start_behind_dist: float = 1.0,
        start_vel_std: float = 0.5,
        start_att_std: float = 0.1,
        start_omega_std: float = 0.0,
        gate_collision: bool = False,
        esc_nonlinearity: float = 0.5,
        max_body_rate: float = 17.45,
        max_velocity: float = 50.0,
        arena_bounds: float = 20.0,
        track_gen: DictConfig | None = None,
        seed: int = 42,
    ) -> None:
        self.n_envs = n_envs
        self.dt = dt
        self.max_steps = max_steps
        self.gate_passage_radius = gate_passage_radius
        self.random_gate_start = random_gate_start
        self.start_behind_dist = start_behind_dist
        self.start_vel_std = start_vel_std
        self.start_att_std = start_att_std
        self.start_omega_std = start_omega_std
        self.gate_collision = gate_collision
        self.esc_nonlinearity = esc_nonlinearity
        self.max_body_rate = max_body_rate
        self.max_velocity = max_velocity
        self.arena_bounds = arena_bounds
        self._track_gen_cfg = track_gen
        self._seed = seed

        # Resolve params: accept VehicleParams directly (Hydra recursive
        # instantiation) or DictConfig (manual construction).
        self.params = _resolve_vehicle_params(params)

    # ------------------------------------------------------------------
    # EnvFactory protocol
    # ------------------------------------------------------------------

    def make_vec_env(
        self,
        domain_rand_cfg: DictConfig,
        reward_cfg: DictConfig,
    ) -> VecEnvAdapter:
        """Build a training VecEnv with domain randomization."""
        track_generator = self._make_track_generator()
        return self._build(
            n_envs=self.n_envs,
            domain_rand_cfg=domain_rand_cfg,
            reward_cfg=reward_cfg,
            track_generator=track_generator,
        )

    def make_eval_env(
        self,
        domain_rand_cfg: DictConfig,
        reward_cfg: DictConfig,
        n_envs: int,
    ) -> VecEnvAdapter:
        """Build an evaluation VecEnv (domain rand disabled)."""
        # Force domain randomization off for eval regardless of config
        eval_dr_cfg = OmegaConf.create(
            OmegaConf.to_container(domain_rand_cfg, resolve=True)
        )
        OmegaConf.update(eval_dr_cfg, "enabled", False)

        # For eval: fixed tracks (no generator), round-robin from
        # 10 pre-generated tracks for reproducibility.
        eval_tracks = None
        if self._track_gen_cfg is not None:
            import numpy as np

            tg = self._make_track_generator()
            eval_rng = np.random.default_rng(self._seed + 1000)
            n_eval_tracks = 10
            fixed = [tg.generate(eval_rng) for _ in range(n_eval_tracks)]
            eval_tracks = [fixed[i % n_eval_tracks] for i in range(n_envs)]

        return self._build(
            n_envs=n_envs,
            domain_rand_cfg=eval_dr_cfg,
            reward_cfg=reward_cfg,
            tracks=eval_tracks,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _make_track_generator(self):
        """Construct a ProceduralTrackGenerator from config, or None."""
        if self._track_gen_cfg is None:
            return None
        from sim.procedural_tracks import ProceduralTrackGenerator

        tg_dict = OmegaConf.to_container(self._track_gen_cfg, resolve=True)
        return ProceduralTrackGenerator(
            arena_half_width=self.arena_bounds,
            **tg_dict,
        )

    def _build(
        self,
        n_envs: int,
        domain_rand_cfg: DictConfig,
        reward_cfg: DictConfig,
        *,
        track_generator=None,
        tracks=None,
    ) -> VecEnvAdapter:
        """Shared builder for both training and eval envs."""
        # Domain randomization
        dr_dict: dict[str, Any] = OmegaConf.to_container(
            domain_rand_cfg, resolve=True
        )  # type: ignore[assignment]
        if dr_dict.get("enabled", False):
            domain_randomizer = DomainRandomizer.from_config(dr_dict)
        else:
            domain_randomizer = None

        # Reward weights
        reward_weights: dict[str, float] = OmegaConf.to_container(
            reward_cfg.weights, resolve=True
        )  # type: ignore[assignment]

        env = GateRaceEnv(
            params=self.params,
            n_envs=n_envs,
            dt=self.dt,
            max_steps=self.max_steps,
            gate_passage_radius=self.gate_passage_radius,
            reward_weights=reward_weights,
            v_max=reward_cfg.get("v_max", 30.0),
            action_smoothness_threshold=reward_cfg.get(
                "action_smoothness_threshold", 0.5
            ),
            random_gate_start=self.random_gate_start,
            start_behind_dist=self.start_behind_dist,
            start_vel_std=self.start_vel_std,
            start_att_std=self.start_att_std,
            start_omega_std=self.start_omega_std,
            gate_collision=self.gate_collision,
            domain_randomizer=domain_randomizer,
            esc_nonlinearity=self.esc_nonlinearity,
            max_body_rate=self.max_body_rate,
            max_velocity=self.max_velocity,
            arena_bounds=self.arena_bounds,
            track_generator=track_generator,
            tracks=tracks,
        )

        log.info(
            "Built GateRaceEnv: n_envs=%d, dt=%.3f, max_steps=%d, dr=%s",
            n_envs,
            self.dt,
            self.max_steps,
            "on" if domain_randomizer is not None else "off",
        )

        return VecEnvAdapter(env)


def _resolve_vehicle_params(
    params: VehicleParams | DictConfig | None,
) -> VehicleParams:
    """Convert *params* to a ``VehicleParams`` instance.

    Handles three cases:
      1. Already a ``VehicleParams`` — return as-is (Hydra recursive
         instantiation with nested ``_target_``).
      2. A ``DictConfig`` — extract fields and construct.
      3. ``None`` — use defaults.
    """
    if params is None:
        return VehicleParams()

    if isinstance(params, VehicleParams):
        return params

    # DictConfig path — resolve and strip Hydra meta-keys
    params_dict: dict[str, Any] = OmegaConf.to_container(
        params, resolve=True
    )  # type: ignore[assignment]
    params_dict.pop("_target_", None)
    params_dict.pop("_recursive_", None)
    params_dict.pop("_convert_", None)
    return VehicleParams(**params_dict)
