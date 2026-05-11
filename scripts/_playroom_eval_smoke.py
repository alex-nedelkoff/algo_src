"""Smoke test: load the 5-expert MoE checkpoint into WarehouseRaceEnv on the
playroom_v1 asset bundle and run an episode.

Mirrors scripts/_warehouse_eval_smoke.py but points at configs/sim/playroom_v1.yaml
and reports gate pass / crash / total reward. The goal: confirm the previously-
trained policy actually flies through 3DGS-scene gates without retraining.

    conda run -n monorace python -m scripts._playroom_eval_smoke
"""
from __future__ import annotations

# Windows DLL workaround (must run BEFORE importing stable_baselines3 / numpy).
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch  # noqa: F401, E402  — preload before stable_baselines3

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from control.algorithms.ppo import PPO as CorvidxPPO  # noqa: E402
from sim.dynamics.params import VehicleParams  # noqa: E402
from sim.envs.warehouse_race_env import WarehouseRaceEnv  # noqa: E402
from sim.types import ActionMode  # noqa: E402


def _training_racing_params() -> VehicleParams:
    """5" racing quad params the MoE was trained against (per m5_closed_loop_demo).

    The default GateRaceEnv params are CrazyFlie-scale (~250 g) — running the
    MoE on those puts the policy OOD via the TRPY mixer's action mapping and
    it stalls / never reaches training-altitude. See `_training_racing_params`
    in scripts/perception/m5_closed_loop_demo.py for the canonical comment.
    """
    return VehicleParams(
        mass=0.752, arm_length=0.170,
        k_thrust=2.49e-6, k_torque=8.80e-8, tau_motor=0.04,
        inertia=np.diag([0.0025, 0.0025, 0.0045]),
        drag_coeff=np.array([0.01, 0.01, 0.005]),
        max_rpm=31470.0,
    )

CHECKPOINT = r"C:\Users\alexj\Documents\algo_src\outputs\expanded_33dim\model.zip"
CONFIG = "configs/sim/playroom_v1.yaml"


def main() -> None:
    cfg = yaml.safe_load(open(CONFIG))
    env_kwargs = dict(
        asset_dir=cfg["asset_dir"],
        gate_order=cfg["gate_order"],
        world_offset_z=cfg["world_offset_z"],
        reorient_to_racing_line=cfg["reorient_to_racing_line"],
        legacy_obs_v1=cfg["legacy_obs_v1"],
        action_mode=ActionMode(cfg["action_mode"]),
        dt=cfg["dt"],
        max_steps=cfg["max_steps"],
        ceiling=cfg["ceiling"],
        arena_bounds=cfg["arena_bounds"],
        gate_passage_radius=cfg["gate_passage_radius"],
        gate_collision=cfg["gate_collision"],
        # Match training vehicle dynamics — the MoE was trained on a 5" racing
        # quad (~752 g). Falling back to GateRaceEnv's default (CrazyFlie,
        # ~250 g) is the documented OOD-stall cause in m5_closed_loop_demo.py.
        params=_training_racing_params(),
    )
    # Optional overrides for n_lookahead_gates / n_action_history (needed
    # when legacy_obs_v1=false to target a specific obs dim).
    if "n_lookahead_gates" in cfg:
        env_kwargs["n_lookahead_gates"] = cfg["n_lookahead_gates"]
    if "n_action_history" in cfg:
        env_kwargs["n_action_history"] = cfg["n_action_history"]
    env = WarehouseRaceEnv(**env_kwargs)
    print(f"env obs={env.observation_space.shape} action={env.action_space.shape}")

    # Match moe_generalist.yaml: 5 experts (one expanded from a 4-expert base),
    # 2x128 hidden, top-2 router, log_std_init=-1.0.
    ppo = CorvidxPPO(
        moe=True,
        n_experts=5,
        expert_hidden_dim=128,
        top_k=2,
        balance_coef=0.01,
        log_std_init=-1.0,
    )
    ppo._model = ppo._create_model(env)
    ppo.load(CHECKPOINT, env=env)
    print(f"model loaded from {CHECKPOINT}")

    spawn_pos = np.array(cfg["spawn"]["position"])
    spawn_yaw = float(cfg["spawn"]["yaw_rad"])

    state = np.zeros(17)
    state[0:3] = spawn_pos
    state[6] = float(np.cos(spawn_yaw / 2.0))   # quat w
    state[9] = float(np.sin(spawn_yaw / 2.0))   # quat z (yaw-only)
    obs, _ = env.reset(options={"initial_state": state})
    print(f"spawn={spawn_pos.round(2).tolist()} yaw_deg={np.degrees(spawn_yaw):.1f}")
    print(f"reset obs shape={obs.shape} first 8={obs[:8].round(3)}")

    action, _ = ppo.predict(obs, deterministic=True)
    print(f"first action={action}")

    total_reward = 0.0
    last_gates_passed = 0
    i = -1
    # Per-step logging for offline trajectory visualization.
    positions, rewards, gate_events = [], [], []
    positions.append(env._states[0, 0:3].copy())
    for i in range(cfg["max_steps"]):
        obs, reward, term, trunc, _ = env.step(action)
        total_reward += float(reward[0])
        positions.append(env._states[0, 0:3].copy())
        rewards.append(float(reward[0]))
        gp = int(env._gates_passed[0])
        if gp > last_gates_passed:
            print(f"step {i:>4}: GATE {gp} passed @ "
                  f"pos={env._states[0, 0:3].round(2).tolist()}")
            gate_events.append((i + 1, gp, env._states[0, 0:3].copy()))
            last_gates_passed = gp
        if term[0] or trunc[0]:
            print(f"step {i:>4}: term={bool(term[0])} trunc={bool(trunc[0])} "
                  f"pos={env._states[0, 0:3].round(2).tolist()} "
                  f"reason_code={env._termination_reasons[0]}")
            break
        action, _ = ppo.predict(obs, deterministic=True)

    print(f"\n=== Result ===")
    print(f"  gates_passed = {int(env._gates_passed[0])} / {env.track.num_gates}")
    print(f"  steps        = {i + 1}")
    print(f"  total_reward = {total_reward:.2f}")
    termination_reason = int(env._termination_reasons[0])
    env.close()

    # ---- Save trajectory ------------------------------------------------
    from pathlib import Path
    out_dir = Path("outputs"); out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "playroom_eval_traj.npz"
    np.savez(
        out_path,
        positions=np.array(positions),                    # (N+1, 3)
        rewards=np.array(rewards),                        # (N,)
        gate_pass_steps=np.array([e[0] for e in gate_events]),
        gate_pass_positions=np.array([e[2] for e in gate_events]) if gate_events else np.zeros((0, 3)),
        spawn=spawn_pos,
        termination_reason=termination_reason,
        total_reward=total_reward,
        gates_passed=int(env._gates_passed[0] if hasattr(env, '_gates_passed') else 0),
    )
    print(f"\n  trajectory saved to {out_path}  ({len(positions)} steps)")


if __name__ == "__main__":
    main()
