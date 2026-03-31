"""Deceleration expert pipeline — train, merge, freeze+warmup, benchmark.

Phase 1: Wait for decel expert training to finish
Phase 2: Merge into 5-expert MoE as expert 5 (28→56 dim expansion)
Phase 3: Freeze+warmup training (40M steps)
Phase 4: Benchmark on golden set

Usage:
    nohup python -m autoresearch.decel_pipeline > decel_pipeline.log 2>&1 &
"""
from __future__ import annotations

import glob
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

STATE_DIR = Path("autoresearch/state")
BENCHMARK_SIM_CONFIG = "configs/sim/numpy_quad_2gate_hist.yaml"
GOLDEN_SET_DIR = "outputs/golden_set/"
MOE_CHECKPOINT = "outputs/2026-03-29/20-46-14/final_model.zip"
DECEL_TRAINING_LOG = "training_decel_expert.log"


def wait_for_decel_training() -> str | None:
    """Wait for decel expert training to complete."""
    log.info("Waiting for decel expert training to complete...")
    while True:
        time.sleep(120)
        try:
            with open(DECEL_TRAINING_LOG, "r") as f:
                content = f.read()
            if "Final model saved to" in content:
                for line in content.split("\n"):
                    if "Final model saved to" in line:
                        checkpoint = line.split("Final model saved to")[-1].strip()
                        log.info("Decel training complete: %s", checkpoint)
                        return checkpoint + ".zip"
            if "Traceback" in content and "Final model saved" not in content:
                log.error("Decel training crashed!")
                return None
            # Progress
            for line in reversed(content.strip().split("\n")):
                if "total_timesteps" in line and "|" in line:
                    log.info("Progress: %s", line.strip().strip("|").strip())
                    break
        except FileNotFoundError:
            log.warning("Log not found, waiting...")


def merge_expert(decel_checkpoint: str) -> str:
    """Run checkpoint surgery to merge decel expert into MoE."""
    output = "outputs/merged_6expert/model.zip"
    log.info("Merging decel expert into MoE...")
    log.info("  MoE: %s", MOE_CHECKPOINT)
    log.info("  Decel: %s", decel_checkpoint)

    cmd = [
        sys.executable, "-m", "control.merge_decel_expert",
        "--moe-checkpoint", MOE_CHECKPOINT,
        "--decel-checkpoint", decel_checkpoint,
        "--output", output,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        log.error("Merge failed: %s", result.stderr[-1000:])
        return ""

    log.info("Merge output:\n%s", result.stdout[-500:] if result.stdout else result.stderr[-500:])
    return output


def create_training_config(merged_checkpoint: str) -> str:
    """Create experiment config for freeze+warmup training."""
    config_path = "configs/experiment/moe_6expert_warmup.yaml"

    config = f"""# @package _global_
# 6-expert MoE with deceleration specialist — freeze+warmup
# COR-85

defaults:
  - override /sim: numpy_quad
  - override /control: ppo_moe
  - override /perception: corner_noise
  - override /domain_rand: uniform_30pct
  - override /logging: wandb
  - _self_

sim:
  action_mode: trpy
  n_lookahead_gates: 2
  n_action_history: 7
  max_steps: 1200
  arena_bounds: 10
  gate_passage_radius: 1.0
  gate_collision: true
  start_vel_std: 1.5

total_timesteps: 40_000_000
resume_checkpoint: {merged_checkpoint.replace('.zip', '')}

control:
  moe: true
  n_experts: 6
  expert_hidden_dim: 128
  top_k: 2
  balance_coef: 0.01
  ent_coef: 0.005
  learning_rate: 0.0003
  log_std_init: -1.0

track_gen:
  n_gates_min: 4
  n_gates_max: 10
  gate_spacing_min: 1.0
  gate_spacing_max: 6.0
  turn_angle_min: -170
  turn_angle_max: 170
  elevation_min: 0.5
  elevation_max: 4.0
  elevation_delta_max: 2.0
  elevation_bias: 0.3
  elevation_oscillation: 0.5
  closure_max_angle: 120
  closure_max_retries: 30
  figure8_ratio: 0.25
  zigzag_ratio: 0.15
  figure8:
    loop_radius_min: 1.5
    loop_radius_max: 3.0
    gates_per_loop_min: 3
    gates_per_loop_max: 4
    crossing_offset_min: 0.3
    crossing_offset_max: 0.6
    elevation_min: 1.0
    elevation_max: 3.0
    elevation_delta_max: 0.8
  zigzag:
    turn_angle_min: 100
    turn_angle_max: 160
    n_gates_min: 5
    n_gates_max: 8
    gate_spacing_min: 1.0
    gate_spacing_max: 4.0
    elevation_min: 0.5
    elevation_max: 4.0
    elevation_delta_max: 1.5

perception:
  noise_scale: 1.0
  dropout_onset: null
  dropout_continuation: 0.7

reward:
  weights:
    gate_passage: 100.0
    gate_progress: 3.0
    gate_offset: 3.0
    body_rate: 0.001
    action_smoothness: 0.0
    crash_penalty: 5.0
    spline_proximity: 1.0
    heading_alignment: 0.5
    speed_bonus: 0.0
    spline_speed: 3.0
    boundary_penalty: 2.0
    gate_approach: 0.1
    gate_centering: 0.5
    gate_speed_penalty: 2.0
  v_max: 7.5
  action_smoothness_threshold: 0.5

arpo:
  enabled: false
curriculum: null

expert_warmup:
  enabled: true
  new_expert_idx: 5
  usage_threshold: 0.15
  check_freq: 1_000_000
  max_frozen_steps: 5_000_000
  patience: 2

multi_scene:
  enabled: true
  n_scenes: 10

autopilot:
  enabled: true
  check_freq: 5_000_000
  patience: 5
  degrade_threshold: 0.3

logging:
  tags: ["trpy", "moe", "6expert", "decel", "action-history", "freeze-warmup"]
  group: moe_6expert
  notes: "6-expert MoE with decel specialist. 56-dim obs (action history). Freeze+warmup. gate_speed_penalty=2.0."
"""

    with open(config_path, "w") as f:
        f.write(config)
    log.info("Config written to %s", config_path)
    return config_path


def run_training(config_name: str) -> str | None:
    """Run freeze+warmup training."""
    log.info("=" * 60)
    log.info("TRAINING: 6-expert MoE freeze+warmup (40M steps)")
    log.info("=" * 60)

    cmd = [
        sys.executable, "-m", "training",
        f"+experiment={config_name}",
        "logging.notes=6expert_decel_warmup",
    ]
    log.info("Command: %s", " ".join(cmd))

    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start

    log.info("Finished in %.0fs (exit code %d)", elapsed, result.returncode)

    if result.returncode != 0:
        log.error("FAILED: %s", result.stderr[-1500:])
        return None

    today = datetime.now().strftime("%Y-%m-%d")
    checkpoints = sorted(glob.glob(f"outputs/{today}/*/final_model.zip"))
    return checkpoints[-1] if checkpoints else None


def run_benchmark(checkpoint: str, label: str) -> dict:
    """Run golden set benchmark."""
    log.info("Benchmarking %s (%s)...", checkpoint, label)

    # Need to update benchmark sim config for action history
    # Create a temporary config with n_action_history
    import yaml
    with open(BENCHMARK_SIM_CONFIG) as f:
        sim_cfg = yaml.safe_load(f) if hasattr(yaml, 'safe_load') else {}

    # Use the existing config but the benchmark runner reads sim params
    cmd = [
        sys.executable, "-m", "benchmark", "run",
        "--checkpoint", checkpoint,
        "--golden-set-mode", "holdout",
        "--golden-set-dir", GOLDEN_SET_DIR,
        "--sim-config", BENCHMARK_SIM_CONFIG,
        "--no-trajectories",
        "--wandb-project", "corvidx-drone-racing",
    ]
    env = {**os.environ, "WANDB_MODE": "offline"}
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    summary = {"label": label, "checkpoint": checkpoint}
    if result.returncode != 0:
        log.error("Benchmark failed: %s", result.stderr[-500:])
        summary["error"] = True
        return summary

    for line in result.stdout.split("\n"):
        if "Gates passed:" in line:
            summary["gates"] = float(line.split(":")[1].strip())
        elif "Crash rate:" in line:
            summary["crash_rate"] = line.split(":")[1].strip()
        elif "Laps completed:" in line:
            summary["laps"] = float(line.split(":")[1].strip())
        elif "Fastest lap:" in line:
            summary["fastest_lap"] = line.split(":")[1].strip()
        elif "Avg speed:" in line:
            summary["avg_speed"] = line.split(":")[1].strip()

    log.info("  gates=%.1f, crash=%s, laps=%.1f, speed=%s",
             summary.get("gates", 0), summary.get("crash_rate", "?"),
             summary.get("laps", 0), summary.get("avg_speed", "?"))
    return summary


def main():
    log.info("=" * 70)
    log.info("DECELERATION EXPERT PIPELINE — COR-85")
    log.info("=" * 70)

    results = {"timestamp": datetime.now().isoformat()}

    # Phase 1: Wait for decel training
    log.info("\n>>> PHASE 1: Wait for decel expert training")
    decel_checkpoint = wait_for_decel_training()
    if not decel_checkpoint:
        log.error("Decel training failed. Stopping.")
        return
    results["decel_checkpoint"] = decel_checkpoint

    # Phase 2: Merge
    log.info("\n>>> PHASE 2: Merge decel expert into MoE")
    merged = merge_expert(decel_checkpoint)
    if not merged:
        log.error("Merge failed. Stopping.")
        return
    results["merged_checkpoint"] = merged

    # Phase 3: Create config and train
    log.info("\n>>> PHASE 3: Freeze+warmup training (40M steps)")
    config_path = create_training_config(merged)
    config_name = Path(config_path).stem
    final_checkpoint = run_training(config_name)
    if not final_checkpoint:
        log.error("Training failed. Stopping.")
        return
    results["final_checkpoint"] = final_checkpoint

    # Phase 4: Benchmark
    log.info("\n>>> PHASE 4: Golden set benchmark")
    benchmark = run_benchmark(final_checkpoint, "6-expert MoE with decel")
    results["benchmark"] = benchmark

    # Save
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    results_path = STATE_DIR / "decel_pipeline_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Summary
    log.info("\n" + "=" * 70)
    log.info("DECEL PIPELINE SUMMARY")
    log.info("=" * 70)
    log.info("Previous best (5-expert): 7.7 gates, 6%% crash, 0.6 laps")
    b = results.get("benchmark", {})
    log.info("6-expert with decel: gates=%s, crash=%s, laps=%s, speed=%s",
             b.get("gates", "?"), b.get("crash_rate", "?"),
             b.get("laps", "?"), b.get("avg_speed", "?"))
    log.info("Results saved to %s", results_path)


if __name__ == "__main__":
    main()
