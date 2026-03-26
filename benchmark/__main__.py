# benchmark/__main__.py
"""CLI for the golden set benchmark suite.

Usage:
    python -m benchmark run --checkpoint path/to/model.zip [options]
    python -m benchmark generate --config configs/golden_set/ [--upload]
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

log = logging.getLogger(__name__)


def cmd_run(args: argparse.Namespace) -> None:
    """Run benchmark against a checkpoint."""
    import wandb

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # env vars already set via --env-file or shell

    from benchmark.logger import log_benchmark_results
    from benchmark.runner import run_benchmark

    init_kwargs = {"project": args.wandb_project, "tags": ["golden-benchmark"]}
    if args.wandb_run_id:
        init_kwargs["config"] = {"linked_training_run": args.wandb_run_id}
    wandb.init(**init_kwargs)

    results = run_benchmark(
        checkpoint_path=args.checkpoint,
        golden_set_version=args.golden_set_version,
        golden_set_mode=args.golden_set_mode,
        local_golden_dir=Path(args.golden_set_dir) if args.golden_set_dir else None,
        save_trajectories=not args.no_trajectories,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        sim_config=args.sim_config,
    )

    log_benchmark_results(results)

    agg = results.aggregate
    print(f"\n{'='*60}")
    print(f"Benchmark Summary ({results.golden_set_mode})")
    print(f"{'='*60}")
    print(f"  Fastest lap:    {agg.fastest_lap or 'N/A'}")
    print(f"  Avg lap:        {agg.avg_lap or 'N/A'}")
    print(f"  Laps completed: {agg.laps_completed:.1f}")
    print(f"  Gates passed:   {agg.gates_passed:.1f}")
    print(f"  Crash rate:     {agg.crash_rate:.1%}")
    print(f"  Avg speed:      {agg.avg_speed:.1f} m/s")
    print(f"{'='*60}\n")

    wandb.finish()


def cmd_generate(args: argparse.Namespace) -> None:
    """Generate golden set .npz files from config YAMLs."""
    from omegaconf import OmegaConf

    from benchmark.track_generator import build_track_from_layout, serialize_track

    config_dir = Path(args.config)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    registry = OmegaConf.load(config_dir / "registry.yaml")

    for layout_entry in registry.layouts:
        name = layout_entry.name
        layout_path = config_dir / layout_entry.path
        log.info("Generating track: %s from %s", name, layout_path)

        track, meta = build_track_from_layout(layout_path)
        npz_path = output_dir / f"{name}.npz"
        serialize_track(track, npz_path, **meta)
        log.info("  Saved to %s (%d gates)", npz_path, track.num_gates)

    if args.upload:
        _upload_to_r2(output_dir, registry.version)

    print(f"\nGenerated {len(registry.layouts)} tracks to {output_dir}")


def cmd_leaderboard_refresh(args: argparse.Namespace) -> None:
    """Fetch benchmark data from W&B and upload leaderboard to R2."""
    import json
    import wandb

    log.info("Fetching benchmark runs from W&B...")
    api = wandb.Api()
    runs = api.runs(
        f"{args.wandb_entity}/{args.wandb_project}",
        filters={"tags": {"$in": ["golden-benchmark"]}},
    )

    def deep_dict(obj):
        if hasattr(obj, 'items'):
            return {k: deep_dict(v) for k, v in obj.items()}
        return obj

    edges = []
    for run in runs:
        summary = deep_dict(run.summary)
        config = deep_dict(run.config)
        edges.append({
            "node": {
                "name": run.id,
                "displayName": run.name,
                "createdAt": run.created_at,
                "tags": list(run.tags),
                "summaryMetrics": json.dumps(summary),
                "config": json.dumps(config),
            }
        })
        log.info("  %s (%s) tags=%s", run.name, run.id, run.tags)

    # Save JSON next to leaderboard HTML
    leaderboard_dir = Path(__file__).parent / "leaderboard"
    data_path = leaderboard_dir / "leaderboard-data.json"
    with open(data_path, "w") as f:
        json.dump(edges, f)
    log.info("Saved %d runs to %s", len(edges), data_path)

    # Upload both files to R2
    from artifacts.r2 import make_r2_client_from_env, upload_file, DEFAULT_BUCKET, R2_PUBLIC_BASE

    client = make_r2_client_from_env()

    for fname in ["index.html", "leaderboard-data.json"]:
        fpath = leaderboard_dir / fname
        ct = "text/html" if fname.endswith(".html") else "application/json"
        r2_key = f"leaderboard/{fname}"
        upload_file(client, fpath, DEFAULT_BUCKET, r2_key,
                    content_type=ct, skip_existing=False)
        log.info("  Uploaded: %s/%s", R2_PUBLIC_BASE, r2_key)

    print(f"\nLeaderboard refreshed with {len(edges)} runs.")
    print(f"View at: {R2_PUBLIC_BASE}/leaderboard/index.html")


def _upload_to_r2(output_dir: Path, version: str) -> None:
    """Upload generated .npz files to R2."""
    from artifacts.r2 import make_r2_client_from_env, upload_file, DEFAULT_BUCKET

    client = make_r2_client_from_env()

    for npz_file in sorted(output_dir.glob("*.npz")):
        r2_key = f"golden-set/{version}/{npz_file.name}"
        url = upload_file(client, npz_file, DEFAULT_BUCKET, r2_key,
                          content_type="application/octet-stream")
        if url:
            log.info("  Uploaded: %s", url)


def main() -> None:
    parser = argparse.ArgumentParser(description="Golden Set Benchmark Suite")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_p = subparsers.add_parser("run", help="Run benchmark against a checkpoint")
    run_p.add_argument("--checkpoint", required=True, help="Path to model .zip")
    run_p.add_argument("--golden-set-version", default="v1")
    run_p.add_argument("--golden-set-mode", default="holdout",
                       choices=["holdout", "trained"])
    run_p.add_argument("--wandb-run-id", default=None,
                       help="Link to original training run")
    run_p.add_argument("--wandb-project", default="corvidx-drone-racing")
    run_p.add_argument("--golden-set-dir", default=None,
                       help="Local directory with golden set .npz files (skips R2)")
    run_p.add_argument("--sim-config", default=None,
                       help="Path to sim config YAML for matching physics params")
    run_p.add_argument("--no-trajectories", action="store_true",
                       help="Skip trajectory recording (metrics only)")
    run_p.add_argument("--output-dir", default=None,
                       help="Output directory for trajectories and results")

    gen_p = subparsers.add_parser("generate", help="Generate golden set .npz files")
    gen_p.add_argument("--config", default="configs/golden_set/",
                       help="Path to golden set config directory")
    gen_p.add_argument("--output", default="outputs/golden_set/",
                       help="Output directory for .npz files")
    gen_p.add_argument("--upload", action="store_true",
                       help="Upload to R2 after generating")

    lb_p = subparsers.add_parser("leaderboard-refresh",
                                  help="Fetch W&B data and upload leaderboard to R2")
    lb_p.add_argument("--wandb-project", default="corvidx-drone-racing")
    lb_p.add_argument("--wandb-entity", default="janahanr-corvidx")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.command == "run":
        cmd_run(args)
    elif args.command == "generate":
        cmd_generate(args)
    elif args.command == "leaderboard-refresh":
        cmd_leaderboard_refresh(args)


if __name__ == "__main__":
    main()
