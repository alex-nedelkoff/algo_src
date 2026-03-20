"""Durable monitoring process for training runs.

Runs independently of Claude Code session. Monitors a W&B training run,
refreshes coordination claims, and records results to state files on completion.

Usage:
    nohup python -m autoresearch.runner \
        --run-id <wandb_run_id> \
        --claim-id <hypothesis_id> \
        --state-dir autoresearch/state/ &
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("autoresearch.runner")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse runner CLI arguments."""
    parser = argparse.ArgumentParser(description="Auto-research durable monitor")
    parser.add_argument("--run-id", required=True, help="W&B run ID to monitor")
    parser.add_argument("--claim-id", required=True, help="Hypothesis ID for claim refresh")
    parser.add_argument("--wandb-project", default="corvidx-drone-racing")
    parser.add_argument("--state-dir", default="autoresearch/state/")
    parser.add_argument("--poll-interval", type=int, default=60, help="Seconds between polls")
    parser.add_argument("--claim-refresh-minutes", type=int, default=15)
    parser.add_argument("--max-rpm", type=float, default=31470.0)
    parser.add_argument("--control-freq", type=float, default=100.0)
    parser.add_argument("--budget", type=int, default=5_000_000)
    parser.add_argument("--baseline-fitness", type=float, default=None)
    return parser.parse_args(argv)


def refresh_claim(state_dir: Path, claim_id: str) -> None:
    """Refresh the timestamp on an active claim."""
    from autoresearch.coordination.claims import load_claims, save_claims

    claims_path = state_dir / "claims.json"
    claims = load_claims(claims_path)
    for c in claims:
        if c.hypothesis_id == claim_id and c.status == "running":
            c.timestamp = datetime.now(timezone.utc)
    save_claims(claims, claims_path)


def run_monitor(args: argparse.Namespace) -> None:
    """Main monitoring loop. Polls W&B, refreshes claims, checks early stop.

    This is a skeleton for Phase 2 — full W&B polling requires the wandb SDK.
    Currently only handles claim refresh.
    """
    state_dir = Path(args.state_dir)
    last_claim_refresh = time.time()
    claim_refresh_interval = args.claim_refresh_minutes * 60

    logger.info(f"Monitoring W&B run {args.run_id} for claim {args.claim_id}")
    logger.info(f"Poll interval: {args.poll_interval}s, budget: {args.budget}")

    while True:
        try:
            if time.time() - last_claim_refresh >= claim_refresh_interval:
                refresh_claim(state_dir, args.claim_id)
                last_claim_refresh = time.time()
                logger.info("Claim timestamp refreshed")

            # TODO: Query W&B API for run status and metrics
            # TODO: Check early stopping signals via check_early_stop()
            # TODO: On completion, compute descriptors and validate constraints

            time.sleep(args.poll_interval)

        except KeyboardInterrupt:
            logger.info("Runner interrupted")
            break
        except Exception:
            logger.exception("Runner error, will retry")
            time.sleep(args.poll_interval)


def main(argv: list[str] | None = None) -> None:
    """Entry point for `python -m autoresearch.runner`."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    args = parse_args(argv)
    run_monitor(args)


if __name__ == "__main__":
    main()
