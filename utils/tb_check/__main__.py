"""CLI entrypoint for the TensorBoard log checker.

Usage:
    python -m utils.tb_check <log_dir> [--compare <dir>] [--last <N>] [--diag-only]
"""

from __future__ import annotations

import argparse
import sys

from utils.tb_check import diagnostics, formatter, loader


def main() -> int:
    """CLI entrypoint for tb_check.

    Returns:
        0 on success, 1 on error.
    """
    parser = argparse.ArgumentParser(
        prog="python -m utils.tb_check",
        description="Analyze TensorBoard event logs for training diagnostics",
    )

    parser.add_argument(
        "log_dir",
        help="Path to TB log directory (finds event files recursively)",
    )

    parser.add_argument(
        "--compare",
        metavar="<dir>",
        default=None,
        help="Second TB log directory for side-by-side comparison",
    )

    parser.add_argument(
        "--last",
        metavar="<N>",
        type=int,
        default=None,
        help="Only analyze the last N timesteps",
    )

    parser.add_argument(
        "--diag-only",
        action="store_true",
        help="Skip the metric summary table, show only diagnostic findings",
    )

    args = parser.parse_args()

    # Load run(s)
    try:
        df1 = loader.load_run(args.log_dir, last_n=args.last)
    except Exception as e:
        print(f"Error loading log directory '{args.log_dir}': {e}", file=sys.stderr)
        return 1

    if args.compare:
        try:
            df2 = loader.load_run(args.compare, last_n=args.last)
        except Exception as e:
            print(
                f"Error loading comparison log directory '{args.compare}': {e}",
                file=sys.stderr,
            )
            return 1

    # Run diagnostics
    findings1 = diagnostics.run_diagnostics(df1)

    # Format and output
    if args.compare:
        findings2 = diagnostics.run_diagnostics(df2)
        if args.diag_only:
            output = "\n\n".join([
                f"── run1: {args.log_dir} ──",
                formatter.format_diagnostics_only(findings1),
                f"── run2: {args.compare} ──",
                formatter.format_diagnostics_only(findings2),
            ])
        else:
            output = formatter.format_comparison(
                df1, df2, args.log_dir, args.compare, findings1, findings2
            )
    elif args.diag_only:
        output = formatter.format_diagnostics_only(findings1)
    else:
        output = formatter.format_summary(df1, args.log_dir, findings1)

    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
