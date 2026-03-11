"""CLI entry point: python -m sim.viz <path> [--output-dir <dir>] [--camera-decimation <int>]"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m sim.viz",
        description="Convert .npz trajectory files to rerun .rrd archives.",
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to a .npz file or directory of .npz files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for .rrd files (default: sibling rerun/ directory)",
    )
    parser.add_argument(
        "--camera-decimation",
        type=int,
        default=10,
        help="Render pinhole camera every N steps (default: 10)",
    )

    args = parser.parse_args()
    path: Path = args.path

    if not path.exists():
        print(f"Error: path does not exist: {path}", file=sys.stderr)
        sys.exit(1)

    # Guard rerun import — give clear error before doing any work
    try:
        import rerun  # noqa: F401
    except ImportError:
        print(
            "Error: rerun-sdk is required but not installed.\n"
            "Install it with:  pip install 'rerun-sdk>=0.22'\n"
            "Or install the viz extras:  pip install -e '.[viz]'",
            file=sys.stderr,
        )
        sys.exit(1)

    from sim.viz.rerun_generator import batch_generate, generate_rrd

    if path.is_file() and path.suffix == ".npz":
        rrd_path = generate_rrd(
            path,
            output_path=args.output_dir / path.with_suffix(".rrd").name
            if args.output_dir
            else None,
            camera_decimation=args.camera_decimation,
        )
        print(f"Generated: {rrd_path}")
    elif path.is_dir():
        batch_generate(
            path,
            output_dir=args.output_dir,
            camera_decimation=args.camera_decimation,
        )
    else:
        print(
            f"Error: {path} is not a .npz file or directory",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
