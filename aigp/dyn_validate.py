"""Load a dyn probe log, fit lumped dynamics, write sim_dynamics.json + report."""
from __future__ import annotations

import argparse
import json
import pathlib

from .dyn_fit import fit_from_log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", help="path to dyn probe log.jsonl")
    ap.add_argument("--n-motors", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    samples = [json.loads(l) for l in open(args.log)]
    out = fit_from_log(samples, n_motors=args.n_motors)
    dest = args.out or str(pathlib.Path(args.log).with_name("sim_dynamics.json"))
    with open(dest, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print("wrote", dest)


if __name__ == "__main__":
    main()
