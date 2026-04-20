"""Run every benchmark then generate all plots. Pass --quick for a tiny smoke run."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent


def run(cmd):
    print(f"\n$ {' '.join(str(c) for c in cmd)}")
    subprocess.run(cmd, check=True, cwd=HERE)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="tiny run for smoke-testing")
    args = p.parse_args()
    py = sys.executable

    if args.quick:
        run([py, "bench_scaling.py", "--sizes", "32", "64", "--seeds", "0", "--n-iter", "5"])
        run([py, "bench_accuracy.py", "--n", "32", "--seeds", "0",
             "--conds", "1e4", "1e8", "--gaps", "1e-4", "1e-10"])
        run([py, "bench_gradient.py", "--n", "32", "--seeds", "0",
             "--gaps", "1e-4", "1e-10", "--conds", "1e4", "1e10"])
    else:
        run([py, "bench_scaling.py"])
        run([py, "bench_accuracy.py"])
        run([py, "bench_gradient.py"])

    run([py, "plot.py"])
    print("\nDone. See suite/figs/.")


if __name__ == "__main__":
    main()
