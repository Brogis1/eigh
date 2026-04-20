"""Forward-pass and gradient time vs matrix size, averaged over several seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from common import SOLVERS, benchmark, ensure_ready, well_conditioned


def fwd_fn(solver):
    def f(A, B):
        w, V = solver(A, B)
        return w, V
    return f


def grad_fn(solver):
    # Scalar loss so we can use jax.grad. Sum of eigenvalues + frob of V.
    def loss(A, B):
        w, V = solver(A, B)
        return w.sum() + (V * V).sum()
    return jax.grad(loss, argnums=(0, 1))


def run(sizes, seeds, n_iter, out_path):
    results = []
    for n in sizes:
        for seed in seeds:
            key = jax.random.PRNGKey(seed)
            A, B = well_conditioned(key, n)
            ensure_ready(A, B)
            for name, solver in SOLVERS.items():
                try:
                    fwd = benchmark(fwd_fn(solver), (A, B), n_iter=n_iter)
                    try:
                        grd = benchmark(grad_fn(solver), (A, B), n_iter=max(n_iter // 2, 5))
                        grad_ok = True
                    except Exception as e:
                        grd = {"mean_ms": float("nan"), "std_ms": 0.0, "min_ms": float("nan"), "samples": 0}
                        grad_ok = False
                    row = {
                        "solver": name, "n": n, "seed": seed,
                        "fwd_mean_ms": fwd["mean_ms"], "fwd_std_ms": fwd["std_ms"], "fwd_min_ms": fwd["min_ms"],
                        "grad_mean_ms": grd["mean_ms"], "grad_std_ms": grd["std_ms"], "grad_min_ms": grd["min_ms"],
                        "grad_ok": grad_ok,
                    }
                except Exception as e:
                    row = {"solver": name, "n": n, "seed": seed, "error": str(e)}
                print(row)
                results.append(row)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", type=int, nargs="+", default=[32, 64, 128, 256, 512, 1024])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--n-iter", type=int, default=20)
    p.add_argument("--out", type=Path, default=Path(__file__).parent / "results" / "scaling.json")
    args = p.parse_args()
    run(args.sizes, args.seeds, args.n_iter, args.out)


if __name__ == "__main__":
    main()
