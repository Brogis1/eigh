"""Gradient stability: magnitude and NaN-rate of dLoss/dA across degenerate spectra
and ill-conditioned B. This is the plot that motivates the 'stable' solvers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from common import (
    SOLVERS, ensure_ready, degenerate_spectrum, ill_conditioned,
)


def make_loss(solver):
    # Loss that depends on eigenvectors — this is where the gradient gets hairy
    # near degeneracies. Project onto a fixed random vector to get a scalar.
    def loss(A, B, target):
        w, V = solver(A, B)
        # density-matrix style observable: sum of occupied projectors acting on target
        occ = V.shape[-1] // 2
        P = V[:, :occ] @ V[:, :occ].T
        return (target @ P @ target)
    return loss


def grad_stats(solver, A, B, target):
    loss = make_loss(solver)
    grad = jax.jit(jax.grad(loss, argnums=0))
    try:
        g = grad(A, B, target)
        ensure_ready(g)
        g = np.asarray(g)
        return {
            "grad_norm": float(np.linalg.norm(g)),
            "grad_max": float(np.max(np.abs(g))),
            "has_nan": bool(np.any(np.isnan(g))),
            "has_inf": bool(np.any(np.isinf(g))),
        }
    except Exception as e:
        return {"grad_norm": float("nan"), "grad_max": float("nan"),
                "has_nan": True, "has_inf": False, "error": str(e)}


def run_degeneracy(n, gaps, seeds, out_path):
    results = []
    for gap in gaps:
        for seed in seeds:
            key = jax.random.PRNGKey(seed)
            A, B = degenerate_spectrum(key, n, gap=gap)
            target = jax.random.normal(jax.random.PRNGKey(seed + 1000), (n,))
            ensure_ready(A, B, target)
            for name, solver in SOLVERS.items():
                stats = grad_stats(solver, A, B, target)
                row = {"solver": name, "n": n, "gap": gap, "seed": seed, **stats}
                print(row)
                results.append(row)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out_path}")


def run_ill_cond(n, conds, seeds, out_path):
    results = []
    for cond in conds:
        for seed in seeds:
            key = jax.random.PRNGKey(seed)
            A, B = ill_conditioned(key, n, cond=cond)
            target = jax.random.normal(jax.random.PRNGKey(seed + 1000), (n,))
            ensure_ready(A, B, target)
            for name, solver in SOLVERS.items():
                stats = grad_stats(solver, A, B, target)
                row = {"solver": name, "n": n, "cond_B": cond, "seed": seed, **stats}
                print(row)
                results.append(row)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=128)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--gaps", type=float, nargs="+",
                   default=[1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14])
    p.add_argument("--conds", type=float, nargs="+",
                   default=[1e2, 1e4, 1e6, 1e8, 1e10, 1e12])
    p.add_argument("--outdir", type=Path, default=Path(__file__).parent / "results")
    args = p.parse_args()
    run_degeneracy(args.n, args.gaps, args.seeds, args.outdir / "gradient_degenerate.json")
    run_ill_cond(args.n, args.conds, args.seeds, args.outdir / "gradient_cond.json")


if __name__ == "__main__":
    main()
