"""Accuracy sweeps: eigenvalue error, subspace error, residual — vs cond(B)
and vs degeneracy gap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import numpy as np

from common import (
    SOLVERS, ensure_ready, reference_solution, eigenvalue_error,
    subspace_error, residual, ill_conditioned, degenerate_spectrum,
)


def eval_solver(solver, A, B):
    w, V = jax.jit(solver)(A, B)
    ensure_ready(w, V)
    return np.asarray(w), np.asarray(V)


def run_cond_sweep(n, conds, seeds, out_path):
    results = []
    for cond in conds:
        for seed in seeds:
            key = jax.random.PRNGKey(seed)
            A, B = ill_conditioned(key, n, cond=cond)
            ensure_ready(A, B)
            w_ref, V_ref = reference_solution(A, B)
            for name, solver in SOLVERS.items():
                try:
                    w, V = eval_solver(solver, A, B)
                    row = {
                        "solver": name, "n": n, "cond_B": cond, "seed": seed,
                        "eval_err": eigenvalue_error(w, w_ref),
                        "subspace_err": subspace_error(V, V_ref),
                        "residual": residual(A, B, w, V),
                    }
                except Exception as e:
                    row = {"solver": name, "n": n, "cond_B": cond, "seed": seed, "error": str(e)}
                print(row)
                results.append(row)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out_path}")


def run_degeneracy_sweep(n, gaps, seeds, out_path):
    results = []
    for gap in gaps:
        for seed in seeds:
            key = jax.random.PRNGKey(seed)
            A, B = degenerate_spectrum(key, n, gap=gap)
            ensure_ready(A, B)
            w_ref, V_ref = reference_solution(A, B)
            for name, solver in SOLVERS.items():
                try:
                    w, V = eval_solver(solver, A, B)
                    row = {
                        "solver": name, "n": n, "gap": gap, "seed": seed,
                        "eval_err": eigenvalue_error(w, w_ref),
                        "subspace_err": subspace_error(V, V_ref),
                        "residual": residual(A, B, w, V),
                    }
                except Exception as e:
                    row = {"solver": name, "n": n, "gap": gap, "seed": seed, "error": str(e)}
                print(row)
                results.append(row)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=128)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--conds", type=float, nargs="+",
                   default=[1e2, 1e4, 1e6, 1e8, 1e10, 1e12])
    p.add_argument("--gaps", type=float, nargs="+",
                   default=[1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12])
    p.add_argument("--outdir", type=Path, default=Path(__file__).parent / "results")
    args = p.parse_args()
    run_cond_sweep(args.n, args.conds, args.seeds, args.outdir / "accuracy_cond.json")
    run_degeneracy_sweep(args.n, args.gaps, args.seeds, args.outdir / "accuracy_degenerate.json")


if __name__ == "__main__":
    main()
