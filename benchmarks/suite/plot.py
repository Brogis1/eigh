"""Generate plots from results/*.json. Saves PNG + SVG to figs/."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
RESULTS = HERE / "results"
FIGS = HERE / "figs"

MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
LINESTYLES = ["-", "--", "-.", ":"]
COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def style(i):
    return {
        "marker": MARKERS[i % len(MARKERS)],
        "linestyle": LINESTYLES[i % len(LINESTYLES)],
        "color": COLORS[i % len(COLORS)],
        "markersize": 7,
        "markerfacecolor": "none",
        "markeredgewidth": 1.5,
        "linewidth": 1.5,
        "alpha": 0.85,
    }


def load(name):
    path = RESULTS / name
    if not path.exists():
        print(f"skip {name}: not found")
        return None
    return json.loads(path.read_text())


def agg(rows, x_key, y_key, skip_nan=True):
    buckets = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if "error" in r or y_key not in r:
            continue
        v = r[y_key]
        if skip_nan and (v is None or (isinstance(v, float) and np.isnan(v))):
            continue
        buckets[r["solver"]][r[x_key]].append(v)
    out = {}
    for solver, xmap in buckets.items():
        xs = sorted(xmap)
        means = [float(np.mean(xmap[x])) for x in xs]
        stds = [
            float(np.std(xmap[x], ddof=1)) if len(xmap[x]) > 1 else 0.0
            for x in xs
        ]
        out[solver] = (xs, means, stds)
    return out


def save(fig, name):
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / f"{name}.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIGS / f"{name}.svg", bbox_inches="tight")
    print(f"wrote figs/{name}.png")


def _plot_series(ax, data, errorbar=False):
    for i, solver in enumerate(sorted(data)):
        xs, means, stds = data[solver]
        means = np.array(means)
        stds = np.array(stds)
        s = style(i)
        if errorbar:
            ax.errorbar(xs, means, yerr=stds, label=solver, capsize=3, **s)
        else:
            ax.plot(xs, means, label=solver, **s)


def plot_scaling():
    rows = load("scaling.json")
    if not rows:
        return
    for y_key, title, ylabel, fname in [
        (
            "fwd_mean_ms",
            "Forward-pass time vs matrix size (CPU, float64)",
            "forward-pass wall time per call (ms)",
            "scaling_fwd",
        ),
        (
            "grad_mean_ms",
            "Gradient (jax.grad) time vs matrix size (CPU, float64)",
            "gradient wall time per call (ms)",
            "scaling_grad",
        ),
    ]:
        data = agg(rows, "n", y_key)
        fig, ax = plt.subplots(figsize=(7, 5))
        _plot_series(ax, data, errorbar=True)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("matrix size n (n x n generalized eigenproblem)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        save(fig, fname)
        plt.close(fig)


def plot_accuracy():
    sweeps = [
        (
            "accuracy_cond.json",
            "cond_B",
            "condition number of B, cond(B) = λ_max / λ_min",
            "acc_cond",
        ),
        (
            "accuracy_degenerate.json",
            "gap",
            "eigenvalue-pair gap in A (smaller = more degenerate)",
            "acc_gap",
        ),
    ]
    metrics = [
        (
            "eval_err",
            "eigenvalue error vs scipy: max_i |λ_solver,i - λ_scipy,i|",
        ),
        (
            "subspace_err",
            "subspace angle vs scipy eigenvectors (radians)",
        ),
        (
            "residual",
            "normalized residual  ||A V - B V diag(w)|| / ||A||",
        ),
    ]
    for rows_name, x_key, x_label, fname_prefix in sweeps:
        rows = load(rows_name)
        if not rows:
            continue
        for y_key, ylabel in metrics:
            data = agg(rows, x_key, y_key)
            fig, ax = plt.subplots(figsize=(7, 5))
            _plot_series(ax, data)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(x_label)
            ax.set_ylabel(ylabel)
            ax.set_title(ylabel)
            ax.grid(True, which="both", alpha=0.3)
            ax.legend()
            save(fig, f"{fname_prefix}_{y_key}")
            plt.close(fig)


def plot_gradient():
    sweeps = [
        (
            "gradient_degenerate.json",
            "gap",
            "eigenvalue-pair gap in A (smaller = more degenerate)",
            "grad_gap",
        ),
        (
            "gradient_cond.json",
            "cond_B",
            "condition number of B, cond(B) = λ_max / λ_min",
            "grad_cond",
        ),
    ]
    for rows_name, x_key, x_label, fname_prefix in sweeps:
        rows = load(rows_name)
        if not rows:
            continue
        data = agg(rows, x_key, "grad_norm")
        fig, ax = plt.subplots(figsize=(7, 5))
        _plot_series(ax, data)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(x_label)
        ax.set_ylabel("gradient magnitude  ||dLoss / dA||_2")
        ax.set_title("Gradient magnitude (stability, not correctness)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        save(fig, f"{fname_prefix}_norm")
        plt.close(fig)

        nan_rate = defaultdict(lambda: defaultdict(list))
        for r in rows:
            nan_rate[r["solver"]][r[x_key]].append(
                1.0 if r.get("has_nan") else 0.0
            )
        fig, ax = plt.subplots(figsize=(7, 5))
        for i, solver in enumerate(sorted(nan_rate)):
            xs = sorted(nan_rate[solver])
            ys = [np.mean(nan_rate[solver][x]) for x in xs]
            ax.plot(xs, ys, label=solver, **style(i))
        ax.set_xscale("log")
        ax.set_xlabel(x_label)
        ax.set_ylabel("fraction of seeds where dLoss/dA contains NaN")
        ax.set_title("Gradient NaN rate")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend()
        save(fig, f"{fname_prefix}_nan")
        plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--what",
        choices=["all", "scaling", "accuracy", "gradient"],
        default="all",
    )
    args = p.parse_args()
    if args.what in ("all", "scaling"):
        plot_scaling()
    if args.what in ("all", "accuracy"):
        plot_accuracy()
    if args.what in ("all", "gradient"):
        plot_gradient()


if __name__ == "__main__":
    main()
