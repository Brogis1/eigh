# Eigensolver benchmarking suite (CPU)

Extensive CPU benchmarks for the generalized eigensolvers in `src/jax/`.
The small script at `benchmarks/compare_performance_eigensolvers.py` stays for
quick checks; this suite produces the figures for blog / LinkedIn posts.

## Running

```sh
python suite/run_all.py --quick     # smoke test
python suite/run_all.py             # full run
python suite/plot.py                # just re-render plots
```

Results land in `results/*.json`; figures in `figs/*.{png,svg}`.

## Plots, and why each one exists

All x-axes on sweeps use log scale. Each solver gets a distinct marker +
linestyle + color so overlapping curves remain readable.

### Scaling — `figs/scaling_*.png`

**Why.** The headline story for a blog post: "how fast is each solver as n
grows?" Covers both the forward call and the reverse-mode gradient, because
training cost is fwd + bwd.

- `scaling_fwd.png` — forward pass time (ms) vs matrix size n.
- `scaling_grad.png` — `jax.grad` time (ms) vs matrix size n.

### Accuracy vs conditioning of B — `figs/acc_cond_*.png`

**Why.** The overlap matrix B in DFT can be badly conditioned. A fast solver
that silently loses digits on ill-conditioned B is useless. Ground truth here
is `scipy.linalg.eigh` on float64.

- `acc_cond_eval_err.png` — eigenvalue error vs condition number of B.
- `acc_cond_subspace_err.png` — subspace angle error vs condition number of B.
- `acc_cond_residual.png` — normalized residual vs condition number of B.

### Accuracy vs degenerate spectrum — `figs/acc_gap_*.png`

**Why.** Degenerate or near-degenerate eigenvalues are the classical failure
mode for `eigh`-based solvers. This sweep makes pairs of eigenvalues get closer
and closer; a robust solver should keep its error flat.

- `acc_gap_eval_err.png` — eigenvalue error vs eigenvalue-pair gap in A.
- `acc_gap_subspace_err.png` — subspace angle error vs eigenvalue-pair gap.
- `acc_gap_residual.png` — normalized residual vs eigenvalue-pair gap.

### Gradient behavior — `figs/grad_*.png`

**Why.** This is why the "safe" / "subspace" / "stable" variants exist at all:
standard reverse-mode `eigh` blows up or NaNs on degenerate spectra. We track
gradient norm and NaN rate across the same two sweeps. Caveat: no ground-truth
gradient, these show *stability*, not correctness.

- `grad_gap_norm.png` — gradient norm vs eigenvalue-pair gap.
- `grad_gap_nan.png` — NaN rate of gradient vs eigenvalue-pair gap.
- `grad_cond_norm.png` — gradient norm vs condition number of B.
- `grad_cond_nan.png` — NaN rate of gradient vs condition number of B.

## Metrics (y-axes)

- **Eigenvalue error** — `max_i |λ_solver,i - λ_scipy,i|` after sorting.
- **Subspace angle** — largest principal angle between solver eigenvectors and
  scipy eigenvectors (in radians). Invariant to sign / phase / in-subspace
  rotations, which is the physically meaningful notion in DFT.
- **Normalized residual** — `||A V - B V diag(w)|| / ||A||`.
- **Gradient norm** — `||dLoss/dA||_2` for a density-matrix-style loss.
- **NaN rate** — fraction of seeds where `dLoss/dA` contains any NaN.

## Setup

- JAX pinned to CPU (`jax_platform_name=cpu`), float64.
- Warmup: 3 JIT calls before timing; `block_until_ready` per call.
- Each point = mean over seeds; error bars on scaling plots are std across seeds.
