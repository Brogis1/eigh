# Changelog

All notable changes to `eigh` are documented in this file. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project uses semantic versioning.

## [0.2.0] - 2026-04-20

### Added
- New family of JAX-differentiable generalized eigensolvers under `src/jax/`:
  - `generalized_eigensolver.py`: `standard_eig`, `jax_eig`, `generalized_eigh`,
    `degen_eigh`, `safe_generalized_eigh`, `subspace_eigh`, `subspace_generalized_eigh`.
  - `generalized_eigensolver_stable.py`: `stable_eigh` / `stable_generalized_eigh`
    (pure-JAX Cholesky + Lorentzian-broadened custom VJP).
  - `generalized_eigensolver_pyscfad.py`: `stable_eigh_pyscfad` /
    `stable_eigh_gen_pyscfad` — **recommended** solvers that wrap the fast
    LAPACK/cuSOLVER-backed `eigh` / `eigh_gen` kernels with a
    Lorentzian-broadened VJP for stable gradients at (near-)degeneracies.
- Benchmark suite under `benchmarks/suite/` covering forward/backward scaling,
  accuracy vs. condition number and eigenvalue gap, and gradient stability
  (`bench_scaling.py`, `bench_accuracy.py`, `bench_gradient.py`, `plot.py`,
  `run_all.py`). Pre-rendered figures in `benchmarks/suite/figs/`.
- README: benchmark plots, solver comparison table, verified references
  ([Kasim & Vinko 2021], [Colburn & Majumdar 2021], JAX issues #2748, #5461),
  and self-citation entry.

### Changed
- Docstring of `generalized_eigensolver.py` generalized beyond Kohn–Sham DFT.
- README wording to describe pipelines generally rather than DFT-specifically.

### Fixed
- CI publish workflow for wheel builds on Linux + macOS.

### Packaging
- `requires-python = ">=3.10"`; dropped Python 3.9 classifier (jaxlib
  compatibility — cibuildwheel already skipped 3.9 in [0.1.x]).

## [0.1.2] - 2026-02-03
- Fix logo rendering on PyPI; metadata polish.

## [0.1.1] - 2026-01-xx
- Metadata / authors fixes; manylinux wheel workflow stabilization.

## [0.1.0] - 2026-01-06
- Initial PyPI release: differentiable `eigh` / `eigh_gen` with LAPACK (CPU)
  and cuSOLVER (GPU) backends, extracted and repackaged from
  [pyscfad](https://github.com/fishjojo/pyscfad). Rename of FFI target symbols
  to avoid clashing with JAX's standard `eigh`.
