# Changelog

All notable changes to `eigh` are documented in this file. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project uses semantic versioning.

## [0.3.0] - 2026-06-04

### Added
- **Multi-version JAX compatibility.** A single compiled wheel now works across
  **jax/jaxlib 0.5 → 0.9+** (verified on 0.5.3, 0.6.2, 0.7.0, 0.7.2, 0.8.3,
  0.9.2 — `eigh_gen` correct and twice-differentiable on all). The XLA FFI C ABI
  is stable within `XLA_FFI_API_MAJOR == 0`, so wheels are built against the
  oldest supported `jaxlib` (`jaxlib==0.5.*`) to stay forward-compatible.
- **Python 3.13 support.** Wheels now cover Python 3.10–3.13 via a hybrid
  strategy: version-specific `cp310`/`cp311` wheels plus a single `cp312-abi3`
  stable-ABI wheel that also serves 3.13+.
- CI regression guard (`compiled-ext-jax-matrix` in `tests.yml`) that builds one
  wheel and exercises the compiled handler against jax 0.5/0.6/0.7/0.8/0.9 — the
  test that would have caught the 0.7.0 breakage.
- README **Compatibility** section: Python/JAX matrix, CUDA (CUDA 12 only)
  support envelope, cluster/HPC notes, and GPU build-from-source instructions.
- **Prebuilt GPU wheels** via a new `eigh-cuda12` PyPI package (`pip install
  eigh-cuda12`), built from this same repository. Linux x86_64, CUDA 12, with the
  cuSOLVER kernel compiled in and NVIDIA CUDA runtime wheels as dependencies.
  Added `pyproject-cuda.toml`, `.github/workflows/publish-cuda.yml`, an
  `EIGH_REQUIRE_CUDA` CMake option (fails loudly if `nvcc` is missing so a GPU
  package can never ship a CPU-only binary), and an RPATH into the NVIDIA pip
  CUDA libs. Built in CI; functional GPU correctness verified on hardware before
  release (CI runners have no GPU).

### Changed
- Minimum supported JAX/jaxlib raised to **0.5.0** (was 0.4.0). The pre-FFI
  custom-call era (`jaxlib < 0.5`, e.g. 0.4.x) is no longer supported.
- Wheels build against `manylinux_2_28` (glibc 2.28) — RHEL 8+ / Ubuntu 18.04+.

### Fixed
- **`abi3` wheel was a no-op.** `wheel.py-api` was empty *and* the CMake
  `find_package(Python ...)` requested `Development.Module` without
  `Development.SABIModule`, so nanobind silently fell back to version-locked
  `cpython-3XX` modules despite `STABLE_ABI`. Added `Development.SABIModule` and
  a per-interpreter `cp312` abi3 override so a true `eigh_lapack.abi3.so` is now
  emitted on CPython ≥3.12.

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
