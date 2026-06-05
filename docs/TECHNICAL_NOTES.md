# Technical Notes

Background detail on packaging, ABI compatibility, and building eigh from source.
The [README](../README.md) has the quick-start; this document explains the *why*
and covers the edge cases (pinned/old `jaxlib`, HPC clusters, CUDA targets).

For the full, blow-by-blow packaging playbook (every pitfall hit while building
the CPU + CUDA wheels), see [MANUAL.md](../MANUAL.md).

---

## Python & JAX compatibility

The compiled CPU (LAPACK) handler registers through the **XLA FFI C API**, whose
ABI is stable within `XLA_FFI_API_MAJOR == 0`. A single wheel built against the
oldest supported `jaxlib` therefore loads on every newer one — verified working
on **jax/jaxlib 0.5.3, 0.6.2, 0.7.0, 0.7.2, 0.8.3, 0.9.2, and 0.10** (`eigh_gen`
correct and twice-differentiable on all of them).

| Python | JAX 0.5 / 0.6 | JAX 0.7+ | Wheel shipped |
| --- | --- | --- | --- |
| 3.10 | ✅ | — *(jax 0.7 dropped 3.10)* | `cp310` |
| 3.11 | ✅ | ✅ | `cp311` |
| 3.12 | ✅ | ✅ | `cp312-abi3` |
| 3.13+ | — | ✅ | `cp312-abi3` *(forward-compatible)* |

**Why three CPU wheels, not one?** nanobind's stable ABI (`abi3`) exists only
from CPython 3.12, and `abi3` is *forward*-compatible only (a 3.12-built wheel
runs on 3.12/3.13+, never on 3.10/3.11). So 3.10 and 3.11 get version-specific
wheels and 3.12 ships one `abi3` wheel that also serves 3.13+.

**jaxlib floor.** The prebuilt wheels require `jaxlib >= 0.5` (they are built
against jaxlib 0.5, and the FFI binary ABI is forward-only — see below). Older
`jaxlib` (including 0.4.29) is supported **via source build**, not the wheel.

---

## CUDA / GPU

GPU support uses the classic dense **cuSOLVER** routines
(`cusolverDn{S,D}sygvd`, `cusolverDn{C,Z}hegvd`) dispatched through JAX's GPU FFI
under `platform="gpu"`.

| Aspect | Support | Notes |
| --- | --- | --- |
| CUDA major version | **CUDA 12 only** | JAX ≥0.5 ships only `cuda12` plugins (`jax[cuda12]`). CUDA 11 would need `jax[cuda11]`, dropped in modern JAX. |
| Prebuilt wheels | **`eigh-cuda120`** (CUDA 12.0, glibc 2.17) and **`eigh-cuda128`** (CUDA 12.8, glibc 2.34) | `eigh-cuda120` is forward-compatible to 12.8+ and the safer default; `eigh-cuda128` targets modern toolchains. |
| cuSOLVER API | CUDA 8+ | The `*sygvd`/`*hegvd` dense API is long-stable; no upper CUDA-12 bound from the API surface. |
| Compute capability | nvcc default for the toolkit | No explicit `-arch`; PTX JITs forward to newer GPUs. Set `CMAKE_CUDA_ARCHITECTURES` to target a specific SM. |
| FFI ABI across JAX | `MAJOR == 0`, forward-only | The GPU handler is forward-compatible across jax 0.5→0.10, same as CPU. |
| Source-build jaxlib range | **jaxlib 0.4.29 and 0.5 → 0.10+** | Source builds work against the cluster's own jaxlib (incl. the 0.4.29 some CUDA-12.0 clusters pin); accessor/registration differences are bridged in [`include/ffi_helpers.h`](../include/ffi_helpers.h) and [`_core.py`](../src/python/eigh/_core.py). |

**Why separate `eigh-cuda120` / `eigh-cuda128` packages?** A single normal PyPI
package cannot serve a small CPU wheel to CPU users *and* GPU wheels to GPU
users (PyPI wheel-variants are not GA), and one GPU wheel can only target one
CUDA version. So CPU users `pip install eigh`; GPU users pick the matching
`eigh-cudaXXX`. All are built from this one repo — mirroring `jaxlib` vs
`jax-cuda12-plugin`.

**Things to know:**
- `pip install eigh` is **CPU-only**. Pairing it with `jax[cuda12]` does *not*
  enable GPU — the CUDA kernel must be compiled into the wheel, and the CPU wheel
  doesn't contain it. Use `eigh-cuda120`/`eigh-cuda128`, or build from source.
- GPU wheels are built in CI but **functionally tested only on real GPU hardware**
  before a release (CI has no GPU). CI only checks the extension loads.
- A GPU build is **not** as portable as the CPU wheel: it must match the
  cluster's CUDA 12 runtime and driver.
- The build auto-detects CUDA via `check_language(CUDA)`; set
  `-C cmake.define.EIGH_REQUIRE_CUDA=ON` to make a missing `nvcc` a hard error
  instead of silently producing a CPU-only build.

---

## The XLA FFI binary ABI (why old `jaxlib` needs a source build)

The XLA FFI register struct **changed size at the jaxlib 0.4 → 0.5 boundary**
(28 bytes on ≤0.4.x, 76 bytes on ≥0.5), and that boundary is **not crossable in
either direction**:

- A wheel built against jaxlib **0.5** fails on jaxlib ≤ ~0.4.33 at registration
  with `XlaRuntimeError: Unexpected XLA_FFI_Handler_Register size: expected 76,
  got 28`.
- A wheel built against jaxlib **0.4.29** *registers* on jaxlib ≥0.5 but then
  **segfaults** when the kernel is called (the newer runtime hands the old
  handler a larger call frame than it expects).

Within a generation it *is* forward-compatible — a 0.5-built handler runs on
0.5 → 0.10+ (verified). But you cannot build one binary that serves both jax 0.4
and jax 0.5+. This is why:

- the prebuilt wheels require `jaxlib >= 0.5`, and
- **jax 0.4.29 is supported via source build only** — building against the
  cluster's own jaxlib produces a `.so` whose ABI matches that exact runtime.

eigh bridges the API differences between jaxlib generations internally:
- **C++ FFI buffer accessors** (`typed_data()` ↔ `.data`, `dimensions()` ↔
  `.dimensions`, `element_count()`, `element_type()`, `size_bytes()`,
  `Error::InvalidArgument`, the `ffi::F32`/`DataType::F32` dtype names) are
  resolved at compile time in [`include/ffi_helpers.h`](../include/ffi_helpers.h)
  via the detection idiom — identical codegen on modern jaxlib, member-access
  fallback on 0.4.29.
- **Python-side registration** (`jax.ffi.register_ffi_target` on 0.5+,
  `jax.extend.ffi.register_ffi_target` on 0.4.3x,
  `xla_client.register_custom_call_target` on ≤0.4.30) is bridged in
  [`_core.py`](../src/python/eigh/_core.py).

---

## Clusters / HPC

- **glibc:** the CPU wheel targets `manylinux_2_28` (glibc 2.28) — RHEL 8+,
  Ubuntu 18.04+, most current HPC. The GPU wheels target glibc 2.17
  (`eigh-cuda120`) and 2.34 (`eigh-cuda128`). Older or mismatched glibc → build
  from source. If `pip install` reports *"no matching distribution"* even though
  the package exists, the wheel's glibc/CUDA is newer than the node accepts;
  check `ldd --version` and `nvidia-smi` (driver caps the usable CUDA version)
  and pick the matching variant or build from source.
- **BLAS/LAPACK:** `auditwheel` bundles OpenBLAS + libgfortran into the Linux CPU
  wheel, so nodes **do not** need a system BLAS installed.
- **Pinned old jaxlib (e.g. 0.4.29+cuda12):** build from source with
  `--no-build-isolation --no-deps` so pip does not upgrade your working jax. See
  the README's source-build section. Verified on jaxlib 0.4.29+cuda12, CUDA 12.0,
  driver 525, RHEL 8.

---

## Windows

No prebuilt wheel. The pure-JAX solvers in [src/jax/](../src/jax/) (e.g.
`safe_generalized_eigh`, `subspace_generalized_eigh`, `stable_generalized_eigh`)
work out-of-the-box — `pip install jax numpy scipy` and import from that module.
The fast LAPACK/cuSOLVER-backed `eigh` / `eigh_gen` kernels (and therefore
`stable_eigh_pyscfad` / `stable_eigh_gen_pyscfad`) require building from source
against a local BLAS/LAPACK.
