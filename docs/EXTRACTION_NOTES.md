# Extraction Notes

This document describes what was extracted from pyscfad and how it was adapted for standalone use.

## Source Repository

Extracted from: [pyscfad](https://github.com/fishjojo/pyscfad)

## What Was Extracted

### 1. Python Layer (JAX Integration)

**Source Files:**
- `pyscfad/backend/_jax/lax/linalg.py` → `src/python/eigh.py`
- `pyscfad/backend/_jax/scipy/linalg.py` → `src/python/eigh.py`

**Key Components:**
- `eigh_gen()` function - Core generalized eigenvalue solver
- `eigh()` function - Scipy-compatible interface
- JAX primitive `eigh_gen_p` with custom:
  - JVP rule (forward-mode differentiation)
  - Batching rule (vmap support)
  - CPU/GPU lowering rules

**Adaptations:**
- Changed namespace from `pyscfad` to `eigh`
- Updated imports to use local compiled modules (`eigh_lapack`, `eigh_cuda`)
- Made CUDA support optional with graceful fallback

### 2. CPU Implementation (LAPACK)

**Source Files:**
- `pyscfadlib/lapack/lapack_kernels.h` → `src/cpu/lapack_kernels.h`
- `pyscfadlib/lapack/lapack_kernels.cc` → `src/cpu/lapack_kernels.cc`
- `pyscfadlib/lapack/lapack.cc` → `src/cpu/lapack.cc`

**Key Components:**
- XLA FFI handlers for LAPACK routines:
  - `ssygvd` / `dsygvd` (real symmetric)
  - `chegvd` / `zhegvd` (complex Hermitian)
- Workspace size calculations
- Batch processing loops

**Adaptations:**
- Changed namespace from `pyscfad` to `eigh`
- Changed module name from `pyscfad_lapack` to `eigh_lapack`
- Updated include paths

### 3. GPU Implementation (CUDA)

**Source Files:**
- `pyscfadlib/cuda/solver_kernels.h` → `src/cuda/solver_kernels.h`
- `pyscfadlib/cuda/solver_kernels.cc` → `src/cuda/solver_kernels.cc`
- `pyscfadlib/cuda/solver.cc` → `src/cuda/solver.cc`

**Key Components:**
- XLA FFI handlers for cuSOLVER routines:
  - `cusolverDnSsygvd` / `cusolverDnDsygvd` (real)
  - `cusolverDnChegvd` / `cusolverDnZhegvd` (complex)
- CUDA stream management
- Device memory handling

**Adaptations:**
- Changed namespace from `pyscfad::cuda` to `eigh::cuda`
- Changed module name from `_solver` to `eigh_cuda`
- Simplified vendor.h to use standard CUDA headers
- Updated include paths

### 4. Utility Headers

**Source Files:**
- `pyscfadlib/ffi_helpers.h` → `include/ffi_helpers.h`
- `pyscfadlib/kernel_nanobind_helpers.h` → `include/kernel_nanobind_helpers.h`

**Key Components:**
- `SplitBatch2D()` - Extract batch dimensions
- `AllocateScratchMemory()` / `AllocateWorkspace()` - Memory management
- `MaybeCastNoOverflow()` - Safe type casting
- `EncapsulateFunction()` / `EncapsulateFfiHandler()` - Nanobind wrappers

**Adaptations:**
- Changed namespace from `pyscfad` to `eigh`

### 5. Tests

**Source Files:**
- `pyscfad/scipy/test/test_linalg.py` → `tests/test_eigh.py`

**Adaptations:**
- Expanded test coverage:
  - Basic functionality
  - Forward-mode differentiation
  - Generalized eigenvalue problems
  - Batched operations
  - Multiple dtypes
  - Device placement (CPU/GPU)
- Added device detection and graceful handling
- Made standalone executable

### 6. Build System

**New Files:**
- `CMakeLists.txt` - Unified CMake build
- `build.sh` - Automated build script
- `requirements.txt` - Python dependencies

**Original Build System:**
- pyscfad used separate CMake for LAPACK and Bazel for CUDA
- We unified to CMake for both

## Key Differences from Original

1. **Namespace**: `pyscfad` → `eigh`
2. **Module Names**:
   - `pyscfad_lapack` → `eigh_lapack`
   - `pyscfad_cuda12_plugin._solver` → `eigh_cuda`
3. **Build System**: Unified CMake instead of mixed CMake/Bazel
4. **CUDA Headers**: Direct includes instead of Bazel third_party
5. **Python Package**: Standalone module instead of full pyscfad package
6. **Optional CUDA**: Graceful fallback if CUDA not available

## Dependencies

### Build Time
- CMake 3.18+
- C++17 compiler
- nanobind
- BLAS/LAPACK libraries
- CUDA Toolkit (optional)

### Runtime
- JAX (jaxlib with XLA)
- NumPy
- Python 3.8+

## File Mapping

| Original File | Extracted To | Changes |
|---------------|--------------|---------|
| `pyscfad/backend/_jax/lax/linalg.py` | `src/python/eigh.py` | Namespace, imports |
| `pyscfadlib/lapack/lapack_kernels.h` | `src/cpu/lapack_kernels.h` | Namespace |
| `pyscfadlib/lapack/lapack_kernels.cc` | `src/cpu/lapack_kernels.cc` | Namespace |
| `pyscfadlib/lapack/lapack.cc` | `src/cpu/lapack.cc` | Namespace, module name |
| `pyscfadlib/cuda/solver_kernels.h` | `src/cuda/solver_kernels.h` | Namespace, includes |
| `pyscfadlib/cuda/solver_kernels.cc` | `src/cuda/solver_kernels.cc` | Namespace |
| `pyscfadlib/cuda/solver.cc` | `src/cuda/solver.cc` | Namespace, module name |
| `pyscfadlib/ffi_helpers.h` | `include/ffi_helpers.h` | Namespace |
| `pyscfadlib/kernel_nanobind_helpers.h` | `include/kernel_nanobind_helpers.h` | Namespace |

## Testing Strategy

The test suite validates:

1. **Correctness**: Compare against JAX reference implementation
2. **Differentiation**: Verify JVP matches JAX and finite differences
3. **Batching**: Test vmap functionality
4. **Dtypes**: All supported types (float32/64, complex64/128)
5. **Devices**: Both CPU and GPU (if available)

## License

Retains original Apache 2.0 license from pyscfad.

## Future Improvements

Potential enhancements not in this extraction:

1. **VJP Support**: Add reverse-mode differentiation
2. **Subset Eigenvalues**: Support `subset_by_index` / `subset_by_value`
3. **Advanced Drivers**: Support different LAPACK drivers
4. **Better Error Handling**: More descriptive error messages
5. **Performance**: Optimize batch processing on GPU

## References

- Original project: https://github.com/fishjojo/pyscfad
- JAX documentation: https://jax.readthedocs.io/
- XLA FFI guide: https://openxla.org/xla/ffi
