# Differentiable Eigenvalue Decomposition

Standalone implementation of differentiable eigenvalue decomposition with CPU (LAPACK) and GPU (CUDA) backends. Extracted from [pyscfad](https://github.com/fishjojo/pyscfad).

## Features

- Generalized eigenvalue problems: `A @ V = B @ V @ diag(W)`
- Automatic differentiation via JAX (forward-mode JVP)
- CPU backend using LAPACK (ssygvd, dsygvd, chegvd, zhegvd)
- GPU backend using cuSOLVER (optional)
- Batched operations via vmap
- Support for float32, float64, complex64, complex128
- Degenerate eigenvalue handling with configurable threshold

## Quick Start

### Installation

```bash
cd eigh_standalone

# Build (auto-detects .venv Python)
./build.sh

# Or manually
mkdir build && cd build
cmake -DPython_EXECUTABLE=../.venv/bin/python ..
make
```

### Basic Usage

```python
import sys
sys.path.insert(0, 'eigh_standalone/src/python')

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)  # Enable float64 support

from eigh import eigh

# Standard eigenvalue problem
A = jnp.array([[1., 2.], [2., 1.]])
w, v = eigh(A)

# Generalized eigenvalue problem
B = jnp.eye(2)
w, v = eigh(A, B)

# With automatic differentiation
def loss(A):
    w, _ = eigh(A)
    return w.sum()

gradient = jax.grad(loss)(A)
```

## API Reference

### eigh(a, b=None, *, lower=True, eigvals_only=False, type=1, deg_thresh=1e-9)

Solve eigenvalue problem with scipy-compatible interface.

**Parameters:**
- `a` (array): Hermitian/symmetric matrix, shape `[..., n, n]`
- `b` (array, optional): Positive definite matrix for generalized problem
- `lower` (bool): Use lower triangle (default: True)
- `eigvals_only` (bool): Return only eigenvalues (default: False)
- `type` (int): Problem type - 1: `A@v = B@v@λ`, 2: `A@B@v = v@λ`, 3: `B@A@v = v@λ`
- `deg_thresh` (float): Degeneracy threshold for gradients (default: 1e-9)

**Returns:**
- `w` (array): Eigenvalues, shape `[..., n]`
- `v` (array): Eigenvectors, shape `[..., n, n]` (if `eigvals_only=False`)

### eigh_gen(a, b, *, lower=True, itype=1, deg_thresh=1e-9)

Lower-level interface for generalized eigenvalue problem.

**Parameters:**
- `a`, `b` (array): Input matrices, shape `[..., n, n]`
- `lower` (bool): Use lower triangle
- `itype` (int): Problem type (1, 2, or 3)
- `deg_thresh` (float): Degeneracy threshold

**Returns:**
- `w`, `v` (tuple): Eigenvalues and eigenvectors

## Degenerate Eigenvalues

When eigenvalues are **degenerate** (repeated), eigenvectors are **not unique**. Any linear combination of eigenvectors in the degenerate subspace is valid.

### Impact on Gradients

**Ill-Defined: Individual Eigenvalues**

```python
def smallest_eigenvalue(A):
    w, _ = eigh(A)
    return w.min()  # Gradient ill-defined if min is degenerate

grad = jax.grad(smallest_eigenvalue)(A)  # May give incorrect results
```

Small perturbations cause discontinuous changes in which eigenvector is returned, leading to unstable gradients.

**Well-Defined: Symmetric Functions**

Symmetric functions (invariant to eigenvalue permutation) have well-defined gradients:

```python
def trace(A):
    w, _ = eigh(A)
    return w.sum()  # d/dA[tr(A)] = I

def variance(A):
    w, _ = eigh(A)
    return jnp.var(w)  # Well-defined

def sum_of_squares(A):
    w, _ = eigh(A)
    return (w ** 2).sum()  # Well-defined
```

### The `deg_thresh` Parameter

Controls how close eigenvalues must be to be treated as degenerate:

```python
w, v = eigh(A, deg_thresh=1e-9)  # Default
```

When `|λᵢ - λⱼ| < deg_thresh`, the algorithm masks division by `(λᵢ - λⱼ)` to prevent numerical instability.

**Choosing the threshold:**
- Too small (1e-15): Numerical instability, large gradients
- Too large (1e-3): May incorrectly treat distinct eigenvalues as degenerate
- Default (1e-9): Good balance for most cases

```python
w, v = eigh(A, deg_thresh=1e-6)  # More robust for nearly degenerate
w, v = eigh(A, deg_thresh=1e-12)  # Higher precision for well-separated
```

### Testing Degenerate Cases

**DO test:**
- Eigenvalue equation: `A @ v = λ * v`
- Gradients of symmetric functions (trace, variance)
- Generalized problems: `A @ v = B @ v @ λ`

**DON'T test:**
- Gradients of individual degenerate eigenvalues
- Exact eigenvector matching (not unique)

Example:
```python
# Create matrix with eigenvalues [1, 1, 2, 2]
D = jnp.diag(jnp.array([1., 1., 2., 2.]))
Q, _ = np.linalg.qr(np.random.randn(4, 4))
A = Q @ D @ Q.T

w, v = eigh(A)

# Test eigenvalue equation
assert jnp.max(jnp.abs(A @ v - v @ jnp.diag(w))) < 1e-10

# Test gradient of trace
grad_trace = jax.grad(lambda A: eigh(A)[0].sum())(A)
assert jnp.allclose(grad_trace, jnp.eye(4), atol=1e-6)
```

## Build Requirements

**Build-time:**
- CMake 3.18+
- C++17 compiler
- nanobind
- BLAS/LAPACK libraries
- CUDA Toolkit (optional, for GPU support)

**Runtime:**
- Python 3.8+
- JAX with jaxlib (provides XLA headers)
- NumPy

Install Python dependencies:
```bash
pip install jax jaxlib nanobind numpy
```

## Testing

```bash
python tests/test_eigh.py
```

Tests cover:
- Correctness vs JAX reference implementation
- Forward-mode differentiation (JVP)
- Generalized eigenvalue problems
- Batched operations
- Multiple dtypes (float32/64, complex64/128)
- Degenerate eigenvalue cases
- CPU and GPU execution

## Implementation Details

### Architecture

```
Python API (eigh.py)
    ↓
JAX Primitive (eigh_gen_p)
    ↓
Platform Lowering
    ├─ CPU: LAPACK FFI (ssygvd/dsygvd/chegvd/zhegvd)
    └─ GPU: cuSOLVER FFI (cusolverDnSsygvd/Dsygvd/Chegvd/Zhegvd)
```

### Differentiation

Implements custom JVP (Jacobian-Vector Product) using first-order perturbation theory:

```python
dW = diag(V^H @ dA @ V - V^H @ dB @ V @ diag(W))
dV = V @ (F ⊙ (V^H @ dA @ V - V^H @ dB @ V / 2))
```

where `F[i,j] = 1/(W[i] - W[j])` for non-degenerate pairs, masked when `|W[i] - W[j]| < deg_thresh`.

### File Structure

```
eigh_standalone/
├── CMakeLists.txt          # Build configuration
├── build.sh                # Automated build script
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── example.py              # Usage examples
├── include/                # C++ headers
│   ├── ffi_helpers.h
│   └── kernel_nanobind_helpers.h
├── src/
│   ├── cpu/                # LAPACK backend
│   │   ├── lapack_kernels.{h,cc}
│   │   └── lapack.cc
│   ├── cuda/               # cuSOLVER backend
│   │   ├── solver_kernels.{h,cc}
│   │   └── solver.cc
│   └── python/             # JAX integration
│       └── eigh.py
├── tests/
│   ├── test_eigh.py        # Main test suite
│   └── test_eigh_hard.py   # Hard cases (ill-conditioned, degenerate)
└── docs/                   # Development notes
    ├── EXTRACTION_NOTES.md # Extraction from pyscfad
    ├── VERIFICATION.md     # Original code verification
    └── BUILD_FIXES.md      # Build system fixes
```

## Troubleshooting

### XLA Headers Not Found

```
fatal error: 'xla/ffi/api/c_api.h' file not found
```

**Solution:** XLA headers are in `jaxlib`, not `jax`. Ensure jaxlib is installed:
```bash
pip install jaxlib
```

The build system will automatically find headers at:
```
<venv>/lib/python3.x/site-packages/jaxlib/include/
```

### CUDA Not Detected

For CPU-only build:
```bash
cmake -DPython_EXECUTABLE=../.venv/bin/python \
      -DCMAKE_DISABLE_FIND_PACKAGE_CUDA=TRUE ..
```

### Import Errors

Ensure the module is in your Python path:
```python
import sys
sys.path.insert(0, '/path/to/eigh_standalone/src/python')
```

### Test Failures with float64

Enable JAX x64 mode:
```python
jax.config.update("jax_enable_x64", True)
```

## Performance

- **CPU**: Uses optimized LAPACK routines (MKL, OpenBLAS, etc.)
- **GPU**: Uses cuSOLVER with CUDA streams
- **Batching**: Processes batch elements sequentially (future: parallel GPU batching)
- **Memory**: Workspace size is `O(n²)` per matrix

## License

Apache License 2.0 (inherited from pyscfad)

```
Copyright 2021-2025 Xing Zhang

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
```

## Citation

If you use this code in research, please cite the original pyscfad project:

```bibtex
@software{pyscfad,
  author = {Zhang, Xing},
  title = {PySCFad: Automatic Differentiation for PySCF},
  url = {https://github.com/fishjojo/pyscfad},
  year = {2021-2025}
}
```

## References

- Original project: https://github.com/fishjojo/pyscfad
- JAX documentation: https://jax.readthedocs.io/
- XLA FFI guide: https://openxla.org/xla/ffi
- Magnus, J. R. (1985). "On differentiating eigenvalues and eigenvectors", Econometric Theory, 1(2), 179-191

## See Also

- [example.py](example.py) - Runnable examples
- [tests/test_eigh.py](tests/test_eigh.py) - Comprehensive test suite
- [tests/test_eigh_hard.py](tests/test_eigh_hard.py) - Hard cases: ill-conditioned and degenerate matrices


## Quick Start with Pip

### Installation

The project is now a standard Python package. You can install it directly from the source:

```bash
# Basic installation (CPU)
pip install .

# Installation with local CUDA support (recommended for this environment)
pip install .[cuda-local]

# Installation with bundled CUDA wheels
pip install .[cuda]
```

### Basic Usage

Once installed, you can use it like any other JAX-based library:

```python
import jax
import jax.numpy as jnp
from eigh import eigh

# Enable float64 support in JAX
jax.config.update("jax_enable_x64", True)

# Standard eigenvalue problem
A = jnp.array([[1., 2.], [2., 1.]])
w, v = eigh(A)

# Generalized eigenvalue problem (A @ v = B @ v @ w)
B = jnp.eye(2) + 0.1
w, v = eigh(A, B)

# Automatic differentiation
def loss(A):
    w, _ = eigh(A)
    return jnp.sum(w**2)

gradient = jax.grad(loss)(A)
```

## API Reference

### `eigh(a, b=None, *, lower=True, eigvals_only=False, type=1, deg_thresh=1e-9)`

Scipy-compatible interface for eigenvalue problems.

- **a**, **b**: Input matrices (Hermitian/Symmetric).
- **lower**: If True, use the lower triangle; otherwise upper.
- **eigvals_only**: If True, return only eigenvalues.
- **type**: Problem type (1: `A@v = B@v@λ`, 2: `A@B@v = v@λ`, 3: `B@A@v = v@λ`).
- **deg_thresh**: Threshold for treating eigenvalues as degenerate during differentiation.

### `eigh_gen(a, b, *, lower=True, itype=1, deg_thresh=1e-9)`

Lower-level interface for generalized eigenvalue problems (always returns both eigenvalues and eigenvectors).

## Testing

Run the comprehensive test suite to verify the installation:

```bash
# General tests
pytest tests/test_eigh.py

# Specialized eigh_gen tests (itype 1, 2, 3)
pytest tests/test_eigh_gen.py

# JIT and performance tests
pytest tests/test_eigh_jit.py
```

## File Structure

- `src/python/eigh/`: Core Python package.
  - `_core.py`: JAX primitive and FFI registration logic.
  - `__init__.py`: Public API export.
- `src/cpu/`: LAPACK C++ kernels.
- `src/cuda/`: cuSOLVER CUDA kernels.
- `include/`: C++/FFI helper headers.
- `tests/`: Extensive test suite covering JIT, vmap, and generalized types.
- `pyproject.toml`: Modern build configuration using `scikit-build-core`.

## GPU Environment Setup

If running on a cluster (e.g., via `ssh node07`), use the provided setup scripts:

```bash
source setup_gpu_env_clean.sh
./run_gpu.sh python example_simple.py
```

## License

Apache License 2.0 (inherited from [pyscfad](https://github.com/fishjojo/pyscfad)).
