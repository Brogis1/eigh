# Quick Start Guide

Get up and running with the standalone eigh implementation in 5 minutes.

## 1. Install Dependencies

```bash
# Install Python packages
pip install jax jaxlib nanobind numpy

# On macOS with Homebrew (for BLAS/LAPACK)
brew install openblas lapack

# On Ubuntu/Debian
sudo apt-get install libblas-dev liblapack-dev

# On RHEL/CentOS/Fedora
sudo yum install blas-devel lapack-devel
```

## 2. Build

```bash
cd eigh_standalone
./build.sh
```

Or manually:

```bash
mkdir build && cd build
cmake ..
make
```

## 3. Test

```bash
python tests/test_eigh.py
```

## 4. Use

```python
import sys
sys.path.insert(0, 'src/python')

from eigh import eigh
import jax.numpy as jnp

# Create a simple symmetric matrix
A = jnp.array([[1., 2.], [2., 1.]])

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = eigh(A)

print("Eigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)

# Verify: A @ v = lambda * v
for i in range(2):
    v = eigenvectors[:, i]
    lam = eigenvalues[i]
    assert jnp.allclose(A @ v, lam * v)
    print(f"✓ Eigenvalue {i} verified")
```

## 5. With Automatic Differentiation

```python
import jax
from eigh import eigh

def loss_fn(A):
    eigenvalues, _ = eigh(A)
    return eigenvalues.sum()

# Compute gradient
A = jnp.ones((3, 3))
grad_fn = jax.grad(loss_fn)
gradient = grad_fn(A)

print("Gradient:\n", gradient)
```

## Troubleshooting

### "Module not found"

Make sure to add the Python module to your path:

```python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'python'))
```

### Build fails with XLA headers not found

The build needs JAX's XLA headers. Install JAX first:

```bash
pip install jax jaxlib
```

### CUDA not detected

If you have CUDA but it's not detected:

```bash
cmake -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc ..
```

Or build CPU-only:

```bash
cmake -DCMAKE_DISABLE_FIND_PACKAGE_CUDA=TRUE ..
```

## Next Steps

- Read the full [README.md](README.md) for detailed API documentation
- Check out the [test suite](tests/test_eigh.py) for more examples
- Explore batched operations with `jax.vmap`
