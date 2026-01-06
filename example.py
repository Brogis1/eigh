#!/usr/bin/env python
"""
Simple example demonstrating the standalone eigh implementation.

This example shows:
1. Basic eigenvalue decomposition
2. Generalized eigenvalue problem
3. Automatic differentiation
4. Batched operations
"""

import sys
import os

# Add the module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'python'))

import jax
import jax.numpy as jnp

# Enable 64-bit precision for full dtype support
jax.config.update("jax_enable_x64", True)

from eigh import eigh

print("=" * 60)
print("Standalone Eigh Example")
print("=" * 60)

# Example 1: Basic eigenvalue decomposition
print("\n1. Basic Eigenvalue Decomposition")
print("-" * 60)

A = jnp.array([[4., 1., 2.],
               [1., 5., 3.],
               [2., 3., 6.]])

eigenvalues, eigenvectors = eigh(A)

print(f"Matrix A:\n{A}\n")
print(f"Eigenvalues: {eigenvalues}")
print(f"Eigenvectors:\n{eigenvectors}\n")

# Verify: A @ v = lambda * v
for i in range(3):
    v = eigenvectors[:, i]
    lam = eigenvalues[i]
    residual = jnp.linalg.norm(A @ v - lam * v)
    print(f"  Eigenvalue {i}: λ={lam:.6f}, residual={residual:.2e}")

# Example 2: Generalized eigenvalue problem
print("\n2. Generalized Eigenvalue Problem")
print("-" * 60)

A = jnp.array([[2., 1.], [1., 2.]])
B = jnp.array([[1., 0.5], [0.5, 1.]])

w, v = eigh(A, B)

print(f"Matrix A:\n{A}\n")
print(f"Matrix B:\n{B}\n")
print(f"Eigenvalues: {w}")
print(f"Eigenvectors:\n{v}\n")

# Verify: A @ v = B @ v @ diag(w)
for i in range(2):
    vi = v[:, i]
    wi = w[i]
    lhs = A @ vi
    rhs = B @ vi * wi
    residual = jnp.linalg.norm(lhs - rhs)
    print(f"  Generalized eigenvalue {i}: λ={wi:.6f}, residual={residual:.2e}")

# Example 3: Automatic differentiation
print("\n3. Automatic Differentiation")
print("-" * 60)

def trace_eigenvalues(A):
    """Sum of eigenvalues (same as matrix trace)."""
    w, _ = eigh(A)
    return w.sum()

A = jnp.array([[3., 1.], [1., 2.]])

# Compute function value
value = trace_eigenvalues(A)

# Compute gradient
grad_fn = jax.grad(trace_eigenvalues)
gradient = grad_fn(A)

print(f"Matrix A:\n{A}\n")
print(f"Sum of eigenvalues: {value:.6f}")
print(f"Gradient of sum w.r.t. A:\n{gradient}\n")

# The gradient of trace(eigenvalues) should be close to identity
# because trace is linear and trace(A) = sum(eigenvalues)
print(f"Gradient should be close to identity (for trace):")
print(f"  Max deviation: {jnp.max(jnp.abs(gradient - jnp.eye(2))):.2e}")

# Example 4: Batched operations
print("\n4. Batched Operations")
print("-" * 60)

# Create a batch of 5 matrices
batch_size = 5
A_batch = jnp.stack([
    jnp.eye(3) + jnp.ones((3, 3)) * i
    for i in range(batch_size)
])

print(f"Batch of {batch_size} matrices (3x3 each)")
print(f"Input shape: {A_batch.shape}\n")

# Solve all at once
w_batch, v_batch = eigh(A_batch)

print(f"Output eigenvalues shape: {w_batch.shape}")
print(f"Output eigenvectors shape: {v_batch.shape}\n")

print("Eigenvalues for each matrix in batch:")
for i in range(batch_size):
    print(f"  Matrix {i}: {w_batch[i]}")

# Example 5: Different dtypes
print("\n5. Different Data Types")
print("-" * 60)

dtypes = [
    (jnp.float32, "float32"),
    (jnp.float64, "float64"),
    (jnp.complex64, "complex64"),
    (jnp.complex128, "complex128"),
]

A_real = jnp.array([[2., 1.], [1., 2.]])
A_complex = jnp.array([[2.+0j, 1.-1j], [1.+1j, 2.+0j]])

for dtype, name in dtypes:
    if "complex" in name:
        A = A_complex.astype(dtype)
    else:
        A = A_real.astype(dtype)

    w, v = eigh(A)
    print(f"  {name:12s}: eigenvalues dtype={w.dtype}, eigenvectors dtype={v.dtype}")

# Summary
print("\n" + "=" * 60)
print("Example Complete!")
print("=" * 60)
print("\nFor more examples and documentation, see:")
print("  - README.md       (Full API reference)")
print("  - QUICKSTART.md   (Quick start guide)")
print("  - tests/test_eigh.py  (Comprehensive tests)")
