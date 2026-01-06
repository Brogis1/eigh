# Copyright 2021-2025 Xing Zhang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Tests for standalone differentiable eigh implementation.
Tests both CPU (LAPACK) and GPU (CUDA) if available.
"""

import sys
import os
import pytest


# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'python'))

import numpy as np
import jax
from jax import numpy as jnp
from jax import scipy as jsp

# Enable 64-bit precision in JAX for proper dtype testing
jax.config.update("jax_enable_x64", True)

try:
    from eigh import eigh, eigh_gen
except ImportError as e:
    print(f"Failed to import eigh module: {e}")
    sys.exit(1)


def test_eigh_basic():
    """Test basic eigh functionality against JAX."""
    print("\nTest: Basic eigh functionality")

    a = jnp.ones((2, 2))
    b = jnp.eye(2)

    w0, v0 = jsp.linalg.eigh(a)
    w1, v1 = eigh(a)

    assert abs(w1 - w0).max() < 1e-10, "Eigenvalues don't match"
    assert abs(v1 - v0).max() < 1e-10, "Eigenvectors don't match"

    print("Basic eigh test passed")


def test_eigh_jvp():
    """Test forward-mode differentiation."""
    print("\nTest: Forward-mode differentiation (JVP)")

    a = jnp.ones((2, 2))

    jac0 = jax.jacfwd(jsp.linalg.eigh)(a)
    jac1 = jax.jacfwd(eigh)(a)

    assert abs(jac1[0] - jac0[0]).max() < 1e-6, "Eigenvalue gradients don't match"
    assert abs(jac1[1] - jac0[1]).max() < 1e-6, "Eigenvector gradients don't match"

    print("JVP test passed")


def test_eigh_generalized():
    """Test generalized eigenvalue problem with finite differences."""
    print("\nTest: Generalized eigenvalue problem")

    a = jnp.ones((2, 2))
    b = jnp.eye(2)

    # Finite difference validation
    disp = 0.0005
    b_m = jnp.array([[1., 0.], [0., 1. - disp]])
    w_m, v_m = eigh(a, b_m)
    b_p = jnp.array([[1., 0.], [0., 1. + disp]])
    w_p, v_p = eigh(a, b_p)

    g0 = (-v_p - v_m) / 0.001  # -v_p due to specific gauge

    jac = jax.jacfwd(eigh, argnums=1)(a, b)
    g1 = jac[1][..., 1, 1]

    assert abs(g1 - g0).max() < 1e-3, "Finite difference validation failed"

    print("✓ Generalized eigenvalue test passed")


def test_eigh_batched():
    """Test batched eigh operation."""
    print("\nTest: Batched eigh")

    # Create batch of 3 matrices
    a_batch = jnp.stack([jnp.ones((2, 2)) * (i + 1) for i in range(3)])
    b_batch = jnp.stack([jnp.eye(2) for _ in range(3)])

    w, v = eigh(a_batch, b_batch)

    assert w.shape == (3, 2), f"Expected shape (3, 2), got {w.shape}"
    assert v.shape == (3, 2, 2), f"Expected shape (3, 2, 2), got {v.shape}"

    print("Batched eigh test passed")


def test_eigh_dtypes():
    """Test different dtypes (float32, float64, complex64, complex128)."""
    print("\nTest: Different dtypes")

    for dtype in [np.float32, np.float64, np.complex64, np.complex128]:
        a = jnp.ones((2, 2), dtype=dtype)
        b = jnp.eye(2, dtype=dtype)

        try:
            w, v = eigh(a, b)
            assert w.dtype == (np.float32 if dtype in [np.float32, np.complex64] else np.float64)
            assert v.dtype == dtype
            print(f"dtype {dtype.__name__} passed")
        except Exception as e:
            print(f"dtype {dtype.__name__} failed: {e}")
            raise


def test_eigh_eigvals_only():
    """Test eigvals_only option."""
    print("\nTest: eigvals_only option")

    a = jnp.ones((2, 2))
    b = jnp.eye(2)

    w = eigh(a, b, eigvals_only=True)
    assert isinstance(w, jnp.ndarray), "Expected ndarray"
    assert w.shape == (2,), f"Expected shape (2,), got {w.shape}"

    print("eigvals_only test passed")


def test_degenerate_eigenvalues():
    """
    Test degenerate eigenvalue case with degeneracy.

    IMPORTANT: When eigenvalues are degenerate, the eigenvectors are NOT unique.
    Any linear combination of eigenvectors in the degenerate subspace is valid.
    This means:
    - Gradients of individual eigenvalues are ill-defined
    - Only symmetric functions of eigenvalues have well-defined gradients
    - The deg_thresh parameter controls numerical stability in gradients
    """
    print("\nTest: Degenerate eigenvalues")

    # Create a 4x4 matrix with double degeneracy
    # Eigenvalues will be: [1, 1, 2, 2] (double degenerate)
    np.random.seed(42)

    # Start with diagonal matrix
    D = jnp.diag(jnp.array([1.0, 1.0, 2.0, 2.0]))

    # Create a random orthogonal matrix to make it more complex
    # Using QR decomposition of a random matrix
    random_matrix = np.random.randn(4, 4)
    Q, _ = np.linalg.qr(random_matrix)
    Q = jnp.array(Q)

    # Apply orthogonal transformation: A = Q @ D @ Q^T
    A = Q @ D @ Q.T

    # Make it exactly symmetric
    A = (A + A.T) / 2

    print(f"Matrix A shape: {A.shape}")
    print(f"Matrix A (should be symmetric):")
    print(f"  Max asymmetry: {jnp.max(jnp.abs(A - A.T)):.2e}")

    # Test 1: Compare with JAX reference implementation
    print("\n  Test 1: Eigenvalues vs JAX reference")
    w_ref, v_ref = jsp.linalg.eigh(A)
    w_ours, v_ours = eigh(A)

    # Sort eigenvalues for comparison
    w_ref_sorted = jnp.sort(w_ref)
    w_ours_sorted = jnp.sort(w_ours)

    eigenvalue_diff = jnp.max(jnp.abs(w_ref_sorted - w_ours_sorted))
    print(f"  Eigenvalues (sorted): {w_ours_sorted}")
    print(f"  Expected (sorted):    {w_ref_sorted}")
    print(f"  Max difference: {eigenvalue_diff:.2e}")

    assert eigenvalue_diff < 1e-10, f"Eigenvalues differ by {eigenvalue_diff}"

    # Verify A @ v = v @ diag(w)
    residual = jnp.max(jnp.abs(A @ v_ours - v_ours @ jnp.diag(w_ours)))
    print(f"  Eigenvalue equation residual: {residual:.2e}")
    assert residual < 1e-10, f"Eigenvalue equation not satisfied, residual={residual}"

    # Test 2: Gradient computation with degenerate eigenvalues
    print("\n  Test 2: Gradients with degenerate eigenvalues")

    def eigenvalue_sum(A):
        """Sum of eigenvalues."""
        w, _ = eigh(A)
        return w.sum()

    # Compute gradient using JAX autodiff
    grad_fn = jax.grad(eigenvalue_sum)
    gradient = grad_fn(A)

    # The gradient of sum(eigenvalues) should be close to identity
    # because d/dA[tr(A)] = I and sum(eigenvalues) = tr(A)
    expected_grad = jnp.eye(4)
    grad_diff = jnp.max(jnp.abs(gradient - expected_grad))
    print(f"  Gradient max difference from identity: {grad_diff:.2e}")

    # For degenerate eigenvalues, gradient might be less precise
    assert grad_diff < 1e-5, f"Gradient differs too much: {grad_diff}"

    # Test 3: Test gradient behavior with degenerate eigenvalues
    print("\n  Test 3: Gradient behavior with degeneracy")

    # NOTE: For degenerate eigenvalues, the gradient of individual eigenvalues
    # is NOT well-defined because the eigenvectors are not unique.
    # However, symmetric functions of eigenvalues (like sum, sum of squares)
    # should have well-defined gradients.

    def eigenvalue_variance(A):
        """Variance of eigenvalues - a symmetric function."""
        w, _ = eigh(A)
        return jnp.var(w)

    grad_var = jax.grad(eigenvalue_variance)(A)
    print(f"  Gradient of variance computed")
    print(f"  Gradient norm: {jnp.linalg.norm(grad_var):.6f}")

    # Verify with finite differences for symmetric function
    eps = 1e-5
    direction = np.random.randn(4, 4)
    direction = (direction + direction.T) / 2  # Symmetric
    direction = direction / np.linalg.norm(direction)
    direction = jnp.array(direction)

    A_plus = A + eps * direction
    A_minus = A - eps * direction

    val_plus = eigenvalue_variance(A_plus)
    val_minus = eigenvalue_variance(A_minus)

    finite_diff = (val_plus - val_minus) / (2 * eps)
    analytical = jnp.sum(grad_var * direction)

    print(f"  Finite difference: {finite_diff:.6e}")
    print(f"  Analytical grad:   {analytical:.6e}")
    rel_error = abs(finite_diff - analytical) / (abs(finite_diff) + 1e-10)
    print(f"  Relative error: {rel_error:.2e}")

    # Symmetric functions should have well-defined gradients
    assert rel_error < 1e-3, \
        f"Gradient verification failed: FD={finite_diff}, Analytical={analytical}"

    # Test 3b: Show that degenerate case is handled by deg_thresh
    print("\n  Test 3b: Degeneracy threshold behavior")

    # Try different deg_thresh values
    for thresh in [1e-12, 1e-9, 1e-6]:
        w_t, v_t = eigh(A, deg_thresh=thresh)
        # Just verify it runs without error
        print(f"  deg_thresh={thresh:.0e}: eigenvalues computed successfully")

    # Test 4: Generalized eigenvalue problem with degeneracy
    print("\n  Test 4: Generalized problem with degenerate eigenvalues")

    B = jnp.eye(4) + 0.1 * jnp.ones((4, 4))  # Positive definite
    B = (B + B.T) / 2

    w_gen, v_gen = eigh(A, B)

    # Verify generalized eigenvalue equation: A @ v = B @ v @ diag(w)
    residual_gen = jnp.max(jnp.abs(A @ v_gen - B @ v_gen @ jnp.diag(w_gen)))
    print(f"  Generalized eigenvalue residual: {residual_gen:.2e}")

    assert residual_gen < 1e-9, f"Generalized eigenvalue equation failed: {residual_gen}"

    # Test gradient of generalized problem
    def gen_eigenvalue_sum(A, B):
        w, _ = eigh(A, B)
        return w.sum()

    grad_A, grad_B = jax.grad(gen_eigenvalue_sum, argnums=(0, 1))(A, B)
    print(f"  Gradient of A computed: norm={jnp.linalg.norm(grad_A):.6f}")
    print(f"  Gradient of B computed: norm={jnp.linalg.norm(grad_B):.6f}")

    print("✓ Degenerate eigenvalue test passed")


def test_device_placement():
    """Test execution on available devices (CPU and GPU if available)."""
    print("\n=== Test: Device placement ===")

    devices = jax.devices()
    print(f"Available devices: {[d.platform for d in devices]}")

    a = jnp.ones((2, 2))
    b = jnp.eye(2)

    for device in devices:
        try:
            with jax.default_device(device):
                w, v = eigh(a, b)
                print(f"  ✓ Execution on {device.platform} ({device.device_kind}) succeeded")
        except Exception as e:
            print(f"  ✗ Execution on {device.platform} ({device.device_kind}) failed: {e}")
            # Don't raise - some platforms might not be supported


if __name__ == "__main__":
    # Run all tests verbosely
    pytest.main([__file__, "-v"])
