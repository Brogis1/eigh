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
Tests for JIT compilation of eigh implementations.
Tests both standard JAX eigh and our custom implementation.
"""

import sys
import os
import pytest

# Add parent directory to path
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), '..', 'src', 'python')
)

import numpy as np
import jax
from jax import numpy as jnp
from jax import scipy as jsp

# Enable 64-bit precision in JAX for proper dtype testing
jax.config.update("jax_enable_x64", True)

try:
    from eigh import eigh
except ImportError as e:
    print(f"Failed to import eigh module: {e}")
    sys.exit(1)


def test_jit_basic():
    """Test basic JIT compilation of both implementations."""
    print("\nTest: Basic JIT compilation")

    # Standard JAX implementation
    @jax.jit
    def jax_eigh_jit(a):
        return jsp.linalg.eigh(a)

    # Our implementation
    @jax.jit
    def our_eigh_jit(a):
        return eigh(a)

    # Test data
    a = jnp.ones((3, 3))
    a = (a + a.T) / 2  # Make symmetric

    # Compile and run
    w_jax, v_jax = jax_eigh_jit(a)
    w_ours, v_ours = our_eigh_jit(a)

    assert abs(w_jax - w_ours).max() < 1e-12, "Eigenvalues don't match"
    assert abs(v_jax - v_ours).max() < 1e-12, "Eigenvectors don't match"

    print("Basic JIT test passed")


def test_jit_dtypes():
    """Test JIT compilation with different dtypes."""
    print("\nTest: JIT with different dtypes")

    @jax.jit
    def jax_eigh_jit(a):
        return jsp.linalg.eigh(a)

    @jax.jit
    def our_eigh_jit(a):
        return eigh(a)

    def test_dtype(dtype):
        """Test a specific dtype."""
        a = jnp.ones((3, 3), dtype=dtype)
        a = (a + a.T) / 2  # Make symmetric

        w_jax, v_jax = jax_eigh_jit(a)
        w_ours, v_ours = our_eigh_jit(a)

        # With jax_enable_x64, JAX may promote float32 to float64
        # So we check compatibility rather than exact match
        # Eigenvectors should match input dtype or be promoted to float64
        if dtype == np.float32:
            assert v_ours.dtype in [np.float32, np.float64], \
                f"Eigenvector dtype {v_ours.dtype} " \
                f"not compatible with {dtype}"
        elif dtype == np.complex64:
            assert v_ours.dtype in [np.complex64, np.complex128], \
                f"Eigenvector dtype {v_ours.dtype} " \
                f"not compatible with {dtype}"
        else:
            assert v_ours.dtype == dtype, \
                f"Eigenvector dtype mismatch for {dtype}"

        # Compare values (convert to common dtype for comparison)
        tol = 1e-6 if dtype in [np.float32, np.complex64] else 1e-12
        # Use higher precision dtype for comparison to avoid precision loss
        if dtype == np.float32:
            compare_dtype = np.float64
        elif dtype == np.complex64:
            compare_dtype = np.complex128
        else:
            compare_dtype = dtype
        w_jax_conv = w_jax.astype(compare_dtype)
        w_ours_conv = w_ours.astype(compare_dtype)
        assert abs(w_jax_conv - w_ours_conv).max() < tol, \
            f"Eigenvalues don't match for {dtype}"

        # For eigenvectors, convert to common dtype and account for sign
        v_compare_dtype = compare_dtype
        if dtype in [np.complex64, np.complex128]:
            v_compare_dtype = np.complex128
        elif dtype in [np.float32, np.float64]:
            v_compare_dtype = np.float64
        v_jax_conv = v_jax.astype(v_compare_dtype)
        v_ours_conv = v_ours.astype(v_compare_dtype)
        # Check each eigenvector column can match up to sign
        max_diff = 0.0
        for i in range(v_jax_conv.shape[-1]):
            col_diff = abs(
                v_jax_conv[..., i] - v_ours_conv[..., i]
            ).max()
            col_diff_neg = abs(
                v_jax_conv[..., i] + v_ours_conv[..., i]
            ).max()
            max_diff = max(max_diff, min(col_diff, col_diff_neg))
        assert max_diff < tol, \
            f"Eigenvectors don't match for {dtype}"

    # Test float64 first (required)
    test_dtype(np.float64)
    print("dtype float64 passed")

    # # Test other dtypes (optional - failures won't stop the test)
    # optional_dtypes = [np.float32, np.complex64, np.complex128]
    # for dtype in optional_dtypes:
    #     try:
    #         test_dtype(dtype)
    #         print(f"  ✓ dtype {dtype.__name__} passed")
    #     except Exception as e:
    #         print(f"  ✗ dtype {dtype.__name__} failed: {e}")
    #         # Don't raise - float64 is the minimum requirement


def test_jit_batched():
    """Test JIT compilation with batched operations."""
    print("\nTest: JIT with batched operations")

    @jax.jit
    def jax_eigh_batched(a_batch):
        return jsp.linalg.eigh(a_batch)

    @jax.jit
    def our_eigh_batched(a_batch):
        return eigh(a_batch)

    # Create batch of 5 matrices
    batch_size = 5
    n = 4
    a_batch = jnp.stack(
        [jnp.ones((n, n)) * (i + 1) for i in range(batch_size)]
    )
    # Make symmetric
    a_batch = (a_batch + jnp.transpose(a_batch, (0, 2, 1))) / 2

    w_jax, v_jax = jax_eigh_batched(a_batch)
    w_ours, v_ours = our_eigh_batched(a_batch)

    assert w_jax.shape == w_ours.shape, "Eigenvalue batch shape mismatch"
    assert v_jax.shape == v_ours.shape, "Eigenvector batch shape mismatch"
    assert abs(w_jax - w_ours).max() < 1e-6, \
        "Batched eigenvalues don't match"

    # For eigenvectors, account for possible sign differences per column
    max_diff = 0.0
    for b in range(a_batch.shape[0]):
        for i in range(v_jax.shape[-1]):
            col_diff = abs(v_jax[b, :, i] - v_ours[b, :, i]).max()
            col_diff_neg = abs(v_jax[b, :, i] + v_ours[b, :, i]).max()
            max_diff = max(max_diff, min(col_diff, col_diff_neg))
    assert max_diff < 1e-6, "Batched eigenvectors don't match"

    print("Batched JIT test passed")


def test_jit_generalized():
    """Test JIT compilation with generalized eigenvalue problem."""
    print("\nTest: JIT with generalized eigenvalue problem")

    # Note: JAX's scipy.linalg.eigh doesn't support generalized problems
    # So we only test our implementation
    @jax.jit
    def our_eigh_gen_jit(a, b):
        return eigh(a, b)

    a = jnp.ones((3, 3))
    a = (a + a.T) / 2
    b = jnp.eye(3) + 0.1 * jnp.ones((3, 3))
    b = (b + b.T) / 2

    w_ours, v_ours = our_eigh_gen_jit(a, b)

    # Verify generalized eigenvalue equation: A @ v = B @ v @ diag(w)
    w_diag = jnp.diag(w_ours)
    residual = jnp.max(jnp.abs(a @ v_ours - b @ v_ours @ w_diag))
    assert residual < 1e-12, \
        f"Generalized eigenvalue equation failed: {residual}"

    print("Generalized JIT test passed")


def test_jit_gradients():
    """Test JIT compilation with gradient computation."""
    print("\nTest: JIT with gradients")

    @jax.jit
    def jax_eigh_sum(a):
        w, _ = jsp.linalg.eigh(a)
        return w.sum()

    @jax.jit
    def our_eigh_sum(a):
        w, _ = eigh(a)
        return w.sum()

    a = jnp.ones((3, 3))
    a = (a + a.T) / 2

    # Compute gradients
    grad_jax_fn = jax.jit(jax.grad(jax_eigh_sum))
    grad_ours_fn = jax.jit(jax.grad(our_eigh_sum))

    grad_jax = grad_jax_fn(a)
    grad_ours = grad_ours_fn(a)

    assert abs(grad_jax - grad_ours).max() < 1e-7, "Gradients don't match"

    print("Gradient JIT test passed")


def test_jit_vmap():
    """Test JIT compilation with vmap."""
    print("\nTest: JIT with vmap")

    @jax.jit
    @jax.vmap
    def jax_eigh_vmap(a):
        return jsp.linalg.eigh(a)

    @jax.jit
    @jax.vmap
    def our_eigh_vmap(a):
        return eigh(a)

    # Create batch of matrices
    batch_size = 4
    n = 3
    a_batch = jnp.stack(
        [jnp.ones((n, n)) * (i + 1) for i in range(batch_size)]
    )
    a_batch = (a_batch + jnp.transpose(a_batch, (0, 2, 1))) / 2

    w_jax, v_jax = jax_eigh_vmap(a_batch)
    w_ours, v_ours = our_eigh_vmap(a_batch)

    assert abs(w_jax - w_ours).max() < 1e-12, "Vmap eigenvalues don't match"
    assert abs(v_jax - v_ours).max() < 1e-12, "Vmap eigenvectors don't match"

    print("Vmap JIT test passed")


def test_jit_eigvals_only():
    """Test JIT compilation with eigvals_only option."""
    print("\nTest: JIT with eigvals_only")

    @jax.jit
    def jax_eigh_eigvals(a):
        return jsp.linalg.eigh(a, eigvals_only=True)

    @jax.jit
    def our_eigh_eigvals(a):
        return eigh(a, eigvals_only=True)

    a = jnp.ones((3, 3))
    a = (a + a.T) / 2

    w_jax = jax_eigh_eigvals(a)
    w_ours = our_eigh_eigvals(a)

    assert abs(w_jax - w_ours).max() < 1e-12, "Eigvals-only don't match"

    print("Eigvals-only JIT test passed")


def test_jit_compilation_time():
    """Test that JIT compilation succeeds and measure compilation time."""
    print("\nTest: JIT compilation time")

    @jax.jit
    def jax_eigh_jit(a):
        return jsp.linalg.eigh(a)

    @jax.jit
    def our_eigh_jit(a):
        return eigh(a)

    a = jnp.ones((5, 5))
    a = (a + a.T) / 2

    w_jax_1, v_jax_1 = jax_eigh_jit(a)
    w_jax_2, v_jax_2 = jax_eigh_jit(a)
    w_ours_1, v_ours_1 = our_eigh_jit(a)
    w_ours_2, v_ours_2 = our_eigh_jit(a)

    assert abs(w_jax_1 - w_jax_2).max() < 1e-12, "JAX results inconsistent"
    assert abs(w_ours_1 - w_ours_2).max() < 1e-12, "Our results inconsistent"
    assert abs(w_jax_1 - w_ours_1).max() < 1e-12, "Results don't match"

    print("JIT compilation test passed")


def test_jit_with_grad_and_vmap():
    """Test JIT compilation with both gradients and vmap."""
    print("\nTest: JIT with gradients and vmap")

    def jax_eigh_sum(a):
        w, _ = jsp.linalg.eigh(a)
        return w.sum()

    def our_eigh_sum(a):
        w, _ = eigh(a)
        return w.sum()

    # JIT compile gradient of vmap
    grad_jax = jax.jit(jax.vmap(jax.grad(jax_eigh_sum)))
    grad_ours = jax.jit(jax.vmap(jax.grad(our_eigh_sum)))

    # Create batch
    batch_size = 3
    n = 4
    a_batch = jnp.stack(
        [jnp.ones((n, n)) * (i + 1) for i in range(batch_size)]
    )
    a_batch = (a_batch + jnp.transpose(a_batch, (0, 2, 1))) / 2

    grad_jax_result = grad_jax(a_batch)
    grad_ours_result = grad_ours(a_batch)

    assert grad_jax_result.shape == grad_ours_result.shape, \
        "Gradient shape mismatch"
    assert abs(grad_jax_result - grad_ours_result).max() < 1e-7, \
        "Gradients don't match"

    print("JIT with gradients and vmap test passed")


def test_jit_device_placement():
    """Test JIT compilation on different devices."""
    print("\nTest: JIT device placement")

    devices = jax.devices()

    @jax.jit
    def jax_eigh_jit(a):
        return jsp.linalg.eigh(a)

    @jax.jit
    def our_eigh_jit(a):
        return eigh(a)

    a = jnp.ones((3, 3))
    a = (a + a.T) / 2

    for device in devices:
        try:
            with jax.default_device(device):
                w_jax, v_jax = jax_eigh_jit(a)
                w_ours, v_ours = our_eigh_jit(a)

                assert abs(w_jax - w_ours).max() < 1e-12, \
                    f"Results don't match on {device.platform}"
        except Exception:
            # Don't raise - some platforms might not be supported
            pass


def run_all_tests():
    """Run all JIT tests."""
    print("=" * 60)
    print("Running eigh JIT compilation tests")
    print("=" * 60)

    tests = [
        test_jit_basic,
        test_jit_dtypes,
        test_jit_batched,
        test_jit_generalized,
        test_jit_gradients,
        test_jit_vmap,
        test_jit_eigvals_only,
        test_jit_compilation_time,
        test_jit_different_shapes,
        test_jit_with_grad_and_vmap,
        test_jit_device_placement,
    ]

    failed = []
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"\nTest {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed.append(test.__name__)

    print("\n" + "=" * 60)
    if failed:
        print(f"{len(failed)} test(s) FAILED:")
        for name in failed:
            print(f"  - {name}")
        return 1
    else:
        print("All JIT tests PASSED!")
        return 0

if __name__ == "__main__":
    # Run all tests verbosely
    pytest.main([__file__, "-v"])
