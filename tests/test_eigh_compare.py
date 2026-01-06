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
Comparisons against standard JAX eigh (including generalized via reduction).
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

jax.config.update("jax_enable_x64", True)

try:
    from eigh import eigh
except ImportError as e:
    print(f"✗ Failed to import eigh module: {e}")
    sys.exit(1)


def _eigvec_max_diff(v_ref, v_test):
    """Compute max difference between eigenvectors up to a global phase/sign."""
    inner = jnp.sum(jnp.conj(v_ref) * v_test, axis=-2)
    denom = jnp.where(jnp.abs(inner) > 0, jnp.abs(inner), 1.0)
    phase = inner / denom
    aligned = v_test * phase[..., None, :]
    return jnp.max(jnp.abs(v_ref - aligned))


def test_compare_against_jax():
    """Compare our eigh against JAX for standard (non-generalized) problems."""
    print("\nTest: Compare against JAX (standard)")
    rng = np.random.default_rng(42)
    shapes = [2, 3, 5]
    dtypes = [np.float32, np.float64] # np.complex64, np.complex128 fail
    # WARNING: complex dtypes are not supported yet
    # print a warning message
    print("WARNING: complex dtypes are not supported yet")

    for n in shapes:
        for dtype in dtypes:
            a = rng.standard_normal((n, n)).astype(dtype)
            if np.issubdtype(dtype, np.complexfloating):
                a = a + 1j * rng.standard_normal((n, n)).astype(dtype)
            a = (a + a.swapaxes(-1, -2).conj()) / 2
            a = jnp.array(a)

            w_ref, v_ref = jsp.linalg.eigh(a)
            w_ours, v_ours = eigh(a)

            w_ref_sorted = jnp.sort(w_ref)
            w_ours_sorted = jnp.sort(w_ours)

            tol = 1e-6 if dtype in [np.float32, np.complex64] else 1e-12
            assert jnp.max(jnp.abs(w_ref_sorted - w_ours_sorted)) < tol, \
                f"Eigenvalues mismatch for n={n}, dtype={dtype}, w_ref_sorted: {w_ref_sorted}, w_ours_sorted: {w_ours_sorted}"

            diff = _eigvec_max_diff(v_ref, v_ours)
            assert diff < tol, \
                f"Eigenvectors mismatch for n={n}, dtype={dtype}, diff={diff}, v_ref: {v_ref}, v_ours: {v_ours}"

            residual = jnp.max(jnp.abs(a @ v_ours - v_ours @ jnp.diag(w_ours)))
            assert residual < tol * 10, \
                f"Eigen decomposition residual too large: {residual}"

    print("Standard comparison against JAX passed")


def test_compare_batched_against_jax():
    """Compare batched eigh against JAX for standard problems."""
    print("\nTest: Compare batched against JAX (standard)")

    rng = np.random.default_rng(123)
    batch = 4
    n = 4
    dtype = np.float64

    a = rng.standard_normal((batch, n, n)).astype(dtype)
    a = (a + a.swapaxes(-1, -2)) / 2
    a = jnp.array(a)

    w_ref, v_ref = jsp.linalg.eigh(a)
    w_ours, v_ours = eigh(a)

    w_ref_sorted = jnp.sort(w_ref, axis=-1)
    w_ours_sorted = jnp.sort(w_ours, axis=-1)
    tol = 1e-12
    assert jnp.max(jnp.abs(w_ref_sorted - w_ours_sorted)) < tol, \
        "Batched eigenvalues mismatch"

    diff = _eigvec_max_diff(v_ref, v_ours)
    assert diff < tol, f"Batched eigenvectors mismatch, diff={diff}"

    w_diag = jnp.vectorize(jnp.diag, signature="(n)->(n,n)")(w_ours)
    residual = jnp.max(jnp.abs(a @ v_ours - v_ours @ w_diag))
    assert residual < tol * 10, f"Batched residual too large: {residual}"

    print("Batched comparison against JAX passed")


def test_compare_generalized_against_jax():
    """Compare generalized eigh by reducing to standard problem."""
    print("\nTest: Compare generalized against JAX (via reduction)")

    rng = np.random.default_rng(7)
    n = 4
    dtype = np.float64

    a = rng.standard_normal((n, n)).astype(dtype)
    a = (a + a.T) / 2
    b = rng.standard_normal((n, n)).astype(dtype)
    b = (b + b.T) / 2
    b = b + n * np.eye(n, dtype=dtype)  # make PD

    a = jnp.array(a)
    b = jnp.array(b)

    w_ours, v_ours = eigh(a, b)

    L = jsp.linalg.cholesky(b, lower=True)
    Linv = jsp.linalg.solve_triangular(
        L, jnp.eye(n, dtype=dtype), lower=True
    )
    a_tilde = Linv @ a @ Linv.T.conj()
    w_ref, q = jsp.linalg.eigh(a_tilde)
    v_ref = jsp.linalg.solve_triangular(L.T.conj(), q, lower=False)

    w_ref_sorted = jnp.sort(w_ref)
    w_ours_sorted = jnp.sort(w_ours)
    tol = 1e-12
    assert jnp.max(jnp.abs(w_ref_sorted - w_ours_sorted)) < tol, \
        "Generalized eigenvalues mismatch"

    diff = _eigvec_max_diff(v_ref, v_ours)
    assert diff < tol, f"Generalized eigenvectors mismatch, diff={diff}"

    residual = jnp.max(jnp.abs(a @ v_ours - b @ v_ours @ jnp.diag(w_ours)))
    assert residual < tol * 10, f"Generalized residual too large: {residual}"

    print("Generalized comparison against JAX passed")


if __name__ == "__main__":
    # Run all tests verbosely
    pytest.main([__file__, "-v"])
