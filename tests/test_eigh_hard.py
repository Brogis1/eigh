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

"""Hard case comparison: our eigh vs JAX eigh."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'python'))

import numpy as np
import jax
import jax.numpy as jnp
from jax import scipy as jsp
from eigh import eigh
import pytest

jax.config.update("jax_enable_x64", True)


def construct_symmetric(eigvals, random_seed=42):
    """A = Q @ diag(eigvals) @ Q.T with random orthogonal Q."""
    np.random.seed(random_seed)
    n = len(eigvals)
    Q, _ = np.linalg.qr(np.random.randn(n, n))
    Q = jnp.array(Q)
    D = jnp.diag(jnp.array(eigvals))
    A = Q @ D @ Q.T
    return (A + A.T) / 2


def test_ill_conditioned():
    """Ill-conditioned: eigenvalues span 16 orders of magnitude."""
    eigvals = [1e-8, 1e-4, 1.0, 1e4, 1e8]
    A = construct_symmetric(eigvals)

    w_jax, v_jax = jsp.linalg.eigh(A)
    w_ours, v_ours = eigh(A)

    assert jnp.allclose(jnp.sort(w_jax), jnp.sort(w_ours), rtol=1e-6)
    assert jnp.max(jnp.abs(A @ v_ours - v_ours @ jnp.diag(w_ours))) < 1e-6


def test_nearly_degenerate():
    """Nearly degenerate: eigenvalue spacing 1e-10."""
    eigvals = [1.0, 1.0 + 1e-10, 2.0, 2.0 + 1e-10]
    A = construct_symmetric(eigvals)

    w_jax, v_jax = jsp.linalg.eigh(A)
    w_ours, v_ours = eigh(A, deg_thresh=1e-8)

    assert jnp.allclose(jnp.sort(w_jax), jnp.sort(w_ours), atol=1e-8)
    assert jnp.max(jnp.abs(A @ v_ours - v_ours @ jnp.diag(w_ours))) < 1e-8


def test_exact_degenerate_double():
    """Exact degeneracy: [1, 1, 2, 2]."""
    A = construct_symmetric([1.0, 1.0, 2.0, 2.0])

    w_jax, v_jax = jsp.linalg.eigh(A)
    w_ours, v_ours = eigh(A)

    assert jnp.allclose(jnp.sort(w_jax), jnp.sort(w_ours), atol=1e-10)
    assert jnp.max(jnp.abs(A @ v_ours - v_ours @ jnp.diag(w_ours))) < 1e-10


def test_exact_degenerate_triple():
    """Exact degeneracy: [1, 1, 1, 5]."""
    A = construct_symmetric([1.0, 1.0, 1.0, 5.0])

    w_jax, v_jax = jsp.linalg.eigh(A)
    w_ours, v_ours = eigh(A)

    assert jnp.allclose(jnp.sort(w_jax), jnp.sort(w_ours), atol=1e-10)
    assert jnp.max(jnp.abs(A @ v_ours - v_ours @ jnp.diag(w_ours))) < 1e-10


def test_all_degenerate():
    """All equal: identity matrix."""
    A = jnp.eye(4)

    w_jax, v_jax = jsp.linalg.eigh(A)
    w_ours, v_ours = eigh(A)

    assert jnp.allclose(w_jax, w_ours, atol=1e-12)
    assert jnp.max(jnp.abs(A @ v_ours - v_ours @ jnp.diag(w_ours))) < 1e-12


def test_zero_eigenvalue():
    """Zero eigenvalue in spectrum."""
    eigvals = [0.0, 1.0, 2.0, 3.0]
    A = construct_symmetric(eigvals)

    w_jax, v_jax = jsp.linalg.eigh(A)
    w_ours, v_ours = eigh(A)

    assert jnp.allclose(jnp.sort(w_jax), jnp.sort(w_ours), atol=1e-10)
    assert jnp.max(jnp.abs(A @ v_ours - v_ours @ jnp.diag(w_ours))) < 1e-9


def test_negative_eigenvalues():
    """Negative eigenvalues."""
    eigvals = [-5.0, -1.0, 1.0, 5.0]
    A = construct_symmetric(eigvals)

    w_jax, v_jax = jsp.linalg.eigh(A)
    w_ours, v_ours = eigh(A)

    assert jnp.allclose(jnp.sort(w_jax), jnp.sort(w_ours), atol=1e-10)
    assert jnp.max(jnp.abs(A @ v_ours - v_ours @ jnp.diag(w_ours))) < 1e-10


def test_rank_deficient():
    """Rank deficient: multiple zero eigenvalues."""
    eigvals = [0.0, 0.0, 1.0, 2.0]
    A = construct_symmetric(eigvals)

    w_jax, v_jax = jsp.linalg.eigh(A)
    w_ours, v_ours = eigh(A)

    assert jnp.allclose(jnp.sort(w_jax), jnp.sort(w_ours), atol=1e-10)
    assert jnp.max(jnp.abs(A @ v_ours - v_ours @ jnp.diag(w_ours))) < 1e-10


def test_gradient_trace():
    """Gradient d/dA[tr(A)] = I for both implementations."""
    A = construct_symmetric([1.0, 2.0, 3.0])

    def trace_jax(A):
        w, _ = jsp.linalg.eigh(A)
        return w.sum()

    def trace_ours(A):
        w, _ = eigh(A)
        return w.sum()

    grad_jax = jax.grad(trace_jax)(A)
    grad_ours = jax.grad(trace_ours)(A)

    assert jnp.allclose(grad_jax, jnp.eye(3), atol=1e-8)
    assert jnp.allclose(grad_ours, jnp.eye(3), atol=1e-8)
    assert jnp.allclose(grad_jax, grad_ours, atol=1e-8)


def test_gradient_symmetric_degenerate():
    """Gradient of symmetric function with degeneracy."""
    A = construct_symmetric([1.0, 1.0, 2.0, 2.0])

    def fn_jax(A):
        w, _ = jsp.linalg.eigh(A)
        return jnp.sum(w**2)

    def fn_ours(A):
        w, _ = eigh(A)
        return jnp.sum(w**2)

    grad_jax = jax.grad(fn_jax)(A)
    grad_ours = jax.grad(fn_ours)(A)

    eps = 1e-6
    direction = np.random.randn(4, 4)
    direction = (direction + direction.T) / 2
    direction = jnp.array(direction) / jnp.linalg.norm(direction)

    fd = (fn_ours(A + eps * direction) - fn_ours(A - eps * direction)) / (2 * eps)
    analytical = jnp.sum(grad_ours * direction)

    assert jnp.abs(fd - analytical) / (jnp.abs(fd) + 1e-10) < 1e-4
    assert jnp.allclose(grad_jax, grad_ours, atol=1e-6)


def test_gradient_ill_conditioned():
    """Gradient with ill-conditioned matrix."""
    A = construct_symmetric([1e-6, 1.0, 1e6])

    def fn_jax(A):
        w, _ = jsp.linalg.eigh(A)
        return jnp.sum(w**2)

    def fn_ours(A):
        w, _ = eigh(A)
        return jnp.sum(w**2)

    grad_jax = jax.grad(fn_jax)(A)
    grad_ours = jax.grad(fn_ours)(A)

    eps = 1e-5
    direction = np.random.randn(3, 3)
    direction = (direction + direction.T) / 2
    direction = jnp.array(direction) / jnp.linalg.norm(direction)

    fd = (fn_ours(A + eps * direction) - fn_ours(A - eps * direction)) / (2 * eps)
    analytical = jnp.sum(grad_ours * direction)

    assert jnp.abs(fd - analytical) / (jnp.abs(fd) + 1e-10) < 1e-3
    assert jnp.allclose(grad_jax, grad_ours, rtol=1e-4)


def test_generalized_ill_conditioned():
    """Generalized problem with ill-conditioned B."""
    A = construct_symmetric([1.0, 2.0, 3.0])
    B = construct_symmetric([1e-4, 1.0, 1e4], random_seed=43)

    w, v = eigh(A, B)

    assert jnp.max(jnp.abs(A @ v - B @ v @ jnp.diag(w))) < 1e-6


def test_generalized_degenerate():
    """Generalized problem with degenerate A."""
    A = construct_symmetric([1.0, 1.0, 2.0])
    B = jnp.eye(3)

    w, v = eigh(A, B)

    assert jnp.max(jnp.abs(A @ v - B @ v @ jnp.diag(w))) < 1e-10


def test_complex_hermitian():
    """Complex Hermitian matrix."""
    np.random.seed(42)
    n = 3
    M = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    A = jnp.array(M + M.conj().T) / 2

    w_jax, v_jax = jsp.linalg.eigh(A)
    w_ours, v_ours = eigh(A)

    assert jnp.allclose(jnp.sort(w_jax), jnp.sort(w_ours), atol=1e-10)
    assert jnp.max(jnp.abs(A @ v_ours - v_ours @ jnp.diag(w_ours))) < 1e-10


def test_complex_degenerate():
    """Complex Hermitian with degeneracy."""
    np.random.seed(42)
    n = 4
    Q, _ = np.linalg.qr(np.random.randn(n, n) + 1j * np.random.randn(n, n))
    Q = jnp.array(Q)
    D = jnp.diag(jnp.array([1.0, 1.0, 2.0, 2.0]))
    A = Q @ D @ Q.conj().T
    A = (A + A.conj().T) / 2

    w_jax, v_jax = jsp.linalg.eigh(A)
    w_ours, v_ours = eigh(A)

    assert jnp.allclose(jnp.sort(w_jax), jnp.sort(w_ours), atol=1e-10)
    assert jnp.max(jnp.abs(A @ v_ours - v_ours @ jnp.diag(w_ours))) < 1e-10


def test_batched_mixed():
    """Batch with varied conditioning."""
    A_batch = jnp.stack([
        construct_symmetric([1.0, 2.0], random_seed=i)
        for i in range(3)
    ])

    w_jax, v_jax = jsp.linalg.eigh(A_batch)
    w_ours, v_ours = eigh(A_batch)

    for i in range(3):
        assert jnp.allclose(jnp.sort(w_jax[i]), jnp.sort(w_ours[i]), atol=1e-10)
        residual = jnp.max(jnp.abs(A_batch[i] @ v_ours[i] - v_ours[i] @ jnp.diag(w_ours[i])))
        assert residual < 1e-10


if __name__ == "__main__":
    # Run all tests verbosely
    pytest.main([__file__, "-v"])
