"""Tests for the JAX generalized eigensolvers in src/jax/.

Covers forward-pass correctness (AV = BVΛ, V^T B V = I) and gradient stability
(no NaNs, including near-degenerate spectra) for:
  - safe_generalized_eigh
  - subspace_generalized_eigh
  - stable_generalized_eigh
  - stable_eigh_gen_pyscfad
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.jax.generalized_eigensolver import (  # noqa: E402
    safe_generalized_eigh,
    subspace_generalized_eigh,
)
from src.jax.generalized_eigensolver_stable import stable_generalized_eigh  # noqa: E402
from src.jax.generalized_eigensolver_pyscfad import stable_eigh_gen_pyscfad  # noqa: E402


SOLVERS = [
    ("safe", safe_generalized_eigh),
    ("subspace", subspace_generalized_eigh),
    ("stable", stable_generalized_eigh),
    ("pyscfad", stable_eigh_gen_pyscfad),
]


def _make_problem(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, n))
    A = jnp.asarray(0.5 * (a + a.T))
    b = rng.standard_normal((n, n))
    B = jnp.asarray(b @ b.T + n * np.eye(n))
    return A, B


def _assert_generalized_eig(A, B, w, V, tol=1e-8):
    # AV = BVΛ
    lhs = A @ V
    rhs = B @ V @ jnp.diag(w)
    assert jnp.allclose(lhs, rhs, atol=tol, rtol=tol), "AV != BVΛ"
    # Eigenvalues sorted ascending
    assert jnp.all(jnp.diff(w) >= -1e-9)


@pytest.mark.parametrize("name,solver", SOLVERS)
@pytest.mark.parametrize("n", [4, 10])
def test_forward_correctness(name, solver, n):
    A, B = _make_problem(n, seed=n)
    w, V = solver(A, B)
    assert w.shape == (n,)
    assert V.shape == (n, n)
    _assert_generalized_eig(A, B, w, V)


@pytest.mark.parametrize("name,solver", SOLVERS)
def test_eigenvectors_B_orthonormal(name, solver):
    A, B = _make_problem(6, seed=42)
    _, V = solver(A, B)
    M = V.T @ B @ V
    assert jnp.allclose(M, jnp.eye(V.shape[0]), atol=1e-6)


@pytest.mark.parametrize("name,solver", SOLVERS)
def test_matches_scipy(name, solver):
    import scipy.linalg
    A, B = _make_problem(8, seed=7)
    w_ref, _ = scipy.linalg.eigh(np.asarray(A), np.asarray(B))
    w, _ = solver(A, B)
    assert jnp.allclose(jnp.sort(w), jnp.sort(jnp.asarray(w_ref)), atol=1e-8)


@pytest.mark.parametrize("name,solver", SOLVERS)
def test_gradient_no_nan(name, solver):
    A, B = _make_problem(5, seed=1)

    def loss(A):
        w, _ = solver(A, B)
        return jnp.sum(w ** 2)

    g = jax.grad(loss)(A)
    assert not jnp.any(jnp.isnan(g)), f"{name} produced NaN gradients"
    assert not jnp.any(jnp.isinf(g)), f"{name} produced Inf gradients"


@pytest.mark.parametrize("name,solver", SOLVERS)
def test_gradient_near_degenerate(name, solver):
    """Eigenvalue sum gradient is well-defined even at exact degeneracy."""
    n = 4
    B = jnp.eye(n)
    # A with repeated eigenvalues
    Q, _ = jnp.linalg.qr(jnp.asarray(np.random.default_rng(0).standard_normal((n, n))))
    A = Q @ jnp.diag(jnp.array([1.0, 1.0, 2.0, 2.0])) @ Q.T
    A = 0.5 * (A + A.T)

    def loss(A):
        w, _ = solver(A, B)
        return jnp.sum(w)

    g = jax.grad(loss)(A)
    assert not jnp.any(jnp.isnan(g)), f"{name} NaN on degenerate spectrum"


@pytest.mark.parametrize("name,solver", SOLVERS)
def test_jit(name, solver):
    A, B = _make_problem(6, seed=3)
    jitted = jax.jit(solver)
    w, V = jitted(A, B)
    _assert_generalized_eig(A, B, w, V)
